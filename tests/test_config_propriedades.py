"""Testes de propriedade da configuração."""

from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from pdf_rag_chatbot.config import carregar_configuracao
from pdf_rag_chatbot.erros import ErroChaveApi, ErroConfiguracao

VALIDOS = ("local", "externo")
CHAVE_VALIDA = "chave-de-teste-123"

# Valor de variável de provedor: ausente (None), válido, ou texto arbitrário.
# O texto arbitrário exclui o que `strip()` reduz a um valor aceito, pois a
# configuração tolera espaços nas pontas de "local" e "externo".
valor_provedor = st.one_of(
    st.none(),
    st.sampled_from(VALIDOS),
    st.text(max_size=20).filter(lambda t: t.strip() not in VALIDOS),
    st.sampled_from(["Local", "EXTERNO", "locall", "remoto", "", "   ", "local,externo"]),
)


def _montar_env(embeddings: str | None, geracao: str | None) -> dict[str, str]:
    env = {"OPENAI_API_KEY": CHAVE_VALIDA}
    if embeddings is not None:
        env["PROVEDOR_EMBEDDINGS"] = embeddings
    if geracao is not None:
        env["PROVEDOR_GERACAO"] = geracao
    return env


# Feature: pdf-rag-chatbot, Property 23: Validação dos provedores configurados
# **Validates: Requirements 5.1, 5.2**
@settings(max_examples=200)
@given(embeddings=valor_provedor, geracao=valor_provedor)
def test_propriedade_23_validacao_dos_provedores(embeddings, geracao):
    env = _montar_env(embeddings, geracao)
    aceitos = embeddings in VALIDOS and geracao in VALIDOS

    with mock.patch.object(
        socket.socket, "connect", side_effect=AssertionError("requisição de rede")
    ) as conectar, mock.patch.object(
        socket, "create_connection", side_effect=AssertionError("requisição de rede")
    ) as criar_conexao:
        if aceitos:
            config = carregar_configuracao(env=env, arquivo_env=None)
            assert config.provedor_embeddings == embeddings
            assert config.provedor_geracao == geracao
        else:
            with pytest.raises(ErroConfiguracao) as info:
                carregar_configuracao(env=env, arquivo_env=None)
            mensagem = str(info.value)
            # A variável inválida é a primeira verificada: embeddings, depois geração.
            variavel = (
                "PROVEDOR_EMBEDDINGS"
                if embeddings not in VALIDOS
                else "PROVEDOR_GERACAO"
            )
            assert variavel in mensagem
            assert "local" in mensagem and "externo" in mensagem

        assert conectar.call_count == 0
        assert criar_conexao.call_count == 0


# Valores de host sem caracteres especiais do formato `.env` (aspas, `#`, `$`, espaços).
valor_host = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Nd"), max_codepoint=127),
    min_size=1,
    max_size=15,
).map(lambda t: f"http://{t}:11434")


# Feature: pdf-rag-chatbot, Property 24: Precedência do ambiente sobre o `.env`
# **Validates: Requirements 5.4**
@settings(max_examples=100)
@given(
    geracao_env=st.sampled_from(VALIDOS),
    geracao_arquivo=st.sampled_from(VALIDOS),
    host_env=valor_host,
    host_arquivo=valor_host,
    usar_os_environ=st.booleans(),
)
def test_propriedade_24_ambiente_prevalece_sobre_o_env(
    geracao_env, geracao_arquivo, host_env, host_arquivo, usar_os_environ
):
    assume(geracao_env != geracao_arquivo)
    assume(host_env != host_arquivo)

    ambiente = {
        "PROVEDOR_EMBEDDINGS": "local",
        "PROVEDOR_GERACAO": geracao_env,
        "OLLAMA_HOST": host_env,
        "OPENAI_API_KEY": CHAVE_VALIDA,
    }

    with tempfile.TemporaryDirectory() as diretorio:
        arquivo = Path(diretorio) / ".env"
        arquivo.write_text(
            f"PROVEDOR_GERACAO={geracao_arquivo}\nOLLAMA_HOST={host_arquivo}\n",
            encoding="utf-8",
        )

        if usar_os_environ:
            with mock.patch.dict(os.environ, ambiente, clear=True):
                config = carregar_configuracao(arquivo_env=arquivo)
        else:
            config = carregar_configuracao(env=ambiente, arquivo_env=arquivo)

    assert config.provedor_geracao == geracao_env
    assert config.ollama_host == host_env


# Chaves em branco: vazia ou composta apenas de espaços e outros brancos ASCII.
chave_em_branco = st.text(alphabet=" \t\n\r\x0b\x0c", max_size=20)


# Feature: pdf-rag-chatbot, Property 25: Chave de API em branco é rejeitada antes de qualquer requisição
# **Validates: Requirements 5.5**
@settings(max_examples=100)
@given(
    chave=chave_em_branco,
    embeddings=st.sampled_from(VALIDOS),
    geracao=st.sampled_from(VALIDOS),
)
def test_propriedade_25_chave_em_branco_e_rejeitada(chave, embeddings, geracao):
    assume("externo" in (embeddings, geracao))

    env = {
        "PROVEDOR_EMBEDDINGS": embeddings,
        "PROVEDOR_GERACAO": geracao,
        "OPENAI_API_KEY": chave,
    }

    with mock.patch.object(
        socket.socket, "connect", side_effect=AssertionError("requisição de rede")
    ) as conectar, mock.patch.object(
        socket, "create_connection", side_effect=AssertionError("requisição de rede")
    ) as criar_conexao:
        with pytest.raises(ErroChaveApi) as info:
            carregar_configuracao(env=env, arquivo_env=None)

        assert "OPENAI_API_KEY" in str(info.value)
        assert conectar.call_count == 0
        assert criar_conexao.call_count == 0


# ---------------------------------------------------------------------------
# Property 28: tempo limite e temperatura de geração
# ---------------------------------------------------------------------------

TIMEOUT_PADRAO = 60
TEMPERATURA_PADRAO = 0.2

# Valor de variável: ausente (None), vazio, inteiro (dentro e fora da faixa),
# decimal com ponto ou vírgula, "0,5" e texto arbitrário.
_texto_numerico = st.text(alphabet="0123456789.,-+eE _ax", max_size=8)
valor_numerico_bruto = st.one_of(
    st.none(),
    st.sampled_from(["", "   ", "0,5", "0.5", "0,2", "1,0", "1,01", "-0,1", "abc", "nan", "inf"]),
    st.integers(min_value=-50, max_value=1000).map(str),
    st.integers(min_value=10, max_value=600).map(str),
    st.integers(min_value=-5, max_value=5).map(str),
    st.floats(min_value=-2, max_value=3, allow_nan=False).map(repr),
    st.floats(min_value=0, max_value=1, allow_nan=False).map(
        lambda f: repr(f).replace(".", ",")
    ),
    _texto_numerico,
    st.text(max_size=10),
)


def _esperado_timeout(bruto: str | None) -> int | None:
    """Devolve o timeout esperado ou None quando deve ser rejeitado."""
    texto = (bruto or "").strip()
    if not texto:
        return TIMEOUT_PADRAO
    try:
        valor = int(texto)
    except ValueError:
        return None
    return valor if 10 <= valor <= 600 else None


def _esperada_temperatura(bruto: str | None) -> float | None:
    """Devolve a temperatura esperada ou None quando deve ser rejeitada."""
    texto = (bruto or "").strip()
    if not texto:
        return TEMPERATURA_PADRAO
    try:
        valor = float(texto.replace(",", "."))
    except ValueError:
        return None
    return valor if 0.0 <= valor <= 1.0 else None


def _env_geracao(timeout: str | None, temperatura: str | None) -> dict[str, str]:
    env = {"PROVEDOR_EMBEDDINGS": "local", "PROVEDOR_GERACAO": "local"}
    if timeout is not None:
        env["TIMEOUT_GERACAO_S"] = timeout
    if temperatura is not None:
        env["TEMPERATURA_GERACAO"] = temperatura
    return env


# Feature: pdf-rag-chatbot, Property 28: Tempo limite e temperatura de geração são validados
# **Validates: Requirements 3.10, 3.11, 3.13, 3.14**
@settings(max_examples=300)
@given(timeout=valor_numerico_bruto, temperatura=valor_numerico_bruto)
def test_propriedade_28_validacao_de_timeout_e_temperatura(timeout, temperatura):
    esperado_timeout = _esperado_timeout(timeout)
    esperada_temperatura = _esperada_temperatura(temperatura)
    env = _env_geracao(timeout, temperatura)

    with mock.patch.object(
        socket.socket, "connect", side_effect=AssertionError("requisição de rede")
    ) as conectar, mock.patch.object(
        socket, "create_connection", side_effect=AssertionError("requisição de rede")
    ) as criar_conexao:
        if esperado_timeout is not None and esperada_temperatura is not None:
            config = carregar_configuracao(env=env, arquivo_env=None)
            assert config.timeout_geracao_s == esperado_timeout
            assert config.temperatura_geracao == pytest.approx(esperada_temperatura)
        else:
            with pytest.raises(ErroConfiguracao) as info:
                carregar_configuracao(env=env, arquivo_env=None)
            mensagem = str(info.value)
            # O tempo limite é validado antes da temperatura.
            if esperado_timeout is None:
                assert "TIMEOUT_GERACAO_S" in mensagem
                assert "10" in mensagem and "600" in mensagem
            else:
                assert "TEMPERATURA_GERACAO" in mensagem
                assert "0" in mensagem and "1" in mensagem

        assert conectar.call_count == 0
        assert criar_conexao.call_count == 0


# Valores aceitos: inteiros da faixa e temperaturas com ponto ou vírgula.
_timeout_aceito = st.integers(min_value=10, max_value=600)
_temperatura_aceita = st.floats(min_value=0, max_value=1, allow_nan=False)


def _config_e_requisicoes(provedor: str, timeout: int, temperatura: float, usar_virgula: bool):
    import httpx

    from pdf_rag_chatbot.fabrica import criar_gerador_de_respostas

    texto_temperatura = repr(temperatura)
    if usar_virgula:
        texto_temperatura = texto_temperatura.replace(".", ",")
    env = {
        "PROVEDOR_EMBEDDINGS": "local",
        "PROVEDOR_GERACAO": provedor,
        "OPENAI_API_KEY": CHAVE_VALIDA,
        "TIMEOUT_GERACAO_S": str(timeout),
        "TEMPERATURA_GERACAO": texto_temperatura,
    }
    config = carregar_configuracao(env=env, arquivo_env=None)
    requisicoes: list[httpx.Request] = []

    def tratar(requisicao: httpx.Request) -> httpx.Response:
        requisicoes.append(requisicao)
        if requisicao.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    cliente = httpx.Client(transport=httpx.MockTransport(tratar))
    try:
        gerador = criar_gerador_de_respostas(config, cliente)
        gerador.gerar("Pergunta?", [], [])
    finally:
        cliente.close()
    return config, requisicoes


# Feature: pdf-rag-chatbot, Property 28: Tempo limite e temperatura de geração são validados
# **Validates: Requirements 3.10, 3.11, 3.13, 3.14**
@settings(max_examples=100)
@given(
    timeout=_timeout_aceito,
    temperatura=_temperatura_aceita,
    usar_virgula=st.booleans(),
)
def test_propriedade_28_valores_aceitos_chegam_ao_ollama(timeout, temperatura, usar_virgula):
    import json

    config, requisicoes = _config_e_requisicoes("local", timeout, temperatura, usar_virgula)

    assert config.timeout_geracao_s == timeout
    assert len(requisicoes) == 1
    requisicao = requisicoes[0]
    assert requisicao.url.path == "/api/chat"
    corpo = json.loads(requisicao.content)
    assert corpo["options"]["temperature"] == pytest.approx(temperatura)
    assert requisicao.extensions["timeout"]["read"] == timeout


# Feature: pdf-rag-chatbot, Property 28: Tempo limite e temperatura de geração são validados
# **Validates: Requirements 3.10, 3.11, 3.13, 3.14**
@settings(max_examples=100)
@given(
    timeout=_timeout_aceito,
    temperatura=_temperatura_aceita,
    usar_virgula=st.booleans(),
)
def test_propriedade_28_valores_aceitos_chegam_a_openai(timeout, temperatura, usar_virgula):
    import json

    config, requisicoes = _config_e_requisicoes("externo", timeout, temperatura, usar_virgula)

    assert config.timeout_geracao_s == timeout
    assert len(requisicoes) == 1
    requisicao = requisicoes[0]
    assert requisicao.url.path == "/v1/chat/completions"
    corpo = json.loads(requisicao.content)
    assert corpo["temperature"] == pytest.approx(temperatura)
    assert requisicao.extensions["timeout"]["read"] == timeout
