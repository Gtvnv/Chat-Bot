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
