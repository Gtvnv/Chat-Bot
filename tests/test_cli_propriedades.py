"""Testes de propriedade para a CLI e o reconhecimento de comandos."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import httpx
from hypothesis import given, settings
from hypothesis import strategies as st

from pdf_rag_chatbot import mensagens
from pdf_rag_chatbot.chatbot import Comando, reconhecer_comando
from pdf_rag_chatbot.cli import executar

_ESPACOS = st.text(alphabet=" \t\n\r", max_size=5)


@st.composite
def _variacao(draw, palavra: str) -> str:
    """Variação de caixa de `palavra` com espaços no início e no fim."""
    letras = [draw(st.sampled_from([c.lower(), c.upper()])) for c in palavra]
    return draw(_ESPACOS) + "".join(letras) + draw(_ESPACOS)


# Feature: pdf-rag-chatbot, Property 22: Reconhecimento de comandos
# Validates: Requirements 4.4, 4.6, 4.7
@settings(max_examples=100, deadline=None)
@given(entrada=_variacao("sair"))
def test_sair_reconhecido_em_qualquer_variacao(entrada: str) -> None:
    assert reconhecer_comando(entrada) is Comando.SAIR


# Feature: pdf-rag-chatbot, Property 22: Reconhecimento de comandos
# Validates: Requirements 4.4, 4.6, 4.7
@settings(max_examples=100, deadline=None)
@given(entrada=_variacao("documentos"))
def test_documentos_reconhecido_em_qualquer_variacao(entrada: str) -> None:
    assert reconhecer_comando(entrada) is Comando.DOCUMENTOS


# Feature: pdf-rag-chatbot, Property 22: Reconhecimento de comandos
# Validates: Requirements 4.4, 4.6, 4.7
@settings(max_examples=100, deadline=None)
@given(
    texto=st.text(max_size=60),
    sufixo=st.text(alphabet=st.characters(blacklist_categories=("Z", "C")), min_size=1, max_size=10),
    comando=st.sampled_from(["sair", "documentos"]),
)
def test_texto_diferente_do_comando_inteiro_e_pergunta(texto: str, sufixo: str, comando: str) -> None:
    # "sair" e "documentos" só valem como a entrada inteira.
    assert reconhecer_comando(f"{comando} {sufixo}") is None
    assert reconhecer_comando(f"{comando}{sufixo}") is None
    if texto.strip().casefold() not in {c.value for c in Comando} and not (
        texto.split(maxsplit=1)[:1] and texto.split(maxsplit=1)[0].casefold() in {"carregar", "topk"}
    ):
        assert reconhecer_comando(texto) is None

# ---------------------------------------------------------------------------
# Property 26: chave de API nunca aparece em saídas
# ---------------------------------------------------------------------------
_CHAVES = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
    min_size=12,
    max_size=40,
).map(lambda s: "sk-" + s)

_MODOS = ["401", "500_ecoando_chave", "excecao_com_chave", "embeddings_ok_geracao_401"]


class _Captura(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.textos: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        texto = self.format(record)
        self.textos.append(texto)


def _transporte(modo: str, chave: str, eventos: list[tuple[str, str]]) -> httpx.MockTransport:
    def tratar(requisicao: httpx.Request) -> httpx.Response:
        eventos.append(("requisicao", str(requisicao.url)))
        if modo == "excecao_com_chave":
            raise RuntimeError(f"falha interna com {chave}")
        if modo == "500_ecoando_chave":
            return httpx.Response(500, text=f"erro com a chave {chave}")
        if modo == "embeddings_ok_geracao_401" and requisicao.url.path.endswith("/embeddings"):
            import json

            entradas = json.loads(requisicao.content)["input"]
            dados = [{"index": i, "embedding": [1.0, 0.5, 0.25]} for i in range(len(entradas))]
            return httpx.Response(200, json={"data": dados})
        return httpx.Response(401, json={"error": {"message": f"chave {chave} inválida"}})

    return httpx.MockTransport(tratar)


# Feature: pdf-rag-chatbot, Property 26: Chave de API nunca aparece em saídas
# Validates: Requirements 5.6, 6.3, 6.4
@settings(max_examples=100, deadline=None)
@given(chave=_CHAVES, modo=st.sampled_from(_MODOS))
def test_chave_api_nunca_aparece_nas_saidas(chave: str, modo: str) -> None:
    eventos: list[tuple[str, str]] = []
    captura = _Captura()
    raiz = logging.getLogger()
    nivel_anterior = raiz.level
    raiz.addHandler(captura)
    raiz.setLevel(logging.DEBUG)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            documento = Path(tmp) / "doc.txt"
            documento.write_text("Texto de exemplo para indexação.\n" * 5, encoding="utf-8")
            env = {
                "PROVEDOR_EMBEDDINGS": "externo",
                "PROVEDOR_GERACAO": "externo",
                "OPENAI_API_KEY": chave,
                "DIRETORIO_INDICE": str(Path(tmp) / "index"),
            }
            entradas = iter([f'carregar "{documento}"', "documentos", "Qual é o tema?", "sair"])

            def saida(texto: str) -> None:
                eventos.append(("saida", texto))

            cliente = httpx.Client(transport=_transporte(modo, chave, eventos))
            try:
                executar(
                    [],
                    entrada=lambda _prompt: next(entradas),
                    saida=saida,
                    env=env,
                    arquivo_env=None,
                    cliente_http=cliente,
                )
            finally:
                cliente.close()
    finally:
        raiz.removeHandler(captura)
        raiz.setLevel(nivel_anterior)

    saidas = [t for tipo, t in eventos if tipo == "saida"]
    for texto in saidas + captura.textos:
        assert chave not in texto

    # O aviso identifica o provedor e vem antes da primeira requisição externa.
    aviso = mensagens.aviso_terceiros("OpenAI")
    assert "OpenAI" in aviso
    posicao_aviso = next(i for i, (t, x) in enumerate(eventos) if t == "saida" and x == aviso)
    primeira = next((i for i, (t, _) in enumerate(eventos) if t == "requisicao"), None)
    assert primeira is not None and posicao_aviso < primeira
