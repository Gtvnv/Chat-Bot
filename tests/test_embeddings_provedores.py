"""Testes unitários de OllamaEmbeddings e OpenAIEmbeddings com httpx.MockTransport."""

from __future__ import annotations

import httpx
import pytest

from pdf_rag_chatbot.embeddings import ProvedorEmbeddings
from pdf_rag_chatbot.embeddings.ollama import OllamaEmbeddings
from pdf_rag_chatbot.embeddings.openai import OpenAIEmbeddings
from pdf_rag_chatbot.erros import ErroChaveApi, ErroEmbeddings

TEXTOS = ["primeiro", "segundo"]
CHAVE = "sk-segredo-123"


def _cliente(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _ollama(handler) -> OllamaEmbeddings:
    return OllamaEmbeddings(_cliente(handler), "http://localhost:11434/", "nomic-embed-text")


def _openai(handler) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(_cliente(handler), CHAVE, "text-embedding-3-small")


def test_ollama_resposta_valida_e_requisicao_correta() -> None:
    vistos: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        vistos.append(req)
        return httpx.Response(200, json={"embeddings": [[1, 0.5], [0.0, 2.0]]})

    provedor = _ollama(handler)
    assert isinstance(provedor, ProvedorEmbeddings)
    assert provedor.embutir(TEXTOS) == [[1.0, 0.5], [0.0, 2.0]]
    assert vistos[0].url.path == "/api/embed"
    assert b'"model":"nomic-embed-text"' in vistos[0].content.replace(b" ", b"")


def test_openai_resposta_valida_ordena_por_index_e_envia_chave() -> None:
    vistos: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        vistos.append(req)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 2.0]},
                    {"index": 0, "embedding": [1.0, 0.5]},
                ]
            },
        )

    provedor = _openai(handler)
    assert isinstance(provedor, ProvedorEmbeddings)
    assert provedor.embutir(TEXTOS) == [[1.0, 0.5], [0.0, 2.0]]
    assert str(vistos[0].url) == "https://api.openai.com/v1/embeddings"
    assert vistos[0].headers["Authorization"] == f"Bearer {CHAVE}"


@pytest.mark.parametrize("fabrica", [_ollama, _openai])
def test_lista_vazia_nao_faz_requisicao(fabrica) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("não deveria haver requisição")

    assert fabrica(handler).embutir([]) == []


@pytest.mark.parametrize("status", [401, 403])
def test_openai_401_403_viram_erro_chave_api(status: int) -> None:
    with pytest.raises(ErroChaveApi) as info:
        _openai(lambda r: httpx.Response(status)).embutir(TEXTOS)
    assert CHAVE not in str(info.value)
    assert "OPENAI_API_KEY" in str(info.value)


@pytest.mark.parametrize("status", [401, 403])
def test_ollama_401_403_viram_erro_embeddings(status: int) -> None:
    with pytest.raises(ErroEmbeddings):
        _ollama(lambda r: httpx.Response(status)).embutir(TEXTOS)


@pytest.mark.parametrize("fabrica", [_ollama, _openai])
@pytest.mark.parametrize(
    "resposta",
    [
        httpx.Response(500),
        httpx.Response(200, content=b"nao e json"),
        httpx.Response(200, json=[]),
        httpx.Response(200, json={}),
        httpx.Response(200, json={"embeddings": [[1.0]], "data": [{"embedding": [1.0]}]}),
    ],
)
def test_falhas_viram_erro_embeddings(fabrica, resposta: httpx.Response) -> None:
    with pytest.raises(ErroEmbeddings):
        fabrica(lambda r: resposta).embutir(TEXTOS)


@pytest.mark.parametrize("fabrica", [_ollama, _openai])
def test_timeout_vira_erro_embeddings(fabrica) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("lento", request=req)

    with pytest.raises(ErroEmbeddings):
        fabrica(handler).embutir(TEXTOS)


def test_repr_openai_nao_expoe_chave() -> None:
    assert CHAVE not in repr(_openai(lambda r: httpx.Response(200)))
