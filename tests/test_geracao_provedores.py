"""Testes unitários de OllamaGeracao e OpenAIGeracao com httpx.MockTransport."""

from __future__ import annotations

import httpx
import pytest

from pdf_rag_chatbot.erros import ErroChaveApi, ErroProvedor
from pdf_rag_chatbot.geracao import ProvedorGeracao
from pdf_rag_chatbot.geracao.ollama import OllamaGeracao
from pdf_rag_chatbot.geracao.openai import OpenAIGeracao

MENSAGENS = [{"role": "user", "content": "Olá"}]
CHAVE = "sk-segredo-123"


def _cliente(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _ollama(handler) -> OllamaGeracao:
    return OllamaGeracao(_cliente(handler), "http://localhost:11434/", "llama3.2")


def _openai(handler) -> OpenAIGeracao:
    return OpenAIGeracao(_cliente(handler), CHAVE, "gpt-4o-mini")


def test_ollama_envia_stream_false_e_devolve_texto() -> None:
    vistos: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        vistos.append(req)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok [1]"}})

    provedor = _ollama(handler)
    assert isinstance(provedor, ProvedorGeracao)
    assert provedor.completar(MENSAGENS, 60.0) == "ok [1]"
    assert vistos[0].url.path == "/api/chat"
    assert b'"stream":false' in vistos[0].content.replace(b" ", b"")


def test_openai_usa_chat_completions_e_devolve_texto() -> None:
    vistos: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        vistos.append(req)
        return httpx.Response(200, json={"choices": [{"message": {"content": "resp"}}]})

    assert _openai(handler).completar(MENSAGENS, 60.0) == "resp"
    assert vistos[0].url.path == "/v1/chat/completions"
    assert vistos[0].headers["Authorization"] == f"Bearer {CHAVE}"


@pytest.mark.parametrize("status", [401, 403])
def test_openai_401_403_viram_erro_chave_api(status: int) -> None:
    with pytest.raises(ErroChaveApi) as info:
        _openai(lambda r: httpx.Response(status)).completar(MENSAGENS, 60.0)
    assert CHAVE not in str(info.value)


@pytest.mark.parametrize("fabrica", [_ollama, _openai])
@pytest.mark.parametrize(
    "resposta",
    [
        httpx.Response(500),
        httpx.Response(200, content=b"nao e json"),
        httpx.Response(200, json={}),
        httpx.Response(200, json={"message": {"content": 5}, "choices": []}),
    ],
)
def test_falhas_viram_erro_provedor(fabrica, resposta: httpx.Response) -> None:
    with pytest.raises(ErroProvedor):
        fabrica(lambda r: resposta).completar(MENSAGENS, 60.0)


@pytest.mark.parametrize("fabrica", [_ollama, _openai])
def test_timeout_vira_erro_provedor(fabrica) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("lento", request=req)

    with pytest.raises(ErroProvedor):
        fabrica(handler).completar(MENSAGENS, 60.0)


def test_timeout_repassado_ao_httpx() -> None:
    capturado: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        capturado.append(req.extensions["timeout"])
        return httpx.Response(200, json={"message": {"content": "x"}})

    _ollama(handler).completar(MENSAGENS, 60.0)
    assert capturado[0]["read"] == 60.0


def test_repr_openai_nao_expoe_chave() -> None:
    assert CHAVE not in repr(_openai(lambda r: httpx.Response(200)))


TEXTO_PARCIAL = "texto parcial que nao deve vazar"


def _falha_sem_texto(fabrica, handler, excecao) -> BaseException:
    """Executa ``completar`` esperando falha e garante que nada foi devolvido."""
    resultado: list[str] = []
    with pytest.raises(excecao) as info:
        resultado.append(fabrica(handler).completar(MENSAGENS, 60.0))
    assert resultado == []
    return info.value


@pytest.mark.parametrize("fabrica", [_ollama, _openai])
def test_erro_500_com_corpo_parcial_nao_devolve_texto(fabrica) -> None:
    corpo = {"message": {"content": TEXTO_PARCIAL}, "error": TEXTO_PARCIAL}
    erro = _falha_sem_texto(
        fabrica, lambda r: httpx.Response(500, json=corpo), ErroProvedor
    )
    assert TEXTO_PARCIAL not in str(erro)


@pytest.mark.parametrize("fabrica", [_ollama, _openai])
def test_json_malformado_nao_devolve_texto(fabrica) -> None:
    erro = _falha_sem_texto(
        fabrica,
        lambda r: httpx.Response(200, content=TEXTO_PARCIAL.encode()),
        ErroProvedor,
    )
    assert TEXTO_PARCIAL not in str(erro)


@pytest.mark.parametrize("fabrica", [_ollama, _openai])
def test_timeout_nao_devolve_texto(fabrica) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("lento", request=req)

    _falha_sem_texto(fabrica, handler, ErroProvedor)


@pytest.mark.parametrize("status", [401, 403])
def test_openai_erro_chave_api_nao_devolve_texto_nem_expoe_chave(status: int) -> None:
    erro = _falha_sem_texto(
        _openai,
        lambda r: httpx.Response(status, json={"error": TEXTO_PARCIAL}),
        ErroChaveApi,
    )
    assert CHAVE not in str(erro)
    assert CHAVE not in repr(erro)
    assert TEXTO_PARCIAL not in str(erro)
    assert "OPENAI_API_KEY" in str(erro)


@pytest.mark.parametrize("status", [401, 403, 500])
def test_erros_openai_nao_expoem_chave(status: int) -> None:
    erro = _falha_sem_texto(_openai, lambda r: httpx.Response(status), BaseException)
    assert CHAVE not in str(erro)
    assert CHAVE not in repr(erro)


@pytest.mark.parametrize("status", [401, 403])
def test_ollama_401_403_viram_erro_provedor(status: int) -> None:
    erro = _falha_sem_texto(_ollama, lambda r: httpx.Response(status), ErroProvedor)
    assert not isinstance(erro, ErroChaveApi)
    assert str(status) in str(erro)
