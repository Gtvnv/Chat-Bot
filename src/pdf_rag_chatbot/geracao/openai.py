"""Provedor externo de geração de respostas via API REST da OpenAI."""

from __future__ import annotations

from typing import Sequence

import httpx

from pdf_rag_chatbot.erros import ErroChaveApi, ErroProvedor
from pdf_rag_chatbot.geracao.ollama import TIMEOUT_PADRAO_S
from pdf_rag_chatbot.geracao.prompt import Mensagem

URL_CHAT = "https://api.openai.com/v1/chat/completions"
NOME_VARIAVEL_CHAVE = "OPENAI_API_KEY"


class OpenAIGeracao:
    """Gera respostas com ``POST /v1/chat/completions``."""

    def __init__(
        self,
        cliente: httpx.Client,
        chave_api: str,
        nome_modelo: str,
        timeout_s: float = TIMEOUT_PADRAO_S,
    ) -> None:
        self._cliente = cliente
        self._chave_api = chave_api
        self.nome_modelo = nome_modelo
        self._timeout_s = timeout_s

    def __repr__(self) -> str:
        return f"OpenAIGeracao(nome_modelo={self.nome_modelo!r})"

    def completar(
        self, mensagens: Sequence[Mensagem], timeout_s: float | None = None
    ) -> str:
        timeout = self._timeout_s if timeout_s is None else timeout_s
        try:
            resposta = self._cliente.post(
                URL_CHAT,
                headers={"Authorization": f"Bearer {self._chave_api}"},
                json={
                    "model": self.nome_modelo,
                    "messages": [dict(m) for m in mensagens],
                },
                timeout=timeout,
            )
            if resposta.status_code in (401, 403):
                raise ErroChaveApi(NOME_VARIAVEL_CHAVE)
            resposta.raise_for_status()
            dados = resposta.json()
        except ErroChaveApi:
            raise
        except httpx.TimeoutException:
            raise ErroProvedor(
                "Tempo limite excedido ao gerar a resposta na OpenAI."
            ) from None
        except httpx.HTTPStatusError as exc:
            raise ErroProvedor(
                f"A OpenAI respondeu com erro HTTP {exc.response.status_code} "
                "ao gerar a resposta."
            ) from None
        except httpx.HTTPError:
            raise ErroProvedor(
                "Falha de comunicação com a OpenAI ao gerar a resposta."
            ) from None
        except ValueError:
            raise ErroProvedor("Resposta da OpenAI não é um JSON válido.") from None

        escolhas = dados.get("choices") if isinstance(dados, dict) else None
        primeira = escolhas[0] if isinstance(escolhas, list) and escolhas else None
        mensagem = primeira.get("message") if isinstance(primeira, dict) else None
        conteudo = mensagem.get("content") if isinstance(mensagem, dict) else None
        if not isinstance(conteudo, str) or not conteudo.strip():
            raise ErroProvedor("Resposta da OpenAI malformada.")
        return conteudo
