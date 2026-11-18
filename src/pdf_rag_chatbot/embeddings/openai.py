"""Provedor externo de embeddings via API REST da OpenAI."""

from __future__ import annotations

from typing import Sequence

import httpx

from pdf_rag_chatbot.embeddings.ollama import TIMEOUT_PADRAO_S, validar_vetores
from pdf_rag_chatbot.erros import ErroChaveApi, ErroEmbeddings

URL_EMBEDDINGS = "https://api.openai.com/v1/embeddings"
NOME_VARIAVEL_CHAVE = "OPENAI_API_KEY"


class OpenAIEmbeddings:
    """Gera embeddings com ``POST /v1/embeddings``."""

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
        return f"OpenAIEmbeddings(nome_modelo={self.nome_modelo!r})"

    def embutir(self, textos: Sequence[str]) -> list[list[float]]:
        textos = list(textos)
        if not textos:
            return []
        try:
            resposta = self._cliente.post(
                URL_EMBEDDINGS,
                headers={"Authorization": f"Bearer {self._chave_api}"},
                json={"model": self.nome_modelo, "input": textos},
                timeout=self._timeout_s,
            )
            if resposta.status_code in (401, 403):
                raise ErroChaveApi(NOME_VARIAVEL_CHAVE)
            resposta.raise_for_status()
            dados = resposta.json()
        except ErroChaveApi:
            raise
        except httpx.TimeoutException as exc:
            raise ErroEmbeddings(
                "Tempo limite excedido ao gerar embeddings na OpenAI."
            ) from None
        except httpx.HTTPStatusError as exc:
            raise ErroEmbeddings(
                f"A OpenAI respondeu com erro HTTP {exc.response.status_code} "
                "ao gerar embeddings."
            ) from None
        except httpx.HTTPError:
            raise ErroEmbeddings(
                "Falha de comunicação com a OpenAI ao gerar embeddings."
            ) from None
        except ValueError:
            raise ErroEmbeddings("Resposta da OpenAI não é um JSON válido.") from None

        itens = dados.get("data") if isinstance(dados, dict) else None
        if not isinstance(itens, list) or any(
            not isinstance(i, dict) for i in itens
        ):
            raise ErroEmbeddings("Resposta da OpenAI malformada.")
        if all(isinstance(i.get("index"), int) for i in itens):
            itens = sorted(itens, key=lambda i: i["index"])
        return validar_vetores([i.get("embedding") for i in itens], len(textos))
