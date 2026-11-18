"""Provedor local de embeddings via API HTTP do Ollama."""

from __future__ import annotations

from typing import Any, Sequence

import httpx

from pdf_rag_chatbot.tipos import TempoLimite
from pdf_rag_chatbot.erros import ErroEmbeddings

TIMEOUT_PADRAO_S = 60.0


def validar_vetores(bruto: Any, esperado: int) -> list[list[float]]:
    """Valida o formato dos vetores recebidos e devolve listas de floats."""
    if not isinstance(bruto, list) or len(bruto) != esperado:
        raise ErroEmbeddings("Resposta do provedor de embeddings malformada.")
    vetores: list[list[float]] = []
    for item in bruto:
        if not isinstance(item, list) or not item:
            raise ErroEmbeddings("Resposta do provedor de embeddings malformada.")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in item):
            raise ErroEmbeddings("Resposta do provedor de embeddings malformada.")
        vetores.append([float(v) for v in item])
    return vetores


class OllamaEmbeddings:
    """Gera embeddings com ``POST {host}/api/embed``."""

    def __init__(
        self,
        cliente: httpx.Client,
        host: str,
        nome_modelo: str,
        timeout_s: TempoLimite = TIMEOUT_PADRAO_S,
    ) -> None:
        self._cliente = cliente
        self._url = f"{host.rstrip('/')}/api/embed"
        self.nome_modelo = nome_modelo
        self._timeout_s = timeout_s

    def embutir(self, textos: Sequence[str]) -> list[list[float]]:
        textos = list(textos)
        if not textos:
            return []
        try:
            resposta = self._cliente.post(
                self._url,
                json={"model": self.nome_modelo, "input": textos},
                timeout=self._timeout_s,
            )
            resposta.raise_for_status()
            dados = resposta.json()
        except httpx.TimeoutException as exc:
            raise ErroEmbeddings(
                "Tempo limite excedido ao gerar embeddings no Ollama."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ErroEmbeddings(
                f"O Ollama respondeu com erro HTTP {exc.response.status_code} "
                "ao gerar embeddings."
            ) from exc
        except httpx.HTTPError as exc:
            raise ErroEmbeddings(
                "Falha de comunicação com o Ollama ao gerar embeddings."
            ) from exc
        except ValueError as exc:
            raise ErroEmbeddings("Resposta do Ollama não é um JSON válido.") from exc

        if not isinstance(dados, dict):
            raise ErroEmbeddings("Resposta do Ollama malformada.")
        return validar_vetores(dados.get("embeddings"), len(textos))
