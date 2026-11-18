"""Provedor local de geração de respostas via API HTTP do Ollama."""

from __future__ import annotations

from typing import Sequence

import httpx

from pdf_rag_chatbot.erros import ErroProvedor
from pdf_rag_chatbot.geracao.prompt import Mensagem
from pdf_rag_chatbot.tipos import TempoLimite

TIMEOUT_PADRAO_S = 60.0
TEMPERATURA_PADRAO = 0.2


class OllamaGeracao:
    """Gera respostas com ``POST {host}/api/chat`` (``stream: false``)."""

    def __init__(
        self,
        cliente: httpx.Client,
        host: str,
        nome_modelo: str,
        timeout_s: TempoLimite = TIMEOUT_PADRAO_S,
        temperatura: float = TEMPERATURA_PADRAO,
    ) -> None:
        self._cliente = cliente
        self._url = f"{host.rstrip('/')}/api/chat"
        self.nome_modelo = nome_modelo
        self._timeout_s = timeout_s
        self._temperatura = temperatura

    def completar(
        self, mensagens: Sequence[Mensagem], timeout_s: TempoLimite | None = None
    ) -> str:
        timeout = self._timeout_s if timeout_s is None else timeout_s
        try:
            resposta = self._cliente.post(
                self._url,
                json={
                    "model": self.nome_modelo,
                    "messages": [dict(m) for m in mensagens],
                    "stream": False,
                    "options": {"temperature": self._temperatura},
                },
                timeout=timeout,
            )
            resposta.raise_for_status()
            dados = resposta.json()
        except httpx.TimeoutException as exc:
            raise ErroProvedor(
                "Tempo limite excedido ao gerar a resposta no Ollama."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ErroProvedor(
                f"O Ollama respondeu com erro HTTP {exc.response.status_code} "
                "ao gerar a resposta."
            ) from exc
        except httpx.HTTPError as exc:
            raise ErroProvedor(
                "Falha de comunicação com o Ollama ao gerar a resposta."
            ) from exc
        except ValueError as exc:
            raise ErroProvedor("Resposta do Ollama não é um JSON válido.") from exc

        mensagem = dados.get("message") if isinstance(dados, dict) else None
        conteudo = mensagem.get("content") if isinstance(mensagem, dict) else None
        if not isinstance(conteudo, str) or not conteudo.strip():
            raise ErroProvedor("Resposta do Ollama malformada.")
        return conteudo
