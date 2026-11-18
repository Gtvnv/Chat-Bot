"""Interface de provedores de embeddings e gerador em lotes."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from pdf_rag_chatbot.erros import ErroEmbeddings

TAMANHO_LOTE_PADRAO = 64


@runtime_checkable
class ProvedorEmbeddings(Protocol):
    """Contrato de um provedor que converte textos em vetores."""

    nome_modelo: str

    def embutir(self, textos: Sequence[str]) -> list[list[float]]:
        """Devolve um vetor por texto, na mesma ordem da entrada."""
        ...


class GeradorDeEmbeddings:
    """Gera embeddings em lotes e valida quantidade e dimensão dos vetores."""

    def __init__(
        self,
        provedor: ProvedorEmbeddings,
        tamanho_lote: int = TAMANHO_LOTE_PADRAO,
    ) -> None:
        if isinstance(tamanho_lote, bool) or not isinstance(tamanho_lote, int) or tamanho_lote < 1:
            raise ValueError("tamanho_lote deve ser um inteiro maior ou igual a 1.")
        self._provedor = provedor
        self._tamanho_lote = tamanho_lote

    @property
    def nome_modelo(self) -> str:
        return self._provedor.nome_modelo

    def gerar(self, textos: Sequence[str]) -> np.ndarray:
        """Devolve matriz float32 de forma (n, d), um vetor por texto."""
        textos = list(textos)
        if not textos:
            return np.empty((0, 0), dtype=np.float32)

        vetores: list[list[float]] = []
        for inicio in range(0, len(textos), self._tamanho_lote):
            lote = textos[inicio : inicio + self._tamanho_lote]
            resultado = self._provedor.embutir(lote)
            if len(resultado) != len(lote):
                raise ErroEmbeddings(
                    f"O provedor devolveu {len(resultado)} vetores para "
                    f"{len(lote)} textos."
                )
            vetores.extend(resultado)

        dimensao = len(vetores[0])
        if dimensao == 0 or any(len(v) != dimensao for v in vetores):
            raise ErroEmbeddings("Os vetores devolvidos têm dimensões inconsistentes.")

        try:
            matriz = np.asarray(vetores, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ErroEmbeddings("Os vetores devolvidos não são numéricos.") from exc
        if matriz.shape != (len(textos), dimensao):
            raise ErroEmbeddings("Os vetores devolvidos têm formato inválido.")
        return matriz
