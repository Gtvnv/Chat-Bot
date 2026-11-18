"""Recuperação dos Trechos mais similares a uma pergunta (Recuperador)."""

from __future__ import annotations

import os
from collections.abc import Iterable

from pdf_rag_chatbot.config import validar_top_k
from pdf_rag_chatbot.embeddings.base import GeradorDeEmbeddings
from pdf_rag_chatbot.indice_vetorial import IndiceVetorial
from pdf_rag_chatbot.modelos import ResultadoBusca

LIMITE_TOTAL_COMPARATIVA = 10


def fontes_mencionadas(pergunta: str, fontes: Iterable[str]) -> list[str]:
    """Fontes cujo nome, com ou sem extensão, aparece na pergunta.

    A comparação ignora diferença entre maiúsculas e minúsculas. A ordem e a
    ausência de repetição seguem a de ``fontes``.
    """
    texto = pergunta.casefold()
    mencionadas: list[str] = []
    for fonte in fontes:
        if fonte in mencionadas:
            continue
        nome_completo = fonte.casefold()
        nome_sem_extensao = os.path.splitext(fonte)[0].casefold()
        if nome_completo in texto or (nome_sem_extensao and nome_sem_extensao in texto):
            mencionadas.append(fonte)
    return mencionadas


class Recuperador:
    """Busca no índice os Trechos mais similares à pergunta."""

    def __init__(
        self,
        embeddings: GeradorDeEmbeddings,
        indice: IndiceVetorial,
        top_k: int = 4,
    ) -> None:
        self._embeddings = embeddings
        self._indice = indice
        self._top_k = validar_top_k(top_k)

    @property
    def top_k(self) -> int:
        return self._top_k

    def definir_top_k(self, valor: object) -> None:
        """Atualiza o Top_K; se o valor é inválido, mantém o anterior."""
        self._top_k = validar_top_k(valor)  # ErroConfiguracao não altera o estado

    def recuperar(self, pergunta: str) -> list[ResultadoBusca]:
        """Devolve os Trechos relevantes, em ordem decrescente de similaridade.

        Falha ao gerar o embedding da pergunta propaga ``ErroEmbeddings`` sem
        alterar o índice.
        """
        vetor = self._embeddings.gerar([pergunta])[0]

        mencionadas = fontes_mencionadas(pergunta, self._indice.listar_fontes())
        if len(mencionadas) < 2:
            return self._indice.buscar(vetor, self._top_k)

        por_arquivo = max(1, LIMITE_TOTAL_COMPARATIVA // len(mencionadas))
        resultados: list[ResultadoBusca] = []
        for fonte in mencionadas:
            resultados.extend(self._indice.buscar(vetor, por_arquivo, fontes=[fonte]))
        # sort estável: empates mantêm a ordem de busca por arquivo
        resultados.sort(key=lambda r: r.similaridade, reverse=True)
        return resultados[:LIMITE_TOTAL_COMPARATIVA]
