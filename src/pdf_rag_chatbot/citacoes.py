"""Extração e formatação de Citações a partir dos marcadores ``[n]`` da resposta."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pdf_rag_chatbot.modelos import Citacao, ResultadoBusca

_PADRAO_MARCADOR = re.compile(r"\[(\d+)\]")
TITULO_FONTES = "Fontes:"


def extrair_citacoes(resposta: str, trechos: Sequence[ResultadoBusca]) -> list[Citacao]:
    """Converte os marcadores ``[n]`` da resposta em Citações.

    Números fora de ``1..len(trechos)`` são ignorados, repetições por
    ``(fonte, pagina)`` são removidas e a ordem da primeira ocorrência é mantida.
    """
    citacoes: list[Citacao] = []
    vistas: set[tuple[str, int]] = set()
    for correspondencia in _PADRAO_MARCADOR.finditer(resposta):
        numero = int(correspondencia.group(1))
        if not 1 <= numero <= len(trechos):
            continue
        trecho = trechos[numero - 1].trecho
        chave = (trecho.fonte, trecho.pagina)
        if chave in vistas:
            continue
        vistas.add(chave)
        citacoes.append(Citacao(fonte=trecho.fonte, pagina=trecho.pagina, tipo=trecho.tipo))
    return citacoes


def formatar_citacao(citacao: Citacao) -> str:
    """Devolve ``arquivo.pdf, p. 12`` (PDF) ou ``sentencas.txt, linha 3`` (texto)."""
    rotulo = "linha" if citacao.tipo == "texto" else "p."
    return f"{citacao.fonte}, {rotulo} {citacao.pagina}"


def formatar_lista_citacoes(citacoes: Sequence[Citacao]) -> str:
    """Formata as Citações sob o título "Fontes:", uma por linha."""
    linhas = [TITULO_FONTES, *(formatar_citacao(c) for c in citacoes)]
    return "\n".join(linhas)
