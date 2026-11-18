"""Divisão de texto em Trechos por janelas fixas de caracteres."""

from __future__ import annotations


def dividir_em_trechos(texto: str, tamanho: int, sobreposicao: int) -> list[str]:
    """Divide ``texto`` em janelas de até ``tamanho`` caracteres.

    Função pura. O texto é limpo com ``strip`` e, se ficar vazio, devolve ``[]``.
    O passo entre janelas é ``tamanho - sobreposicao``, então dois Trechos
    consecutivos compartilham exatamente ``sobreposicao`` caracteres. A divisão
    termina quando uma janela alcança o fim do texto.

    A validação das faixas aceitas (Requisito 1.3) é feita em
    ``validar_parametros_chunking``; aqui só se protege contra um passo não
    positivo, que causaria um laço infinito.
    """
    if tamanho <= 0 or sobreposicao < 0 or sobreposicao >= tamanho:
        raise ValueError(
            "tamanho deve ser positivo e sobreposicao deve estar em [0, tamanho)"
        )

    texto = texto.strip()
    if not texto:
        return []

    passo = tamanho - sobreposicao
    trechos: list[str] = []
    inicio = 0
    while True:
        fim = inicio + tamanho
        trechos.append(texto[inicio:fim])
        if fim >= len(texto):
            break
        inicio += passo
    return trechos
