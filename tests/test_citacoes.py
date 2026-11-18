"""Testes de exemplo para as Citações."""

from __future__ import annotations

from pdf_rag_chatbot.citacoes import (
    extrair_citacoes,
    formatar_citacao,
    formatar_lista_citacoes,
)
from pdf_rag_chatbot.modelos import Citacao, ResultadoBusca


def _resultado(fonte: str, pagina: int, tipo: str = "pdf") -> ResultadoBusca:
    from pdf_rag_chatbot.modelos import Trecho

    return ResultadoBusca(
        trecho=Trecho(texto="texto", fonte=fonte, pagina=pagina, tipo=tipo),
        similaridade=0.9,
    )


def test_formata_citacao_de_pdf_com_pagina():
    citacao = Citacao(fonte="artigo.pdf", pagina=12, tipo="pdf")
    assert formatar_citacao(citacao) == "artigo.pdf, p. 12"


def test_formata_citacao_de_texto_com_linha():
    citacao = Citacao(fonte="sentencas.txt", pagina=3, tipo="texto")
    assert formatar_citacao(citacao) == "sentencas.txt, linha 3"


def test_formata_lista_com_titulo_e_uma_citacao_por_linha():
    citacoes = [
        Citacao(fonte="artigo.pdf", pagina=2, tipo="pdf"),
        Citacao(fonte="sentencas.txt", pagina=5, tipo="texto"),
    ]
    assert formatar_lista_citacoes(citacoes) == (
        "Fontes:\nartigo.pdf, p. 2\nsentencas.txt, linha 5"
    )


def test_lista_vazia_gera_apenas_o_titulo():
    assert formatar_lista_citacoes([]) == "Fontes:"


def test_resposta_sem_marcadores_nao_gera_citacoes():
    trechos = [_resultado("a.pdf", 1), _resultado("b.pdf", 2)]
    assert extrair_citacoes("Resposta sem nenhuma referência.", trechos) == []


def test_marcador_fora_da_faixa_e_ignorado():
    trechos = [_resultado("a.pdf", 1), _resultado("b.pdf", 2)]
    citacoes = extrair_citacoes("Afirmação [0], outra [3] e válida [2].", trechos)
    assert citacoes == [Citacao(fonte="b.pdf", pagina=2, tipo="pdf")]


def test_marcadores_repetidos_geram_uma_citacao_na_ordem_da_primeira_ocorrência():
    trechos = [_resultado("a.pdf", 1), _resultado("b.pdf", 2)]
    citacoes = extrair_citacoes("Um [2], dois [1], de novo [2] e [1].", trechos)
    assert citacoes == [
        Citacao(fonte="b.pdf", pagina=2, tipo="pdf"),
        Citacao(fonte="a.pdf", pagina=1, tipo="pdf"),
    ]


def test_trechos_distintos_da_mesma_fonte_e_pagina_geram_uma_citacao():
    trechos = [_resultado("a.pdf", 4), _resultado("a.pdf", 4)]
    citacoes = extrair_citacoes("Fato [1] e fato [2].", trechos)
    assert citacoes == [Citacao(fonte="a.pdf", pagina=4, tipo="pdf")]


def test_sem_trechos_nenhum_marcador_e_valido():
    assert extrair_citacoes("Texto [1].", []) == []
