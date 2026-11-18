"""Testes de propriedade para ``dividir_em_trechos``."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from pdf_rag_chatbot.chunking import dividir_em_trechos


@st.composite
def _entradas_chunking(draw):
    """Gera (texto, tamanho, sobreposicao) dentro das faixas do Requisito 1.3."""
    tamanho = draw(st.integers(min_value=100, max_value=10000))
    sobreposicao = draw(st.integers(min_value=0, max_value=tamanho // 2))
    # Textos de até ~3 janelas de tamanho máximo, para exercitar vários Trechos
    # sem tornar os exemplos lentos; inclui espaços nas pontas.
    tamanho_maximo = min(3 * tamanho + 50, 12000)
    texto = draw(st.text(max_size=tamanho_maximo))
    return texto, tamanho, sobreposicao


# Feature: pdf-rag-chatbot, Property 1: Chunking respeita tamanho, sobreposição e preserva o texto
# **Validates: Requirements 1.2**
@settings(max_examples=100)
@given(_entradas_chunking())
def test_chunking_respeita_tamanho_sobreposicao_e_preserva_texto(entrada):
    texto, tamanho, sobreposicao = entrada
    trechos = dividir_em_trechos(texto, tamanho, sobreposicao)
    esperado = texto.strip()

    if not esperado:
        assert trechos == []
        return

    # Trechos não vazios e com no máximo `tamanho` caracteres.
    assert trechos
    for trecho in trechos:
        assert 1 <= len(trecho) <= tamanho

    # Consecutivos compartilham exatamente `sobreposicao` caracteres.
    for anterior, seguinte in zip(trechos, trechos[1:]):
        assert len(anterior) == tamanho
        assert len(seguinte) > sobreposicao
        if sobreposicao > 0:
            assert anterior[-sobreposicao:] == seguinte[:sobreposicao]

    # Reconstrução do texto original sem espaços nas pontas.
    reconstruido = trechos[0] + "".join(t[sobreposicao:] for t in trechos[1:])
    assert reconstruido == esperado
