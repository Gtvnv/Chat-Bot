"""Testes de exemplo para a divisão de texto em Trechos (Requisitos 1.2 e 9.5)."""

from __future__ import annotations

import pytest

from pdf_rag_chatbot.chunking import dividir_em_trechos


def test_texto_vazio_devolve_lista_vazia():
    assert dividir_em_trechos("", 100, 20) == []


def test_texto_so_com_espacos_devolve_lista_vazia():
    assert dividir_em_trechos(" \t\n  ", 100, 20) == []


def test_texto_menor_que_o_tamanho_gera_um_unico_trecho():
    assert dividir_em_trechos("texto curto", 100, 20) == ["texto curto"]


def test_texto_com_espacos_nas_pontas_e_limpo_antes_de_dividir():
    assert dividir_em_trechos("  abc  ", 100, 0) == ["abc"]


def test_sobreposicao_zero_gera_trechos_contiguos_sem_repeticao():
    trechos = dividir_em_trechos("abcdefghij", 4, 0)
    assert trechos == ["abcd", "efgh", "ij"]
    assert "".join(trechos) == "abcdefghij"


def test_sobreposicao_maxima_de_50_por_cento():
    # tamanho 4, sobreposição 2 (50%): passo 2
    trechos = dividir_em_trechos("abcdefgh", 4, 2)
    assert trechos == ["abcd", "cdef", "efgh"]
    for anterior, seguinte in zip(trechos, trechos[1:]):
        assert anterior[-2:] == seguinte[:2]


def test_texto_com_exatamente_o_tamanho_gera_um_trecho():
    assert dividir_em_trechos("abcd", 4, 2) == ["abcd"]


def test_parametros_invalidos_levantam_value_error():
    with pytest.raises(ValueError):
        dividir_em_trechos("abc", 4, 4)
    with pytest.raises(ValueError):
        dividir_em_trechos("abc", 0, 0)
