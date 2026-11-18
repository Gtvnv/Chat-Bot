"""Testes de propriedade para ``GeradorDeEmbeddings``."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pdf_rag_chatbot.embeddings.base import GeradorDeEmbeddings
from pdf_rag_chatbot.erros import ErroEmbeddings

from dobles import EmbeddingsFalso


class _ProvedorInconsistente:
    """Provedor que corrompe a quantidade ou as dimensões dos vetores."""

    nome_modelo = "inconsistente"

    def __init__(self, modo: str) -> None:
        self.modo = modo
        self.chamadas = 0
        self.emitidos = 0

    def embutir(self, textos: Sequence[str]) -> list[list[float]]:
        self.chamadas += 1
        if self.modo == "quantidade":
            return [[1.0, 2.0, 3.0] for _ in textos][:-1]
        # Dimensão: vetores consecutivos (mesmo entre lotes) alternam 3 e 2.
        vetores = [[1.0] * (3 if (self.emitidos + i) % 2 == 0 else 2)
                   for i in range(len(textos))]
        self.emitidos += len(textos)
        return vetores


_textos = st.lists(st.text(max_size=50), min_size=1, max_size=40)


# Feature: pdf-rag-chatbot, Property 9: Embeddings têm um vetor por texto e mesma dimensão
# **Validates: Requirements 2.1**
@settings(max_examples=100)
@given(
    textos=_textos,
    tamanho_lote=st.integers(min_value=1, max_value=50),
    dimensao=st.integers(min_value=1, max_value=32),
)
def test_embeddings_um_vetor_por_texto_com_mesma_dimensao(
    textos, tamanho_lote, dimensao
):
    gerador = GeradorDeEmbeddings(EmbeddingsFalso(dimensao=dimensao), tamanho_lote)

    matriz = gerador.gerar(textos)

    assert matriz.shape == (len(textos), dimensao)


# Feature: pdf-rag-chatbot, Property 9: Embeddings têm um vetor por texto e mesma dimensão
# **Validates: Requirements 2.1**
@settings(max_examples=100)
@given(
    textos=st.lists(st.text(max_size=50), min_size=2, max_size=40),
    tamanho_lote=st.integers(min_value=1, max_value=50),
    modo=st.sampled_from(["quantidade", "dimensao"]),
)
def test_embeddings_inconsistentes_lancam_erro(textos, tamanho_lote, modo):
    gerador = GeradorDeEmbeddings(_ProvedorInconsistente(modo), tamanho_lote)

    with pytest.raises(ErroEmbeddings):
        gerador.gerar(textos)
