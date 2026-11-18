"""Teste de propriedade da ida e volta entre indexar e recuperar."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from pdf_rag_chatbot.embeddings import GeradorDeEmbeddings
from pdf_rag_chatbot.indice_vetorial import IndiceVetorial
from pdf_rag_chatbot.modelos import Trecho
from pdf_rag_chatbot.recuperador import Recuperador

from dobles import EmbeddingsFalso


@st.composite
def _trechos_distintos(draw):
    """Textos únicos distribuídos em até 3 fontes, mais o índice do Trecho a consultar."""
    textos = draw(
        st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz áéíóúç", min_size=1, max_size=80),
            min_size=1,
            max_size=15,
            unique=True,
        )
    )
    # Nomes de fonte com dígitos e sublinhado, que não aparecem nos textos gerados.
    quantidade_fontes = draw(st.integers(min_value=1, max_value=3))
    trechos = []
    for posicao, texto in enumerate(textos):
        fonte = f"arquivo_{posicao % quantidade_fontes}.pdf"
        pagina = draw(st.integers(min_value=1, max_value=50))
        trechos.append(Trecho(texto=texto, fonte=fonte, pagina=pagina))
    alvo = draw(st.integers(min_value=0, max_value=len(trechos) - 1))
    return trechos, alvo


@settings(max_examples=100, deadline=None)
@given(dados=_trechos_distintos())
def test_consultar_com_texto_exato_devolve_o_trecho_na_primeira_posicao(dados) -> None:
    # Feature: pdf-rag-chatbot, Property 12: Ida e volta entre indexar e recuperar
    # **Validates: Requirements 2.10**
    trechos, alvo = dados
    provedor = EmbeddingsFalso()
    gerador = GeradorDeEmbeddings(provedor)
    with tempfile.TemporaryDirectory() as tmp:
        indice = IndiceVetorial(Path(tmp) / "indice", provedor.nome_modelo)
        indice.carregar()

        por_fonte: dict[str, list[Trecho]] = {}
        for trecho in trechos:
            por_fonte.setdefault(trecho.fonte, []).append(trecho)
        for fonte, grupo in por_fonte.items():
            vetores = gerador.gerar([t.texto for t in grupo])
            indice.substituir_fonte(fonte, grupo, np.asarray(vetores))

        recuperador = Recuperador(gerador, indice)  # Top_K padrão
        resultados = recuperador.recuperar(trechos[alvo].texto)

        assert resultados
        assert resultados[0].trecho == trechos[alvo]
