"""Testes de propriedade para ``IndiceVetorial``."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pdf_rag_chatbot.erros import ErroIndiceIncompativel
from pdf_rag_chatbot.indice_vetorial import IndiceVetorial
from pdf_rag_chatbot.modelos import Trecho

_MODELO = "modelo-a"


@st.composite
def _conteudo_indice(draw):
    """Gera ``{fonte: [(Trecho, vetor), ...]}`` com dimensão única e vetores não nulos."""
    dimensao = draw(st.integers(min_value=1, max_value=8))
    vetor = st.lists(
        st.integers(min_value=-5, max_value=5).map(float),
        min_size=dimensao,
        max_size=dimensao,
    ).filter(lambda v: any(x != 0.0 for x in v))
    fontes = draw(
        st.lists(
            st.text(alphabet="abcdefghij", min_size=1, max_size=6).map(lambda s: s + ".pdf"),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    conteudo = {}
    for fonte in fontes:
        itens = draw(
            st.lists(
                st.tuples(
                    st.text(max_size=60),
                    st.integers(min_value=1, max_value=500),
                    st.sampled_from(["pdf", "texto"]),
                    vetor,
                ),
                min_size=1,
                max_size=5,
            )
        )
        conteudo[fonte] = [
            (Trecho(texto=t, fonte=fonte, pagina=p, tipo=tp), v) for t, p, tp, v in itens
        ]
    return conteudo


@settings(max_examples=100, deadline=None)
@given(conteudo=_conteudo_indice())
def test_persistencia_do_indice_e_ida_e_volta(conteudo) -> None:
    # Feature: pdf-rag-chatbot, Property 10: Persistência do índice é uma ida e volta
    # **Validates: Requirements 2.2, 2.3, 2.4**
    with tempfile.TemporaryDirectory() as tmp:
        diretorio = Path(tmp) / "indice"
        original = IndiceVetorial(diretorio, _MODELO)
        original.carregar()
        for fonte, itens in conteudo.items():
            trechos = [t for t, _ in itens]
            vetores = np.array([v for _, v in itens], dtype=np.float32)
            original.substituir_fonte(fonte, trechos, vetores)

        recarregado = IndiceVetorial(diretorio, _MODELO)
        recarregado.carregar()

        assert len(recarregado) == len(original)
        assert recarregado.listar_fontes() == original.listar_fontes()
        assert recarregado._trechos == original._trechos
        assert recarregado._vetores.shape == original._vetores.shape
        np.testing.assert_allclose(
            recarregado._vetores, original._vetores, rtol=0, atol=1e-6
        )

        # Modelo de embedding diferente não é aceito na carga.
        outro_modelo = IndiceVetorial(diretorio, "modelo-b")
        with pytest.raises(ErroIndiceIncompativel):
            outro_modelo.carregar()


@st.composite
def _indice_e_consulta(draw):
    """Gera o conteúdo do índice, um vetor de consulta não nulo e ``k`` em [1, 20]."""
    conteudo = draw(_conteudo_indice())
    dimensao = len(next(iter(conteudo.values()))[0][1])
    consulta = draw(
        st.lists(
            st.integers(min_value=-5, max_value=5).map(float),
            min_size=dimensao,
            max_size=dimensao,
        ).filter(lambda v: any(x != 0.0 for x in v))
    )
    k = draw(st.integers(min_value=1, max_value=20))
    return conteudo, consulta, k


@settings(max_examples=100, deadline=None)
@given(dados=_indice_e_consulta())
def test_busca_devolve_mais_similares_em_ordem_decrescente(dados) -> None:
    # Feature: pdf-rag-chatbot, Property 11: Busca devolve os mais similares em ordem decrescente
    # **Validates: Requirements 2.5**
    conteudo, consulta, k = dados
    with tempfile.TemporaryDirectory() as tmp:
        indice = IndiceVetorial(Path(tmp) / "indice", _MODELO)
        indice.carregar()
        for fonte, itens in conteudo.items():
            trechos = [t for t, _ in itens]
            vetores = np.array([v for _, v in itens], dtype=np.float32)
            indice.substituir_fonte(fonte, trechos, vetores)

        total = len(indice)
        resultados = indice.buscar(np.array(consulta, dtype=np.float32), k)

        # Exatamente min(k, N) resultados, todos pertencentes ao índice.
        assert len(resultados) == min(k, total)
        for r in resultados:
            assert r.trecho in indice._trechos

        # Ordem não crescente de similaridade.
        sims = [r.similaridade for r in resultados]
        assert all(a >= b for a, b in zip(sims, sims[1:]))

        # Similaridades de referência calculadas de forma independente (float64).
        q = np.array(consulta, dtype=np.float64)
        q /= np.linalg.norm(q)
        todas = indice._vetores.astype(np.float64) @ q
        esperadas = np.sort(todas)[::-1][: len(resultados)]
        np.testing.assert_allclose(sims, esperadas, rtol=0, atol=1e-5)

        # Nenhum Trecho fora dos resultados é mais similar que o último resultado.
        ultimo = sims[-1]
        assert np.sum(todas > ultimo + 1e-5) <= len(resultados) - 1
