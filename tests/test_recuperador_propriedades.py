"""Testes de propriedade para ``Recuperador``."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pdf_rag_chatbot.config import validar_top_k
from pdf_rag_chatbot.embeddings import GeradorDeEmbeddings
from pdf_rag_chatbot.erros import ErroConfiguracao
from pdf_rag_chatbot.indice_vetorial import IndiceVetorial
from pdf_rag_chatbot.modelos import Trecho
from pdf_rag_chatbot.recuperador import Recuperador

from dobles import EmbeddingsFalso

_VALIDOS = st.integers(min_value=1, max_value=20)

_INVALIDOS = st.one_of(
    st.integers(max_value=0),
    st.integers(min_value=21),
    st.floats(allow_nan=True, allow_infinity=True),
    st.booleans(),
    st.text(max_size=10),
    st.none(),
    st.lists(st.integers(), max_size=3),
).filter(lambda v: not (type(v) is int and 1 <= v <= 20))


def _criar_recuperador(diretorio: Path, top_k: int) -> Recuperador:
    provedor = EmbeddingsFalso()
    indice = IndiceVetorial(diretorio, provedor.nome_modelo)
    indice.carregar()
    return Recuperador(GeradorDeEmbeddings(provedor), indice, top_k)


@settings(max_examples=100, deadline=None)
@given(valor=_VALIDOS)
def test_top_k_valido_e_aceito(valor: int) -> None:
    # Feature: pdf-rag-chatbot, Property 13: Validação de Top_K aceita exatamente os inteiros em [1, 20]
    # **Validates: Requirements 2.6, 2.7**
    assert validar_top_k(valor) == valor
    with tempfile.TemporaryDirectory() as tmp:
        recuperador = _criar_recuperador(Path(tmp) / "indice", 4)
        recuperador.definir_top_k(valor)
        assert recuperador.top_k == valor


@settings(max_examples=100, deadline=None)
@given(anterior=_VALIDOS, valor=_INVALIDOS)
def test_top_k_invalido_e_rejeitado_e_mantem_valor_anterior(
    anterior: int, valor: object
) -> None:
    # Feature: pdf-rag-chatbot, Property 13: Validação de Top_K rejeita os demais valores e mantém o anterior
    # **Validates: Requirements 2.6, 2.7**
    with pytest.raises(ErroConfiguracao) as erro:
        validar_top_k(valor)
    assert "1" in str(erro.value) and "20" in str(erro.value)

    with tempfile.TemporaryDirectory() as tmp:
        recuperador = _criar_recuperador(Path(tmp) / "indice", anterior)
        with pytest.raises(ErroConfiguracao):
            recuperador.definir_top_k(valor)
        assert recuperador.top_k == anterior


@st.composite
def _cenario_comparativo(draw):
    """Índice com vários arquivos, pergunta que menciona 2 a 10 deles e Top_K."""
    quantidade = draw(st.integers(min_value=2, max_value=12))
    # nomes de largura fixa: nenhum é substring de outro
    fontes = [f"arquivo{n:02d}x.pdf" for n in range(quantidade)]
    trechos_por_fonte = {
        fonte: draw(st.integers(min_value=1, max_value=4)) for fonte in fontes
    }
    mencionados = draw(
        st.lists(
            st.sampled_from(fontes),
            min_size=2,
            max_size=min(10, quantidade),
            unique=True,
        )
    )
    com_extensao = draw(st.booleans())
    nomes = [m if com_extensao else m[: -len(".pdf")] for m in mencionados]
    pergunta = "Compare " + " e ".join(n.upper() for n in nomes) + " sobre o tema."
    top_k = draw(st.integers(min_value=1, max_value=20))
    return trechos_por_fonte, mencionados, pergunta, top_k


@settings(max_examples=100, deadline=None)
@given(cenario=_cenario_comparativo())
def test_pergunta_comparativa_recupera_trechos_de_cada_arquivo_mencionado(
    cenario,
) -> None:
    # Feature: pdf-rag-chatbot, Property 19: Perguntas comparativas recuperam Trechos de cada arquivo mencionado
    # **Validates: Requirements 3.5**
    trechos_por_fonte, mencionados, pergunta, top_k = cenario
    with tempfile.TemporaryDirectory() as tmp:
        provedor = EmbeddingsFalso()
        embeddings = GeradorDeEmbeddings(provedor)
        indice = IndiceVetorial(Path(tmp) / "indice", provedor.nome_modelo)
        indice.carregar()
        for fonte, quantidade in trechos_por_fonte.items():
            trechos = [
                Trecho(texto=f"{fonte} conteúdo {n}", fonte=fonte, pagina=n + 1)
                for n in range(quantidade)
            ]
            indice.substituir_fonte(
                fonte, trechos, embeddings.gerar([t.texto for t in trechos])
            )

        resultados = Recuperador(embeddings, indice, top_k).recuperar(pergunta)

    fontes_recuperadas = {r.trecho.fonte for r in resultados}
    assert set(mencionados) <= fontes_recuperadas
    assert len(resultados) <= 10
