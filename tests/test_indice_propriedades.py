"""Testes de propriedade para ``IndiceVetorial``."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

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


# ---------------------------------------------------------------------------
# Property 31: gravação interrompida
# ---------------------------------------------------------------------------

_MODULO = "pdf_rag_chatbot.indice_vetorial"


class _FalhaInjetada(RuntimeError):
    """Falha simulada em um ponto da gravação."""


@st.composite
def _estado_anterior_e_operacao(draw):
    """Gera o conteúdo anterior, a operação (fonte, itens) e o ponto de falha."""
    anterior = draw(_conteudo_indice())
    dimensao = len(next(iter(anterior.values()))[0][1])
    vetor = st.lists(
        st.integers(min_value=-5, max_value=5).map(float),
        min_size=dimensao,
        max_size=dimensao,
    ).filter(lambda v: any(x != 0.0 for x in v))
    fonte = draw(st.sampled_from(sorted(anterior) + ["nova.pdf"]))
    itens = draw(
        st.lists(
            st.tuples(
                st.text(max_size=60),
                st.integers(min_value=1, max_value=500),
                st.sampled_from(["pdf", "texto"]),
                vetor,
            ),
            min_size=0,  # lista vazia remove a fonte
            max_size=4,
        )
    )
    novos = [(Trecho(texto=t, fonte=fonte, pagina=p, tipo=tp), v) for t, p, tp, v in itens]
    # Há 10 operações de gravação por chamada; valores acima disso não injetam falha.
    ponto = draw(st.integers(min_value=0, max_value=12))
    return anterior, fonte, novos, ponto


def _popular(indice: IndiceVetorial, conteudo) -> None:
    for fonte, itens in conteudo.items():
        indice.substituir_fonte(
            fonte,
            [t for t, _ in itens],
            np.array([v for _, v in itens], dtype=np.float32),
        )


def _estado(indice: IndiceVetorial):
    return list(indice._trechos), np.array(indice._vetores, copy=True)


def _assert_estado_igual(a, b) -> None:
    assert a[0] == b[0]
    assert a[1].shape == b[1].shape
    np.testing.assert_allclose(a[1], b[1], rtol=0, atol=1e-6)


def _executar_com_falha(indice: IndiceVetorial, fonte, novos, ponto: int) -> bool:
    """Roda ``substituir_fonte`` falhando na ``ponto``-ésima operação de gravação.

    Operações contadas (em ordem): ``open`` de escrita, ``np.save``, ``os.fsync`` e
    ``os.replace``. Devolve ``True`` se a falha foi disparada.
    """
    contador = {"n": 0, "falhou": False}
    open_real, save_real = open, np.save
    fsync_real, replace_real = os.fsync, os.replace

    def _passo():
        if contador["n"] == ponto:
            contador["falhou"] = True
            contador["n"] += 1
            raise _FalhaInjetada()
        contador["n"] += 1

    def _open(arquivo, modo="r", *args, **kwargs):
        if any(c in modo for c in "wax+"):
            _passo()
        return open_real(arquivo, modo, *args, **kwargs)

    def _save(*args, **kwargs):
        _passo()
        return save_real(*args, **kwargs)

    def _fsync(fd):
        _passo()
        return fsync_real(fd)

    def _replace(origem, destino):
        _passo()
        return replace_real(origem, destino)

    trechos = [t for t, _ in novos]
    vetores = (
        np.array([v for _, v in novos], dtype=np.float32)
        if novos
        else np.empty((0, 0), dtype=np.float32)
    )
    with (
        patch(f"{_MODULO}.open", _open, create=True),
        patch(f"{_MODULO}.np.save", _save),
        patch(f"{_MODULO}.os.fsync", _fsync),
        patch(f"{_MODULO}.os.replace", _replace),
    ):
        try:
            indice.substituir_fonte(fonte, trechos, vetores)
        except _FalhaInjetada:
            pass
    return contador["falhou"]


@settings(max_examples=100, deadline=None)
@given(dados=_estado_anterior_e_operacao())
def test_gravacao_interrompida_preserva_estado_completo(dados) -> None:
    # Feature: pdf-rag-chatbot, Property 31: Gravação interrompida do índice preserva um estado completo
    # **Validates: Requirements 2.14, 1.10**
    anterior, fonte, novos, ponto = dados
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        diretorio = raiz / "indice"
        indice = IndiceVetorial(diretorio, _MODELO)
        indice.carregar()
        _popular(indice, anterior)
        estado_anterior = _estado(indice)

        # Estado novo esperado, calculado em um diretório separado sem falhas.
        referencia = IndiceVetorial(raiz / "referencia", _MODELO)
        referencia.carregar()
        _popular(referencia, anterior)
        referencia.substituir_fonte(
            fonte,
            [t for t, _ in novos],
            np.array([v for _, v in novos], dtype=np.float32)
            if novos
            else np.empty((0, 0), dtype=np.float32),
        )
        estado_novo = _estado(referencia)

        falhou = _executar_com_falha(indice, fonte, novos, ponto)

        carregado = IndiceVetorial(diretorio, _MODELO)
        carregado.carregar()
        em_disco = _estado(carregado)

        # Disco: exatamente o estado anterior (se falhou) ou o novo (se concluiu).
        _assert_estado_igual(em_disco, estado_anterior if falhou else estado_novo)

        # Memória da instância original coerente com o disco.
        _assert_estado_igual(_estado(indice), em_disco)

        # Nenhuma versão órfã: só a versão apontada pelo manifesto continua no disco.
        versoes = list((diretorio / "versoes").iterdir())
        assert len(versoes) == 1
        assert not list(diretorio.glob("atual.json.*.tmp"))

        # O índice continua gravável depois da falha.
        indice.substituir_fonte("final.pdf", [], np.empty((0, 0), dtype=np.float32))


@settings(max_examples=100, deadline=None)
@given(
    anterior=_conteudo_indice(),
    arquivos=st.lists(
        st.sampled_from(["vectors.npy", "chunks.json", "meta.json"]),
        min_size=0,
        max_size=3,
        unique=True,
    ),
    corte=st.integers(min_value=0, max_value=20),
    com_manifesto_tmp=st.booleans(),
)
def test_interrupcao_abrupta_mantem_estado_anterior(
    anterior, arquivos, corte, com_manifesto_tmp
) -> None:
    # Feature: pdf-rag-chatbot, Property 31: Gravação interrompida do índice preserva um estado completo
    # **Validates: Requirements 2.14, 1.10**
    with tempfile.TemporaryDirectory() as tmp:
        diretorio = Path(tmp) / "indice"
        indice = IndiceVetorial(diretorio, _MODELO)
        indice.carregar()
        _popular(indice, anterior)
        estado_anterior = _estado(indice)

        # Processo morto: pasta de versão parcial (arquivos truncados) e manifesto
        # temporário, sem troca do manifesto.
        parcial = diretorio / "versoes" / "20990101T000000-deadbeef"
        parcial.mkdir(parents=True)
        for nome in arquivos:
            (parcial / nome).write_bytes(b"lixo-parcial-0123456789"[:corte])
        if com_manifesto_tmp:
            (diretorio / "atual.json.1234.tmp").write_text(
                '{"versao": "20990101T000000-deadbeef"', encoding="utf-8"
            )

        carregado = IndiceVetorial(diretorio, _MODELO)
        carregado.carregar()
        _assert_estado_igual(_estado(carregado), estado_anterior)
