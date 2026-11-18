"""Testes de exemplo do Indice_Vetorial (Requisitos 2.3, 2.4 e 2.5)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from pdf_rag_chatbot.erros import ErroIndiceCorrompido, ErroIndiceIncompativel
from pdf_rag_chatbot.indice_vetorial import (
    ARQUIVO_META,
    ARQUIVO_TRECHOS,
    ARQUIVO_VETORES,
    IndiceVetorial,
)
from pdf_rag_chatbot.modelos import Trecho

MODELO = "modelo-teste"


def _popular(indice: IndiceVetorial, fonte: str, n: int = 3, dim: int = 4) -> None:
    trechos = [Trecho(texto=f"texto {i}", fonte=fonte, pagina=i + 1) for i in range(n)]
    vetores = np.eye(max(n, dim), dim, dtype=np.float32)[:n] + 0.1
    indice.substituir_fonte(fonte, trechos, vetores)


def _indice_gravado(diretorio: Path) -> None:
    indice = IndiceVetorial(diretorio, MODELO)
    _popular(indice, "a.pdf")


# ----------------------------------------------------------------------
# Estado inicial
# ----------------------------------------------------------------------
def test_indice_novo_esta_vazio(tmp_path: Path) -> None:
    indice = IndiceVetorial(tmp_path / "indice", MODELO)
    assert indice.esta_vazio()
    assert len(indice) == 0
    assert indice.listar_fontes() == []
    assert indice.buscar(np.ones(4, dtype=np.float32), 3) == []


def test_diretorio_inexistente_carrega_vazio(tmp_path: Path) -> None:
    indice = IndiceVetorial(tmp_path / "nao_existe", MODELO)
    indice.carregar()
    assert indice.esta_vazio()
    assert not (tmp_path / "nao_existe").exists()


def test_diretorio_existente_sem_arquivos_carrega_vazio(tmp_path: Path) -> None:
    indice = IndiceVetorial(tmp_path, MODELO)
    indice.carregar()
    assert indice.esta_vazio()


# ----------------------------------------------------------------------
# Arquivos corrompidos
# ----------------------------------------------------------------------
def test_arquivos_incompletos_geram_erro_corrompido(tmp_path: Path) -> None:
    _indice_gravado(tmp_path)
    (tmp_path / ARQUIVO_VETORES).unlink()
    with pytest.raises(ErroIndiceCorrompido):
        IndiceVetorial(tmp_path, MODELO).carregar()


def test_meta_json_invalido_gera_erro_corrompido(tmp_path: Path) -> None:
    _indice_gravado(tmp_path)
    (tmp_path / ARQUIVO_META).write_text("{não é json", encoding="utf-8")
    with pytest.raises(ErroIndiceCorrompido):
        IndiceVetorial(tmp_path, MODELO).carregar()


def test_chunks_json_invalido_gera_erro_corrompido(tmp_path: Path) -> None:
    _indice_gravado(tmp_path)
    (tmp_path / ARQUIVO_TRECHOS).write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(ErroIndiceCorrompido):
        IndiceVetorial(tmp_path, MODELO).carregar()


def test_item_de_trecho_com_campos_invalidos_gera_erro_corrompido(tmp_path: Path) -> None:
    _indice_gravado(tmp_path)
    (tmp_path / ARQUIVO_TRECHOS).write_text(
        json.dumps([{"texto": "x", "fonte": "a.pdf", "pagina": 0}]), encoding="utf-8"
    )
    with pytest.raises(ErroIndiceCorrompido):
        IndiceVetorial(tmp_path, MODELO).carregar()


def test_vetores_ilegiveis_geram_erro_corrompido(tmp_path: Path) -> None:
    _indice_gravado(tmp_path)
    (tmp_path / ARQUIVO_VETORES).write_bytes(b"lixo")
    with pytest.raises(ErroIndiceCorrompido):
        IndiceVetorial(tmp_path, MODELO).carregar()


def test_quantidade_de_vetores_diferente_de_trechos_gera_erro_corrompido(
    tmp_path: Path,
) -> None:
    _indice_gravado(tmp_path)
    trechos = json.loads((tmp_path / ARQUIVO_TRECHOS).read_text(encoding="utf-8"))
    (tmp_path / ARQUIVO_TRECHOS).write_text(json.dumps(trechos[:-1]), encoding="utf-8")
    with pytest.raises(ErroIndiceCorrompido):
        IndiceVetorial(tmp_path, MODELO).carregar()


def test_dimensao_diferente_de_meta_gera_erro_corrompido(tmp_path: Path) -> None:
    _indice_gravado(tmp_path)
    meta = json.loads((tmp_path / ARQUIVO_META).read_text(encoding="utf-8"))
    meta["dimensao"] = meta["dimensao"] + 1
    (tmp_path / ARQUIVO_META).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ErroIndiceCorrompido):
        IndiceVetorial(tmp_path, MODELO).carregar()


# ----------------------------------------------------------------------
# Modelo incompatível
# ----------------------------------------------------------------------
def test_modelo_diferente_gera_erro_incompativel(tmp_path: Path) -> None:
    _indice_gravado(tmp_path)
    with pytest.raises(ErroIndiceIncompativel) as info:
        IndiceVetorial(tmp_path, "outro-modelo").carregar()
    assert MODELO in str(info.value)
    assert "outro-modelo" in str(info.value)


# ----------------------------------------------------------------------
# Remoção de fonte
# ----------------------------------------------------------------------
def test_lista_vazia_remove_a_fonte_e_persiste(tmp_path: Path) -> None:
    indice = IndiceVetorial(tmp_path, MODELO)
    _popular(indice, "a.pdf", n=2)
    _popular(indice, "b.pdf", n=3)
    assert indice.listar_fontes() == ["a.pdf", "b.pdf"]

    indice.substituir_fonte("a.pdf", [], np.empty((0, 4), dtype=np.float32))

    assert indice.listar_fontes() == ["b.pdf"]
    assert len(indice) == 3

    recarregado = IndiceVetorial(tmp_path, MODELO)
    recarregado.carregar()
    assert recarregado.listar_fontes() == ["b.pdf"]
    assert len(recarregado) == 3


def test_remover_ultima_fonte_deixa_indice_vazio(tmp_path: Path) -> None:
    indice = IndiceVetorial(tmp_path, MODELO)
    _popular(indice, "a.pdf")
    indice.substituir_fonte("a.pdf", [], np.empty((0, 4), dtype=np.float32))
    assert indice.esta_vazio()

    recarregado = IndiceVetorial(tmp_path, MODELO)
    recarregado.carregar()
    assert recarregado.esta_vazio()


def test_remover_fonte_inexistente_nao_altera_o_indice(tmp_path: Path) -> None:
    indice = IndiceVetorial(tmp_path, MODELO)
    _popular(indice, "a.pdf")
    indice.substituir_fonte("z.pdf", [], np.empty((0, 4), dtype=np.float32))
    assert indice.listar_fontes() == ["a.pdf"]


# ----------------------------------------------------------------------
# Desempenho (Requisitos 2.4 e 2.5)
# ----------------------------------------------------------------------
def test_desempenho_busca_e_gravacao_com_10000_trechos(tmp_path: Path) -> None:
    quantidade, dimensao = 10_000, 384
    gerador = np.random.default_rng(42)
    vetores = gerador.standard_normal((quantidade, dimensao)).astype(np.float32)
    trechos = [
        Trecho(texto=f"trecho {i}", fonte="grande.pdf", pagina=i % 500 + 1)
        for i in range(quantidade)
    ]

    indice = IndiceVetorial(tmp_path / "grande", MODELO)
    inicio = time.perf_counter()
    indice.substituir_fonte("grande.pdf", trechos, vetores)
    duracao_gravacao = time.perf_counter() - inicio
    assert duracao_gravacao < 5.0, f"gravação levou {duracao_gravacao:.2f} s"
    assert len(indice) == quantidade

    consulta = gerador.standard_normal(dimensao).astype(np.float32)
    inicio = time.perf_counter()
    resultados = indice.buscar(consulta, 4)
    duracao_busca = time.perf_counter() - inicio
    assert duracao_busca < 2.0, f"busca levou {duracao_busca:.2f} s"
    assert len(resultados) == 4
    similaridades = [r.similaridade for r in resultados]
    assert similaridades == sorted(similaridades, reverse=True)
