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


def _pasta_atual(diretorio: Path) -> Path:
    """Pasta da versão atual do índice (formato 2)."""
    manifesto = json.loads((diretorio / "atual.json").read_text(encoding="utf-8"))
    return diretorio / "versoes" / manifesto["versao"]


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
    (_pasta_atual(tmp_path) / ARQUIVO_VETORES).unlink()
    with pytest.raises(ErroIndiceCorrompido):
        IndiceVetorial(tmp_path, MODELO).carregar()


def test_meta_json_invalido_gera_erro_corrompido(tmp_path: Path) -> None:
    _indice_gravado(tmp_path)
    (_pasta_atual(tmp_path) / ARQUIVO_META).write_text("{não é json", encoding="utf-8")
    with pytest.raises(ErroIndiceCorrompido):
        IndiceVetorial(tmp_path, MODELO).carregar()


def test_chunks_json_invalido_gera_erro_corrompido(tmp_path: Path) -> None:
    _indice_gravado(tmp_path)
    (_pasta_atual(tmp_path) / ARQUIVO_TRECHOS).write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(ErroIndiceCorrompido):
        IndiceVetorial(tmp_path, MODELO).carregar()


def test_item_de_trecho_com_campos_invalidos_gera_erro_corrompido(tmp_path: Path) -> None:
    _indice_gravado(tmp_path)
    (_pasta_atual(tmp_path) / ARQUIVO_TRECHOS).write_text(
        json.dumps([{"texto": "x", "fonte": "a.pdf", "pagina": 0}]), encoding="utf-8"
    )
    with pytest.raises(ErroIndiceCorrompido):
        IndiceVetorial(tmp_path, MODELO).carregar()


def test_vetores_ilegiveis_geram_erro_corrompido(tmp_path: Path) -> None:
    _indice_gravado(tmp_path)
    (_pasta_atual(tmp_path) / ARQUIVO_VETORES).write_bytes(b"lixo")
    with pytest.raises(ErroIndiceCorrompido):
        IndiceVetorial(tmp_path, MODELO).carregar()


def test_quantidade_de_vetores_diferente_de_trechos_gera_erro_corrompido(
    tmp_path: Path,
) -> None:
    _indice_gravado(tmp_path)
    trechos = json.loads((_pasta_atual(tmp_path) / ARQUIVO_TRECHOS).read_text(encoding="utf-8"))
    (_pasta_atual(tmp_path) / ARQUIVO_TRECHOS).write_text(json.dumps(trechos[:-1]), encoding="utf-8")
    with pytest.raises(ErroIndiceCorrompido):
        IndiceVetorial(tmp_path, MODELO).carregar()


def test_dimensao_diferente_de_meta_gera_erro_corrompido(tmp_path: Path) -> None:
    _indice_gravado(tmp_path)
    meta = json.loads((_pasta_atual(tmp_path) / ARQUIVO_META).read_text(encoding="utf-8"))
    meta["dimensao"] = meta["dimensao"] + 1
    (_pasta_atual(tmp_path) / ARQUIVO_META).write_text(json.dumps(meta), encoding="utf-8")
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


# ----------------------------------------------------------------------
# Formato 2: manifesto, migração do formato 1 e versões órfãs
# (Requisitos 2.3 e 2.14)
# ----------------------------------------------------------------------
def _montar_formato_1(diretorio: Path) -> np.ndarray:
    """Cria à mão os três arquivos do formato 1 na raiz; devolve os vetores."""
    diretorio.mkdir(parents=True, exist_ok=True)
    vetores = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    np.save(diretorio / ARQUIVO_VETORES, vetores, allow_pickle=False)
    (diretorio / ARQUIVO_TRECHOS).write_text(
        json.dumps(
            [
                {"texto": "alfa", "fonte": "a.pdf", "pagina": 1, "tipo": "pdf"},
                {"texto": "beta", "fonte": "a.pdf", "pagina": 2, "tipo": "pdf"},
            ]
        ),
        encoding="utf-8",
    )
    (diretorio / ARQUIVO_META).write_text(
        json.dumps(
            {"modelo_embeddings": MODELO, "dimensao": 3, "versao_formato": 1}
        ),
        encoding="utf-8",
    )
    return vetores


def test_formato_1_e_carregado_e_migrado_na_gravacao(tmp_path: Path) -> None:
    diretorio = tmp_path / "indice"
    vetores = _montar_formato_1(diretorio)

    indice = IndiceVetorial(diretorio, MODELO)
    indice.carregar()
    assert len(indice) == 2
    assert indice.listar_fontes() == ["a.pdf"]
    resultados = indice.buscar(np.array([1.0, 0.0, 0.0], dtype=np.float32), 2)
    assert [r.trecho.texto for r in resultados] == ["alfa", "beta"]

    novo = [Trecho(texto="gama", fonte="b.pdf", pagina=1)]
    indice.substituir_fonte("b.pdf", novo, np.array([[0.0, 0.0, 2.0]], np.float32))

    # Raiz: manifesto + diretório de versões, sem os arquivos do formato 1.
    assert sorted(p.name for p in diretorio.iterdir()) == ["atual.json", "versoes"]
    assert len(list((diretorio / "versoes").iterdir())) == 1
    pasta = _pasta_atual(diretorio)
    meta = json.loads((pasta / ARQUIVO_META).read_text(encoding="utf-8"))
    assert meta["versao_formato"] == 2
    manifesto = json.loads((diretorio / "atual.json").read_text(encoding="utf-8"))
    assert manifesto["versao_formato"] == 2

    # Conteúdo preservado, lido por uma instância nova.
    recarregado = IndiceVetorial(diretorio, MODELO)
    recarregado.carregar()
    assert recarregado.listar_fontes() == ["a.pdf", "b.pdf"]
    assert len(recarregado) == 3
    salvos = np.load(pasta / ARQUIVO_VETORES)
    assert salvos.dtype == np.float32
    np.testing.assert_allclose(salvos[:2], vetores)
    np.testing.assert_allclose(salvos[2], [0.0, 0.0, 1.0])


def test_manifesto_para_versao_inexistente_gera_erro_corrompido(
    tmp_path: Path,
) -> None:
    _indice_gravado(tmp_path)
    (tmp_path / "atual.json").write_text(
        json.dumps({"versao": "nao-existe", "versao_formato": 2}), encoding="utf-8"
    )
    with pytest.raises(ErroIndiceCorrompido):
        IndiceVetorial(tmp_path, MODELO).carregar()


def test_versao_orfa_e_ignorada_e_removida_na_gravacao_seguinte(
    tmp_path: Path,
) -> None:
    _indice_gravado(tmp_path)
    atual = _pasta_atual(tmp_path)
    orfa = tmp_path / "versoes" / "orfa-interrompida"
    orfa.mkdir()
    (orfa / ARQUIVO_VETORES).write_bytes(b"lixo")

    indice = IndiceVetorial(tmp_path, MODELO)
    indice.carregar()
    assert indice.listar_fontes() == ["a.pdf"]
    assert len(indice) == 3
    assert orfa.exists()

    _popular(indice, "b.pdf", n=2)

    assert not orfa.exists()
    assert not atual.exists()
    assert len(list((tmp_path / "versoes").iterdir())) == 1
    recarregado = IndiceVetorial(tmp_path, MODELO)
    recarregado.carregar()
    assert recarregado.listar_fontes() == ["a.pdf", "b.pdf"]


def test_manifesto_com_json_invalido_gera_erro_corrompido(tmp_path: Path) -> None:
    _indice_gravado(tmp_path)
    (tmp_path / "atual.json").write_text("{não é json", encoding="utf-8")
    with pytest.raises(ErroIndiceCorrompido):
        IndiceVetorial(tmp_path, MODELO).carregar()


@pytest.mark.parametrize(
    "versao", ["..", "../fora", "a/b", "a\\b", "/abs", "", 123, None]
)
def test_manifesto_com_versao_insegura_gera_erro_corrompido(
    tmp_path: Path, versao: object
) -> None:
    _indice_gravado(tmp_path)
    (tmp_path / "atual.json").write_text(
        json.dumps({"versao": versao, "versao_formato": 2}), encoding="utf-8"
    )
    with pytest.raises(ErroIndiceCorrompido):
        IndiceVetorial(tmp_path, MODELO).carregar()


@pytest.mark.parametrize("formato", [1, 3, "2", None])
def test_manifesto_com_versao_formato_diferente_gera_erro_corrompido(
    tmp_path: Path, formato: object
) -> None:
    _indice_gravado(tmp_path)
    manifesto = json.loads((tmp_path / "atual.json").read_text(encoding="utf-8"))
    manifesto["versao_formato"] = formato
    (tmp_path / "atual.json").write_text(json.dumps(manifesto), encoding="utf-8")
    with pytest.raises(ErroIndiceCorrompido):
        IndiceVetorial(tmp_path, MODELO).carregar()
