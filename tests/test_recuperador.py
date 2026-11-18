"""Testes de exemplo para ``Recuperador``."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdf_rag_chatbot.embeddings import GeradorDeEmbeddings
from pdf_rag_chatbot.erros import ErroEmbeddings
from pdf_rag_chatbot.indice_vetorial import IndiceVetorial
from pdf_rag_chatbot.modelos import Trecho
from pdf_rag_chatbot.recuperador import Recuperador, fontes_mencionadas

from dobles import EmbeddingsFalso


def _montar(
    diretorio: Path,
    trechos_por_fonte: dict[str, list[str]],
    top_k: int = 4,
    provedor: EmbeddingsFalso | None = None,
) -> tuple[Recuperador, IndiceVetorial, EmbeddingsFalso]:
    """Cria um índice populado e o ``Recuperador`` correspondente."""
    provedor = provedor or EmbeddingsFalso()
    embeddings = GeradorDeEmbeddings(provedor)
    indice = IndiceVetorial(diretorio, provedor.nome_modelo)
    indice.carregar()
    for fonte, textos in trechos_por_fonte.items():
        trechos = [
            Trecho(texto=texto, fonte=fonte, pagina=n + 1)
            for n, texto in enumerate(textos)
        ]
        indice.substituir_fonte(fonte, trechos, embeddings.gerar(textos))
    return Recuperador(embeddings, indice, top_k), indice, provedor


def test_recupera_trechos_mais_similares_em_ordem_decrescente(
    diretorio_indice: Path,
) -> None:
    textos = [f"conteúdo distinto número {n}" for n in range(8)]
    recuperador, _, _ = _montar(diretorio_indice, {"artigo.pdf": textos}, top_k=3)

    resultados = recuperador.recuperar(textos[5])

    assert len(resultados) == 3
    # a pergunta idêntica a um Trecho o devolve em primeiro lugar
    assert resultados[0].trecho.texto == textos[5]
    assert resultados[0].similaridade == pytest.approx(1.0, abs=1e-5)
    similaridades = [r.similaridade for r in resultados]
    assert similaridades == sorted(similaridades, reverse=True)


def test_indice_com_menos_trechos_que_top_k_devolve_todos(
    diretorio_indice: Path,
) -> None:
    textos = ["primeiro trecho", "segundo trecho"]
    recuperador, indice, _ = _montar(
        diretorio_indice, {"curto.pdf": textos}, top_k=10
    )

    resultados = recuperador.recuperar("qualquer pergunta")

    assert len(indice) == 2
    assert len(resultados) == 2
    assert {r.trecho.texto for r in resultados} == set(textos)


def test_falha_no_embedding_da_pergunta_propaga_sem_alterar_o_indice(
    diretorio_indice: Path,
) -> None:
    recuperador, indice, provedor = _montar(
        diretorio_indice, {"artigo.pdf": ["trecho um", "trecho dois"]}
    )
    fontes_antes = indice.listar_fontes()
    tamanho_antes = len(indice)
    arquivos_antes = {
        p.name: p.read_bytes() for p in diretorio_indice.iterdir() if p.is_file()
    }

    provedor.falhar = True
    with pytest.raises(ErroEmbeddings):
        recuperador.recuperar("pergunta que falha")

    assert indice.listar_fontes() == fontes_antes
    assert len(indice) == tamanho_antes
    arquivos_depois = {
        p.name: p.read_bytes() for p in diretorio_indice.iterdir() if p.is_file()
    }
    assert arquivos_depois == arquivos_antes


def test_nome_de_arquivo_com_caixa_diferente_recupera_cada_arquivo(
    diretorio_indice: Path,
) -> None:
    recuperador, indice, _ = _montar(
        diretorio_indice,
        {
            "Relatorio_A.pdf": [f"texto do relatório A {n}" for n in range(5)],
            "Relatorio_B.pdf": [f"texto do relatório B {n}" for n in range(5)],
            "Outro.pdf": [f"texto de outro documento {n}" for n in range(5)],
        },
        top_k=1,
    )

    pergunta = "Compare RELATORIO_A com relatorio_b.PDF"
    assert fontes_mencionadas(pergunta, indice.listar_fontes()) == [
        "Relatorio_A.pdf",
        "Relatorio_B.pdf",
    ]

    resultados = recuperador.recuperar(pergunta)

    fontes = {r.trecho.fonte for r in resultados}
    assert fontes == {"Relatorio_A.pdf", "Relatorio_B.pdf"}
    assert len(resultados) <= 10
    similaridades = [r.similaridade for r in resultados]
    assert similaridades == sorted(similaridades, reverse=True)


def test_um_unico_arquivo_mencionado_usa_busca_com_top_k(
    diretorio_indice: Path,
) -> None:
    recuperador, _, _ = _montar(
        diretorio_indice,
        {
            "manual_alfa.pdf": [f"trecho alfa {n}" for n in range(5)],
            "guia_beta.pdf": [f"trecho beta {n}" for n in range(5)],
        },
        top_k=2,
    )

    resultados = recuperador.recuperar("Fale sobre MANUAL_ALFA.PDF")

    assert len(resultados) == 2
