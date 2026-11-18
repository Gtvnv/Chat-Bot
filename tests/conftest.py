"""Fixtures e configuração compartilhadas pelos testes."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

from dobles import EmbeddingsFalso, GeracaoFalsa, LeitorPdfFalso

# Perfil padrão do Hypothesis: 100 exemplos por propriedade.
settings.register_profile(
    "padrao",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.load_profile("padrao")


@pytest.fixture
def diretorio_indice(tmp_path: Path) -> Path:
    """Diretório temporário (ainda inexistente) para o índice vetorial."""
    return tmp_path / "index"


@pytest.fixture
def diretorio_pdfs(tmp_path: Path) -> Path:
    """Diretório temporário existente para o teste de descoberta de PDFs."""
    diretorio = tmp_path / "pdfs"
    diretorio.mkdir()
    return diretorio


@pytest.fixture
def embeddings_falso() -> EmbeddingsFalso:
    return EmbeddingsFalso()


@pytest.fixture
def geracao_falsa() -> GeracaoFalsa:
    return GeracaoFalsa()


@pytest.fixture
def leitor_pdf_falso() -> LeitorPdfFalso:
    return LeitorPdfFalso()


@pytest.fixture
def indice_vazio(diretorio_indice: Path, embeddings_falso: EmbeddingsFalso):
    """Índice vetorial vazio, compatível com ``EmbeddingsFalso``.

    O import é tardio porque o módulo do índice pode ainda não existir.
    """
    from pdf_rag_chatbot.indice_vetorial import IndiceVetorial

    indice = IndiceVetorial(diretorio_indice, embeddings_falso.nome_modelo)
    indice.carregar()
    return indice
