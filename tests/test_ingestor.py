"""Testes de exemplo para ``Ingestor`` e ``LeitorPdfPypdf``."""

from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from dobles import EmbeddingsFalso, LeitorPdfFalso
from pdf_rag_chatbot.embeddings.base import GeradorDeEmbeddings
from pdf_rag_chatbot.erros import ErroLimiteCaminhos, ErroPdf
from pdf_rag_chatbot.indice_vetorial import IndiceVetorial
from pdf_rag_chatbot.ingestor import Ingestor, LeitorPdfPypdf


def _criar_ingestor(tmp_path: Path, leitor) -> tuple[Ingestor, IndiceVetorial]:
    provedor = EmbeddingsFalso()
    indice = IndiceVetorial(tmp_path / "indice", provedor.nome_modelo)
    indice.carregar()
    ingestor = Ingestor(leitor, GeradorDeEmbeddings(provedor), indice, 200, 20)
    return ingestor, indice


def _gerar_pdf(caminho: Path, paginas: list[str]) -> None:
    """Gera um PDF real; texto vazio produz uma página em branco."""
    pdf = canvas.Canvas(str(caminho))
    for texto in paginas:
        if texto:
            pdf.drawString(72, 750, texto)
        pdf.showPage()
    pdf.save()


def test_diretorio_sem_pdfs_gera_aviso_e_nao_indexa(tmp_path):
    vazio = tmp_path / "vazio"
    vazio.mkdir()
    (vazio / "nota.docx").write_text("x", encoding="utf-8")
    leitor = LeitorPdfFalso(paginas_padrao=["Texto."])
    ingestor, indice = _criar_ingestor(tmp_path, leitor)

    resumo = ingestor.ingerir([vazio])

    assert resumo.sucessos == 0
    assert resumo.total_trechos == 0
    assert resumo.falhas_ou_ignorados == 1
    assert any("Nenhum PDF encontrado" in m for m in resumo.mensagens)
    assert leitor.lidos == []
    assert indice.esta_vazio()


def test_pdf_sem_texto_extraivel_e_ignorado_com_aviso(tmp_path):
    pdf = tmp_path / "escaneado.pdf"
    leitor = LeitorPdfFalso(paginas_por_arquivo={"escaneado.pdf": ["", "   "]})
    ingestor, indice = _criar_ingestor(tmp_path, leitor)

    resumo = ingestor.ingerir([pdf])

    assert resumo.sucessos == 0
    assert resumo.falhas_ou_ignorados == 1
    assert resumo.falhas == ()
    assert any("não tem texto extraível" in m for m in resumo.mensagens)
    assert indice.esta_vazio()


def test_pdf_real_sem_texto_extraivel_e_ignorado(tmp_path):
    pdf = tmp_path / "em_branco.pdf"
    _gerar_pdf(pdf, ["", ""])
    ingestor, indice = _criar_ingestor(tmp_path, LeitorPdfPypdf())

    resumo = ingestor.ingerir([pdf])

    assert resumo.sucessos == 0
    assert resumo.falhas_ou_ignorados == 1
    assert any("não tem texto extraível" in m for m in resumo.mensagens)
    assert indice.esta_vazio()


def test_arquivo_inexistente_vira_falha_e_nao_interrompe_os_demais(tmp_path):
    existente = tmp_path / "ok.pdf"
    _gerar_pdf(existente, ["Conteúdo de teste do arquivo existente."])
    inexistente = tmp_path / "nao_existe.pdf"
    ingestor, indice = _criar_ingestor(tmp_path, LeitorPdfPypdf())

    resumo = ingestor.ingerir([inexistente, existente])

    assert resumo.sucessos == 1
    assert resumo.falhas_ou_ignorados == 1
    assert [f.nome for f in resumo.falhas] == ["nao_existe.pdf"]
    assert indice.listar_fontes() == ["ok.pdf"]


def test_arquivo_txt_inexistente_vira_falha(tmp_path):
    ingestor, indice = _criar_ingestor(tmp_path, LeitorPdfFalso())

    resumo = ingestor.ingerir([tmp_path / "ausente.txt"])

    assert resumo.sucessos == 0
    assert len(resumo.falhas) == 1
    assert indice.esta_vazio()


def test_exatamente_50_caminhos_sao_aceitos(tmp_path):
    caminhos = []
    for i in range(50):
        caminho = tmp_path / f"doc{i:02d}.pdf"
        caminho.write_bytes(b"")
        caminhos.append(caminho)
    leitor = LeitorPdfFalso(paginas_padrao=["Texto de exemplo."])
    ingestor, indice = _criar_ingestor(tmp_path, leitor)

    resumo = ingestor.ingerir(caminhos)

    assert resumo.sucessos == 50
    assert resumo.falhas_ou_ignorados == 0
    assert len(leitor.lidos) == 50
    assert len(indice.listar_fontes()) == 50


def test_51_caminhos_sao_rejeitados_sem_abrir_arquivos(tmp_path):
    caminhos = []
    for i in range(51):
        caminho = tmp_path / f"doc{i:02d}.pdf"
        caminho.write_bytes(b"")
        caminhos.append(caminho)
    leitor = LeitorPdfFalso(paginas_padrao=["Texto de exemplo."])
    ingestor, indice = _criar_ingestor(tmp_path, leitor)

    with pytest.raises(ErroLimiteCaminhos) as info:
        ingestor.ingerir(caminhos)

    assert info.value.recebidos == 51
    assert info.value.limite == 50
    assert leitor.lidos == []
    assert indice.esta_vazio()


def test_leitor_pypdf_le_paginas_de_pdf_real(tmp_path):
    pdf = tmp_path / "real.pdf"
    _gerar_pdf(pdf, ["Primeira pagina sobre testes", "Segunda pagina sobre arquitetura"])

    paginas = LeitorPdfPypdf().ler_paginas(pdf)

    assert len(paginas) == 2
    assert "Primeira pagina sobre testes" in paginas[0]
    assert "Segunda pagina sobre arquitetura" in paginas[1]


def test_leitor_pypdf_arquivo_inexistente_lanca_erro_pdf(tmp_path):
    with pytest.raises(ErroPdf):
        LeitorPdfPypdf().ler_paginas(tmp_path / "ausente.pdf")


def test_ingestor_com_pdf_real_cria_trechos_com_fonte_e_pagina(tmp_path):
    pdf = tmp_path / "artigo.pdf"
    _gerar_pdf(pdf, ["Texto da pagina um", "", "Texto da pagina tres"])
    ingestor, indice = _criar_ingestor(tmp_path, LeitorPdfPypdf())

    resumo = ingestor.ingerir([pdf])

    assert resumo.sucessos == 1
    assert resumo.total_trechos == 2
    assert indice.listar_fontes() == ["artigo.pdf"]
