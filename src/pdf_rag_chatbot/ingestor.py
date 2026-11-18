"""Ingestão de PDFs e do Documento_de_Sentencas: leitura, chunking e indexação."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from pdf_rag_chatbot import mensagens
from pdf_rag_chatbot.chunking import dividir_em_trechos
from pdf_rag_chatbot.config import validar_parametros_chunking
from pdf_rag_chatbot.embeddings.base import GeradorDeEmbeddings
from pdf_rag_chatbot.erros import ErroChatbot, ErroLimiteCaminhos, ErroPdf
from pdf_rag_chatbot.indice_vetorial import IndiceVetorial
from pdf_rag_chatbot.modelos import FalhaArquivo, ResumoIngestao, Trecho


@runtime_checkable
class LeitorPdf(Protocol):
    """Contrato de um leitor que devolve o texto de cada página, em ordem."""

    def ler_paginas(self, caminho: Path) -> list[str]: ...


class LeitorPdfPypdf:
    """Implementa ``LeitorPdf`` com ``pypdf.PdfReader``."""

    def ler_paginas(self, caminho: Path) -> list[str]:
        from pypdf import PdfReader

        try:
            leitor = PdfReader(str(caminho))
            return [pagina.extract_text() or "" for pagina in leitor.pages]
        except Exception as exc:  # arquivo inexistente, corrompido ou criptografado
            raise ErroPdf(
                f"arquivo PDF inexistente ou ilegível: {Path(caminho).name} ({exc})"
            ) from exc


class _ArquivoIgnorado(Exception):
    """Arquivo ignorado com aviso (não é falha)."""


class Ingestor:
    LIMITE_CAMINHOS = 50

    def __init__(
        self,
        leitor: LeitorPdf,
        embeddings: GeradorDeEmbeddings,
        indice: IndiceVetorial,
        tamanho: int,
        sobreposicao: int,
    ) -> None:
        validar_parametros_chunking(tamanho, sobreposicao)  # ErroConfiguracao
        self._leitor = leitor
        self._embeddings = embeddings
        self._indice = indice
        self._tamanho = tamanho
        self._sobreposicao = sobreposicao

    def ingerir(self, caminhos: Sequence[Path]) -> ResumoIngestao:
        caminhos = list(caminhos)
        if len(caminhos) > self.LIMITE_CAMINHOS:
            raise ErroLimiteCaminhos(len(caminhos), self.LIMITE_CAMINHOS)

        avisos: list[str] = []
        falhas: list[FalhaArquivo] = []
        sucessos = 0
        ignorados = 0
        total_trechos = 0

        arquivos: list[Path] = []
        for caminho in caminhos:
            caminho = Path(caminho)
            if caminho.is_dir():
                pdfs = sorted(
                    (
                        f
                        for f in caminho.iterdir()
                        if f.is_file() and f.suffix.lower() == ".pdf"
                    ),
                    key=lambda f: f.name,
                )
                if not pdfs:
                    avisos.append(
                        mensagens.nenhum_pdf_encontrado(caminho.name or str(caminho))
                    )
                    ignorados += 1
                arquivos.extend(pdfs)
            else:
                arquivos.append(caminho)

        for arquivo in arquivos:
            nome = arquivo.name
            try:
                trechos = self._extrair_trechos(arquivo)
                vetores = self._embeddings.gerar([t.texto for t in trechos])
                self._indice.substituir_fonte(nome, trechos, vetores)
            except _ArquivoIgnorado as aviso:
                ignorados += 1
                avisos.append(str(aviso))
            except Exception as exc:  # isolamento por arquivo; estado anterior preservado
                # Só exceções previstas têm texto seguro; o texto bruto de
                # exceções inesperadas pode conter segredos (ex.: chave de API).
                if isinstance(exc, ErroChatbot):
                    motivo = str(exc) or exc.__class__.__name__
                else:
                    motivo = exc.__class__.__name__
                falhas.append(FalhaArquivo(nome, motivo))
                ignorados += 1
                avisos.append(mensagens.falha_arquivo(nome, motivo))
            else:
                sucessos += 1
                total_trechos += len(trechos)
                avisos.append(mensagens.arquivo_indexado(nome, len(trechos)))

        return ResumoIngestao(
            sucessos=sucessos,
            falhas_ou_ignorados=ignorados,
            total_trechos=total_trechos,
            mensagens=tuple(avisos),
            falhas=tuple(falhas),
        )

    # ------------------------------------------------------------------
    def _extrair_trechos(self, arquivo: Path) -> list[Trecho]:
        extensao = arquivo.suffix.lower()
        if extensao == ".pdf":
            return self._trechos_pdf(arquivo)
        if extensao == ".txt":
            return self._trechos_texto(arquivo)
        if not arquivo.exists():
            raise ErroPdf(mensagens.arquivo_inexistente(str(arquivo)))
        raise ErroPdf(mensagens.formato_nao_suportado(arquivo.name))

    def _trechos_pdf(self, arquivo: Path) -> list[Trecho]:
        paginas = self._leitor.ler_paginas(arquivo)
        trechos: list[Trecho] = []
        for numero, texto in enumerate(paginas, start=1):
            if not texto.strip():
                continue
            for parte in dividir_em_trechos(texto, self._tamanho, self._sobreposicao):
                trechos.append(Trecho(parte, arquivo.name, numero, "pdf"))
        if not trechos:
            raise _ArquivoIgnorado(mensagens.sem_texto_extraivel(arquivo.name))
        return trechos

    def _trechos_texto(self, arquivo: Path) -> list[Trecho]:
        try:
            conteudo = arquivo.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ErroPdf("não foi possível decodificar o arquivo como UTF-8") from exc
        except OSError as exc:
            raise ErroPdf(
                f"arquivo inexistente ou ilegível ({exc.__class__.__name__})"
            ) from exc
        trechos = [
            Trecho(linha.strip(), arquivo.name, numero, "texto")
            for numero, linha in enumerate(conteudo.splitlines(), start=1)
            if linha.strip()
        ]
        if not trechos:
            raise ErroPdf("o arquivo não contém nenhuma linha com texto")
        return trechos
