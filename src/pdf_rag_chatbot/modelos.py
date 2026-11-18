"""Modelos de dados imutáveis do PDF RAG Chatbot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TipoFonte = Literal["pdf", "texto"]


@dataclass(frozen=True)
class Trecho:
    texto: str
    fonte: str  # nome do arquivo (ex.: "artigo.pdf")
    pagina: int  # >= 1; para o Documento_de_Sentencas, número da linha
    tipo: TipoFonte = "pdf"


@dataclass(frozen=True)
class Citacao:
    fonte: str
    pagina: int
    tipo: TipoFonte = "pdf"


@dataclass(frozen=True)
class ResultadoBusca:
    trecho: Trecho
    similaridade: float


@dataclass(frozen=True)
class FalhaArquivo:
    nome: str
    motivo: str


@dataclass(frozen=True)
class ResumoIngestao:
    sucessos: int = 0
    falhas_ou_ignorados: int = 0
    total_trechos: int = 0
    mensagens: tuple[str, ...] = field(default_factory=tuple)
    falhas: tuple[FalhaArquivo, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Interacao:
    pergunta: str
    resposta: str


@dataclass(frozen=True)
class RespostaChatbot:
    texto: str
    citacoes: tuple[Citacao, ...] = field(default_factory=tuple)
