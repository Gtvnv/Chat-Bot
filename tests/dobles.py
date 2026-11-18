"""Dublês de teste compartilhados: embeddings, geração e leitura de PDF."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from pathlib import Path

from pdf_rag_chatbot.erros import ErroEmbeddings, ErroPdf, ErroProvedor


class EmbeddingsFalso:
    """Implementa ``ProvedorEmbeddings`` com vetores derivados de ``sha256``.

    O vetor depende só do texto e da dimensão, então textos iguais geram
    vetores iguais e textos distintos geram vetores distintos.
    """

    def __init__(
        self,
        dimensao: int = 16,
        nome_modelo: str = "embeddings-falso",
        falhar: bool = False,
        falhar_apos: int | None = None,
    ) -> None:
        if dimensao < 1:
            raise ValueError("a dimensão deve ser maior que zero")
        self.dimensao = dimensao
        self.nome_modelo = nome_modelo
        self.falhar = falhar
        self.falhar_apos = falhar_apos
        self.chamadas = 0
        self.textos_recebidos: list[str] = []

    def embutir(self, textos: Sequence[str]) -> list[list[float]]:
        self.chamadas += 1
        if self.falhar or (
            self.falhar_apos is not None and self.chamadas > self.falhar_apos
        ):
            raise ErroEmbeddings("falha injetada no gerador de embeddings falso")
        self.textos_recebidos.extend(textos)
        return [self.vetor_de(texto) for texto in textos]

    def vetor_de(self, texto: str) -> list[float]:
        """Vetor determinístico do texto, com valores em [-1, 1) e nunca nulo."""
        valores: list[float] = []
        contador = 0
        while len(valores) < self.dimensao:
            resumo = hashlib.sha256(
                f"{contador}:{texto}".encode("utf-8")
            ).digest()
            for (inteiro,) in struct.iter_unpack(">I", resumo):
                valores.append(inteiro / 2**31 - 1.0)
            contador += 1
        vetor = valores[: self.dimensao]
        if not any(vetor):
            vetor[0] = 1.0
        return vetor


class GeracaoFalsa:
    """Implementa ``ProvedorGeracao`` com respostas programadas.

    As respostas são consumidas em ordem; ao acabarem, a última se repete.
    Cada chamada registra as mensagens recebidas e o timeout informado.
    """

    SENTINELA = "SEM_RESPOSTA"

    def __init__(
        self,
        respostas: Sequence[str] = ("Resposta de teste [1].",),
        nome_modelo: str = "geracao-falsa",
        erro: Exception | None = None,
        simular_timeout: bool = False,
    ) -> None:
        if not respostas:
            raise ValueError("informe ao menos uma resposta programada")
        self.respostas = list(respostas)
        self.nome_modelo = nome_modelo
        self.erro = erro
        self.simular_timeout = simular_timeout
        self.prompts: list[list[object]] = []
        self.timeouts: list[float] = []

    @classmethod
    def com_sentinela(cls) -> GeracaoFalsa:
        return cls([cls.SENTINELA])

    @classmethod
    def com_erro(cls, mensagem: str = "falha injetada na geração") -> GeracaoFalsa:
        return cls(erro=ErroProvedor(mensagem))

    @classmethod
    def com_timeout(cls) -> GeracaoFalsa:
        return cls(simular_timeout=True)

    @property
    def chamadas(self) -> int:
        return len(self.prompts)

    def completar(self, mensagens: Sequence[object], timeout_s: float) -> str:
        self.prompts.append(list(mensagens))
        self.timeouts.append(timeout_s)
        if self.simular_timeout:
            raise ErroProvedor(
                f"tempo limite de {timeout_s:g} s excedido na geração (simulado)"
            )
        if self.erro is not None:
            raise self.erro
        indice = min(len(self.prompts) - 1, len(self.respostas) - 1)
        return self.respostas[indice]


class LeitorPdfFalso:
    """Implementa ``LeitorPdf`` com páginas programadas por nome de arquivo.

    Arquivos não registrados devolvem ``paginas_padrao``. Arquivos registrados
    como inválidos lançam ``ErroPdf``.
    """

    def __init__(
        self,
        paginas_por_arquivo: dict[str, list[str]] | None = None,
        paginas_padrao: list[str] | None = None,
        arquivos_invalidos: Sequence[str] = (),
    ) -> None:
        self.paginas_por_arquivo = dict(paginas_por_arquivo or {})
        self.paginas_padrao = paginas_padrao
        self.arquivos_invalidos = set(arquivos_invalidos)
        self.lidos: list[str] = []

    @staticmethod
    def gerar_paginas(
        quantidade: int, prefixo: str = "Texto da página"
    ) -> list[str]:
        """Páginas com texto único, numeradas a partir de 1."""
        return [f"{prefixo} {n}." for n in range(1, quantidade + 1)]

    @staticmethod
    def paginas_vazias(quantidade: int = 1) -> list[str]:
        """Páginas sem texto extraível, como em PDFs escaneados."""
        return [""] * quantidade

    def ler_paginas(self, caminho: Path) -> list[str]:
        nome = Path(caminho).name
        self.lidos.append(nome)
        if nome in self.arquivos_invalidos:
            raise ErroPdf(f"arquivo PDF ilegível: {nome}")
        if nome in self.paginas_por_arquivo:
            return list(self.paginas_por_arquivo[nome])
        if self.paginas_padrao is not None:
            return list(self.paginas_padrao)
        raise ErroPdf(f"arquivo PDF inexistente: {nome}")
