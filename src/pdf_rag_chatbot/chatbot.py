"""Chatbot: orquestra a consulta (validação, recuperação, geração e histórico)."""

from __future__ import annotations

from enum import Enum

from pdf_rag_chatbot import mensagens
from pdf_rag_chatbot.citacoes import extrair_citacoes
from pdf_rag_chatbot.erros import ErroPergunta
from pdf_rag_chatbot.geracao.base import GeradorDeRespostas
from pdf_rag_chatbot.geracao.prompt import JANELA_HISTORICO as _JANELA, SENTINELA_SEM_RESPOSTA
from pdf_rag_chatbot.indice_vetorial import IndiceVetorial
from pdf_rag_chatbot.modelos import Interacao, RespostaChatbot
from pdf_rag_chatbot.recuperador import Recuperador, fontes_mencionadas

__all__ = [
    "Chatbot",
    "Comando",
    "Interacao",
    "normalizar_pergunta",
    "reconhecer_comando",
    "validar_pergunta",
]


class Comando(str, Enum):
    SAIR = "sair"
    DOCUMENTOS = "documentos"
    CARREGAR = "carregar"
    TOPK = "topk"


def normalizar_pergunta(entrada: str) -> str:
    """Remove espaços no início e no fim."""
    return entrada.strip()


def validar_pergunta(entrada: str) -> str:
    """Devolve a pergunta normalizada ou lança `ErroPergunta` (vazia ou > 2.000)."""
    pergunta = normalizar_pergunta(entrada)
    if not pergunta:
        raise ErroPergunta(mensagens.PERGUNTA_VAZIA)
    if len(pergunta) > mensagens.LIMITE_PERGUNTA:
        raise ErroPergunta(mensagens.PERGUNTA_LONGA)
    return pergunta


def reconhecer_comando(entrada: str) -> Comando | None:
    """Reconhece comandos de sessão sem diferenciar caixa e ignorando espaços nas pontas.

    `sair` e `documentos` valem só quando são a entrada inteira; `carregar` e
    `topk` aceitam argumentos depois do nome do comando.
    """
    texto = normalizar_pergunta(entrada).casefold()
    if texto == Comando.SAIR.value:
        return Comando.SAIR
    if texto == Comando.DOCUMENTOS.value:
        return Comando.DOCUMENTOS
    palavras = texto.split(maxsplit=1)
    if palavras and palavras[0] == Comando.CARREGAR.value:
        return Comando.CARREGAR
    if palavras and palavras[0] == Comando.TOPK.value:
        return Comando.TOPK
    return None


class Chatbot:
    """Responde perguntas com base no índice e mantém o histórico da sessão."""

    JANELA_HISTORICO = _JANELA

    def __init__(
        self,
        recuperador: Recuperador,
        gerador: GeradorDeRespostas,
        indice: IndiceVetorial,
    ) -> None:
        self._recuperador = recuperador
        self._gerador = gerador
        self._indice = indice
        self._historico: list[Interacao] = []

    @property
    def historico(self) -> tuple[Interacao, ...]:
        return tuple(self._historico)

    def limpar_historico(self) -> None:
        self._historico.clear()

    def responder(self, entrada: str) -> RespostaChatbot:
        pergunta = validar_pergunta(entrada)

        if self._indice.esta_vazio():
            return RespostaChatbot(mensagens.INDICE_VAZIO)

        # Trechos e Citações vivem só nesta chamada; exceções os descartam.
        trechos = self._recuperador.recuperar(pergunta)
        if not trechos:
            return self._registrar(pergunta, RespostaChatbot(mensagens.NAO_ENCONTRADA))

        janela = self._historico[-self.JANELA_HISTORICO :]
        texto = self._gerador.gerar(pergunta, trechos, janela)
        if self._e_sentinela(texto):
            return self._registrar(pergunta, RespostaChatbot(mensagens.NAO_ENCONTRADA))

        citacoes = extrair_citacoes(texto, trechos)
        if not citacoes:
            return self._registrar(pergunta, RespostaChatbot(mensagens.NAO_FUNDAMENTADA))

        mencionadas = fontes_mencionadas(pergunta, self._indice.listar_fontes())
        aviso = ""
        if len(mencionadas) >= 2:
            faltando = self._faltando(mencionadas, citacoes)
            if faltando:
                reforco = mensagens.REFORCO_COMPARATIVA.format(arquivos=", ".join(faltando))
                texto2 = self._gerador.gerar(f"{pergunta}\n\n{reforco}", trechos, janela)
                if not self._e_sentinela(texto2):
                    citacoes2 = extrair_citacoes(texto2, trechos)
                    if citacoes2:
                        texto, citacoes = texto2, citacoes2
                faltando = self._faltando(mencionadas, citacoes)
                if faltando:
                    aviso = mensagens.aviso_arquivos_sem_citacao(faltando)

        if aviso:
            texto = f"{texto}\n\n{aviso}"
        return self._registrar(pergunta, RespostaChatbot(texto, tuple(citacoes)))

    @staticmethod
    def _e_sentinela(texto: str) -> bool:
        return texto.strip() == SENTINELA_SEM_RESPOSTA

    @staticmethod
    def _faltando(mencionadas: list[str], citacoes: list) -> list[str]:
        citadas = {c.fonte for c in citacoes}
        return [f for f in mencionadas if f not in citadas]

    def _registrar(self, pergunta: str, resposta: RespostaChatbot) -> RespostaChatbot:
        self._historico.append(Interacao(pergunta, resposta.texto))
        return resposta
