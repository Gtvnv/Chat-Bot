"""Interface de provedores de geração e o Gerador_de_Respostas."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from pdf_rag_chatbot.geracao.prompt import (
    SENTINELA_SEM_RESPOSTA,
    Mensagem,
    montar_prompt,
)
from pdf_rag_chatbot.modelos import Interacao, ResultadoBusca
from pdf_rag_chatbot.tipos import TempoLimite

LIMITE_CARACTERES = 2000
TIMEOUT_PADRAO_S = 60.0


@runtime_checkable
class ProvedorGeracao(Protocol):
    """Contrato de um provedor que completa uma conversa."""

    nome_modelo: str

    def completar(self, mensagens: Sequence[Mensagem], timeout_s: TempoLimite) -> str:
        """Devolve o texto da resposta ou lança `ErroProvedor`."""
        ...


def limitar_resposta(texto: str, limite: int = LIMITE_CARACTERES) -> str:
    """Limita o texto a `limite` caracteres, cortando no último espaço antes do limite."""
    if len(texto) <= limite:
        return texto
    corte = texto[:limite]
    # Se o caractere seguinte é espaço, o corte já cai em fronteira de palavra.
    if texto[limite].isspace():
        return corte.rstrip()
    posicao = max((i for i, c in enumerate(corte) if c.isspace()), default=-1)
    if posicao > 0:
        return corte[:posicao].rstrip()
    return corte


class GeradorDeRespostas:
    """Redige a resposta a partir dos Trechos recuperados."""

    LIMITE_CARACTERES = LIMITE_CARACTERES
    SENTINELA_SEM_RESPOSTA = SENTINELA_SEM_RESPOSTA

    def __init__(self, provedor: ProvedorGeracao, timeout_s: TempoLimite = TIMEOUT_PADRAO_S) -> None:
        self._provedor = provedor
        self._timeout_s = timeout_s

    def gerar(
        self,
        pergunta: str,
        trechos: Sequence[ResultadoBusca],
        historico: Sequence[Interacao],
    ) -> str:
        mensagens = montar_prompt(pergunta, trechos, historico)
        texto = self._provedor.completar(mensagens, self._timeout_s)
        return limitar_resposta(texto, self.LIMITE_CARACTERES)
