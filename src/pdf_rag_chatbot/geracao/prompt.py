"""Montagem pura do prompt enviado ao provedor de geração."""

from __future__ import annotations

from typing import Sequence, TypedDict

from pdf_rag_chatbot.modelos import Interacao, ResultadoBusca

SENTINELA_SEM_RESPOSTA = "SEM_RESPOSTA"
JANELA_HISTORICO = 5


class Mensagem(TypedDict):
    """Mensagem de conversa no formato aceito pelos provedores (role/content)."""

    role: str
    content: str


MENSAGEM_SISTEMA = (
    "Você é um assistente que responde perguntas sobre documentos. "
    "Responda sempre em português.\n"
    "Use apenas as informações dos Trechos fornecidos na mensagem do usuário. "
    "Não use conhecimento externo.\n"
    "Marque cada afirmação com o número do Trecho de onde ela vem, no formato [n], "
    "por exemplo [1] ou [2].\n"
    "Quando citar literalmente um texto escrito em outro idioma, mantenha a citação "
    "entre aspas no idioma original, sem traduzi-la.\n"
    f"Se os Trechos não bastarem para responder, responda exatamente {SENTINELA_SEM_RESPOSTA}, "
    "sem mais nenhum texto.\n"
    "Quando a pergunta relacionar ou comparar arquivos, cite ao menos um Trecho "
    "de cada arquivo comparado."
)


def formatar_trecho(numero: int, resultado: ResultadoBusca) -> str:
    """Formata um Trecho como `[n] (arquivo, p. N): texto`."""
    trecho = resultado.trecho
    return f"[{numero}] ({trecho.fonte}, p. {trecho.pagina}): {trecho.texto}"


def montar_prompt(
    pergunta: str,
    trechos: Sequence[ResultadoBusca],
    historico: Sequence[Interacao],
) -> list[Mensagem]:
    """Monta as mensagens: sistema, últimas 5 interações (cronológicas) e pergunta."""
    mensagens: list[Mensagem] = [{"role": "system", "content": MENSAGEM_SISTEMA}]

    recentes = list(historico)[-JANELA_HISTORICO:]
    for interacao in recentes:
        mensagens.append({"role": "user", "content": interacao.pergunta})
        mensagens.append({"role": "assistant", "content": interacao.resposta})

    blocos = "\n".join(formatar_trecho(n, r) for n, r in enumerate(trechos, start=1))
    final = f"Trechos:\n{blocos}\n\nPergunta: {pergunta}"
    mensagens.append({"role": "user", "content": final})
    return mensagens
