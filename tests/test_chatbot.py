"""Testes de exemplo para ``Chatbot``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pdf_rag_chatbot import mensagens
from pdf_rag_chatbot.chatbot import Chatbot
from pdf_rag_chatbot.embeddings import GeradorDeEmbeddings
from pdf_rag_chatbot.erros import ErroPergunta
from pdf_rag_chatbot.geracao.base import GeradorDeRespostas
from pdf_rag_chatbot.indice_vetorial import IndiceVetorial
from pdf_rag_chatbot.modelos import Trecho
from pdf_rag_chatbot.recuperador import Recuperador

from dobles import EmbeddingsFalso, GeracaoFalsa

PERGUNTA_COMPARATIVA = "Compare a.pdf e b.pdf sobre testes"


def _montar(diretorio: Path, respostas: list[str], trechos: list[Trecho]):
    """Chatbot real com índice populado e dublês de embeddings e geração."""
    provedor = EmbeddingsFalso()
    indice = IndiceVetorial(diretorio / "indice", provedor.nome_modelo)
    indice.carregar()
    fontes = list(dict.fromkeys(t.fonte for t in trechos))
    for fonte in fontes:
        doc = [t for t in trechos if t.fonte == fonte]
        vetores = np.array([provedor.vetor_de(t.texto) for t in doc], dtype=np.float32)
        indice.substituir_fonte(fonte, doc, vetores)
    recuperador = Recuperador(GeradorDeEmbeddings(provedor), indice, 4)
    geracao = GeracaoFalsa(respostas)
    return Chatbot(recuperador, GeradorDeRespostas(geracao), indice), geracao


def test_fluxo_feliz_devolve_resposta_com_citacoes(tmp_path: Path) -> None:
    trecho = Trecho("Testes automatizados reduzem regressões.", "guia.pdf", 3)
    chatbot, geracao = _montar(tmp_path, ["Reduzem regressões [1]."], [trecho])

    resposta = chatbot.responder("  O que reduz regressões?  ")

    assert resposta.texto == "Reduzem regressões [1]."
    assert [(c.fonte, c.pagina) for c in resposta.citacoes] == [("guia.pdf", 3)]
    assert geracao.chamadas == 1
    assert len(chatbot.historico) == 1
    assert chatbot.historico[0].pergunta == "O que reduz regressões?"
    assert chatbot.historico[0].resposta == resposta.texto


def test_comparativa_com_reforco_cita_os_dois_arquivos(tmp_path: Path) -> None:
    trechos = [
        Trecho(PERGUNTA_COMPARATIVA, "a.pdf", 1),
        Trecho("Conteúdo diferente do segundo arquivo.", "b.pdf", 2),
    ]
    # O trecho de a.pdf é idêntico à pergunta, então é o [1]; b.pdf é o [2].
    chatbot, geracao = _montar(
        tmp_path, ["Só o primeiro [1].", "Os dois arquivos [1] [2]."], trechos
    )

    resposta = chatbot.responder(PERGUNTA_COMPARATIVA)

    assert geracao.chamadas == 2
    assert {c.fonte for c in resposta.citacoes} == {"a.pdf", "b.pdf"}
    assert resposta.texto == "Os dois arquivos [1] [2]."
    assert "Aviso" not in resposta.texto


def test_comparativa_com_arquivo_sem_citacao_acrescenta_aviso(tmp_path: Path) -> None:
    trechos = [
        Trecho(PERGUNTA_COMPARATIVA, "a.pdf", 1),
        Trecho("Conteúdo diferente do segundo arquivo.", "b.pdf", 2),
    ]
    chatbot, geracao = _montar(tmp_path, ["Só o primeiro [1]."], trechos)

    resposta = chatbot.responder(PERGUNTA_COMPARATIVA)

    assert geracao.chamadas == 2  # reforço único
    assert [c.fonte for c in resposta.citacoes] == ["a.pdf"]
    assert resposta.texto.endswith(mensagens.aviso_arquivos_sem_citacao(["b.pdf"]))
    assert chatbot.historico[-1].resposta == resposta.texto


def test_pergunta_com_exatamente_2000_caracteres_e_aceita(tmp_path: Path) -> None:
    trecho = Trecho("Conteúdo qualquer.", "doc.pdf", 1)
    chatbot, geracao = _montar(tmp_path, ["Ok [1]."], [trecho])

    resposta = chatbot.responder("a" * 2000)

    assert resposta.texto == "Ok [1]."
    assert geracao.chamadas == 1


def test_pergunta_com_2001_caracteres_e_rejeitada(tmp_path: Path) -> None:
    trecho = Trecho("Conteúdo qualquer.", "doc.pdf", 1)
    chatbot, geracao = _montar(tmp_path, ["Ok [1]."], [trecho])

    with pytest.raises(ErroPergunta) as erro:
        chatbot.responder("a" * 2001)

    assert str(erro.value) == mensagens.PERGUNTA_LONGA
    assert geracao.chamadas == 0
    assert chatbot.historico == ()


def test_resposta_com_exatamente_2000_caracteres_e_preservada(tmp_path: Path) -> None:
    trecho = Trecho("Conteúdo qualquer.", "doc.pdf", 1)
    texto = "[1] " + "x" * 1996
    assert len(texto) == 2000
    chatbot, _ = _montar(tmp_path, [texto], [trecho])

    resposta = chatbot.responder("Pergunta?")

    assert resposta.texto == texto
    assert len(resposta.citacoes) == 1


def test_resposta_acima_de_2000_caracteres_e_limitada(tmp_path: Path) -> None:
    trecho = Trecho("Conteúdo qualquer.", "doc.pdf", 1)
    texto = "[1] " + "palavra " * 400
    assert len(texto) > 2000
    chatbot, _ = _montar(tmp_path, [texto], [trecho])

    resposta = chatbot.responder("Pergunta?")

    assert len(resposta.texto) <= 2000
    assert resposta.citacoes
