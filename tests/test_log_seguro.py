"""Testes de exemplo do filtro de segredos (Requisito 6.3)."""

from __future__ import annotations

import logging

from pdf_rag_chatbot.log_seguro import MASCARA, FiltroSegredos

SEGREDO = "sk-segredo-123"
OUTRO = "gsk-outro-456"


def _registro(msg, args=None, exc_info=None) -> logging.LogRecord:
    return logging.LogRecord("teste", logging.ERROR, __file__, 1, msg, args, exc_info)


def _saida(registro: logging.LogRecord) -> str:
    return logging.Formatter("%(message)s").format(registro)


def test_segredo_na_mensagem_e_mascarado():
    registro = _registro(f"chave={SEGREDO}")
    assert FiltroSegredos([SEGREDO]).filter(registro) is True
    assert _saida(registro) == f"chave={MASCARA}"


def test_segredo_nos_argumentos_e_mascarado():
    registro = _registro("chave=%s", (SEGREDO,))
    FiltroSegredos([SEGREDO]).filter(registro)
    saida = _saida(registro)
    assert SEGREDO not in saida
    assert saida == f"chave={MASCARA}"


def test_segredo_no_texto_da_excecao_e_mascarado():
    try:
        raise ValueError(f"falha com {SEGREDO}")
    except ValueError:
        import sys

        registro = _registro("erro", exc_info=sys.exc_info())
    FiltroSegredos([SEGREDO]).filter(registro)
    saida = logging.Formatter("%(message)s").format(registro)
    assert SEGREDO not in saida
    assert MASCARA in saida
    assert "ValueError" in saida


def test_multiplos_segredos_sao_mascarados():
    registro = _registro("a=%s b=%s", (SEGREDO, OUTRO))
    FiltroSegredos([SEGREDO, OUTRO]).filter(registro)
    saida = _saida(registro)
    assert SEGREDO not in saida
    assert OUTRO not in saida
    assert saida == f"a={MASCARA} b={MASCARA}"
