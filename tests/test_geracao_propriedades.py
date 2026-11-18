"""Testes de propriedade para o Gerador_de_Respostas."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from pdf_rag_chatbot.geracao.base import GeradorDeRespostas
from pdf_rag_chatbot.modelos import ResultadoBusca, Trecho
from dobles import GeracaoFalsa


@st.composite
def _entrada_geracao(draw):
    """Gera pergunta, de 1 a 10 Trechos e um texto arbitrário do provedor."""
    quantidade = draw(st.integers(min_value=1, max_value=10))
    trechos = []
    for i in range(1, quantidade + 1):
        corpo = draw(st.text(alphabet="abc xyz.,", min_size=1, max_size=40))
        trecho = Trecho(
            texto=f"conteudo{i} {corpo}",
            fonte=f"arquivo{i}.pdf",
            pagina=draw(st.integers(min_value=1, max_value=50)),
            tipo="pdf",
        )
        trechos.append(ResultadoBusca(trecho=trecho, similaridade=0.5))
    pergunta = "Pergunta: " + draw(st.text(alphabet="abc xyz?", min_size=1, max_size=40))
    texto = draw(st.text(min_size=0, max_size=5000))
    return pergunta, trechos, texto


# Feature: pdf-rag-chatbot, Property 16: Resposta limitada e prompt completo
# **Validates: Requirements 3.1**
@settings(max_examples=100)
@given(_entrada_geracao())
def test_resposta_limitada_e_prompt_completo(entrada):
    pergunta, trechos, texto = entrada
    provedor = GeracaoFalsa([texto])

    resposta = GeradorDeRespostas(provedor).gerar(pergunta, trechos, [])

    assert len(resposta) <= 2000

    assert provedor.chamadas == 1
    mensagens = provedor.prompts[0]
    conteudo_final = mensagens[-1]["content"]
    assert pergunta in conteudo_final
    for numero, resultado in enumerate(trechos, start=1):
        t = resultado.trecho
        assert f"[{numero}] ({t.fonte}, p. {t.pagina}): {t.texto}" in conteudo_final
