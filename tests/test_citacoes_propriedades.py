"""Testes de propriedade para as Citações."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from pdf_rag_chatbot.citacoes import extrair_citacoes, formatar_citacao
from pdf_rag_chatbot.modelos import ResultadoBusca, Trecho

# Poucas fontes e páginas para provocar repetições de (fonte, pagina).
_FONTES = [("a.pdf", "pdf"), ("b.pdf", "pdf"), ("sentencas.txt", "texto")]


@st.composite
def _trecho_resultado(draw):
    fonte, tipo = draw(st.sampled_from(_FONTES))
    pagina = draw(st.integers(min_value=1, max_value=3))
    trecho = Trecho(texto="texto", fonte=fonte, pagina=pagina, tipo=tipo)
    return ResultadoBusca(trecho=trecho, similaridade=0.5)


@st.composite
def _respostas_e_trechos(draw):
    """Gera Trechos e uma resposta com marcadores válidos, inválidos e repetidos."""
    trechos = draw(st.lists(_trecho_resultado(), min_size=0, max_size=8))
    marcadores = draw(st.lists(st.integers(min_value=0, max_value=len(trechos) + 5), max_size=15))
    pedacos = []
    for numero in marcadores:
        pedacos.append(draw(st.text(alphabet="abc xyz.,", max_size=10)))
        pedacos.append(f"[{numero}]")
    pedacos.append(draw(st.text(alphabet="abc xyz.,", max_size=10)))
    return "".join(pedacos), marcadores, trechos


# Feature: pdf-rag-chatbot, Property 17: Citações só dos Trechos usados, sem repetição
# **Validates: Requirements 3.3**
@settings(max_examples=100)
@given(_respostas_e_trechos())
def test_citacoes_so_dos_trechos_usados_sem_repeticao(entrada):
    resposta, marcadores, trechos = entrada

    citacoes = extrair_citacoes(resposta, trechos)

    # Esperado: Trechos de marcadores válidos, sem repetir (fonte, pagina),
    # na ordem da primeira ocorrência.
    esperado = []
    for numero in marcadores:
        if 1 <= numero <= len(trechos):
            t = trechos[numero - 1].trecho
            if (t.fonte, t.pagina) not in esperado:
                esperado.append((t.fonte, t.pagina))
    assert [(c.fonte, c.pagina) for c in citacoes] == esperado

    # Sem duplicatas e apenas de Trechos realmente referenciados.
    chaves = [(c.fonte, c.pagina) for c in citacoes]
    assert len(chaves) == len(set(chaves))
    usados = {
        (trechos[n - 1].trecho.fonte, trechos[n - 1].trecho.pagina)
        for n in marcadores
        if 1 <= n <= len(trechos)
    }
    assert set(chaves) <= usados

    # Cada Citação formatada traz o nome do arquivo e a localização.
    for c in citacoes:
        texto = formatar_citacao(c)
        assert c.fonte in texto
        assert str(c.pagina) in texto
        assert ("linha" in texto) == (c.tipo == "texto")
