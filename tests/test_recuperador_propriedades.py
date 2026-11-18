"""Testes de propriedade para ``Recuperador``."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pdf_rag_chatbot.config import validar_top_k
from pdf_rag_chatbot.embeddings import GeradorDeEmbeddings
from pdf_rag_chatbot.erros import ErroConfiguracao
from pdf_rag_chatbot.indice_vetorial import IndiceVetorial
from pdf_rag_chatbot.modelos import Trecho
from pdf_rag_chatbot.recuperador import Recuperador, fontes_mencionadas

from dobles import EmbeddingsFalso

_VALIDOS = st.integers(min_value=1, max_value=20)

_INVALIDOS = st.one_of(
    st.integers(max_value=0),
    st.integers(min_value=21),
    st.floats(allow_nan=True, allow_infinity=True),
    st.booleans(),
    st.text(max_size=10),
    st.none(),
    st.lists(st.integers(), max_size=3),
).filter(lambda v: not (type(v) is int and 1 <= v <= 20))


def _criar_recuperador(diretorio: Path, top_k: int) -> Recuperador:
    provedor = EmbeddingsFalso()
    indice = IndiceVetorial(diretorio, provedor.nome_modelo)
    indice.carregar()
    return Recuperador(GeradorDeEmbeddings(provedor), indice, top_k)


@settings(max_examples=100, deadline=None)
@given(valor=_VALIDOS)
def test_top_k_valido_e_aceito(valor: int) -> None:
    # Feature: pdf-rag-chatbot, Property 13: Validação de Top_K aceita exatamente os inteiros em [1, 20]
    # **Validates: Requirements 2.6, 2.7**
    assert validar_top_k(valor) == valor
    with tempfile.TemporaryDirectory() as tmp:
        recuperador = _criar_recuperador(Path(tmp) / "indice", 4)
        recuperador.definir_top_k(valor)
        assert recuperador.top_k == valor


@settings(max_examples=100, deadline=None)
@given(anterior=_VALIDOS, valor=_INVALIDOS)
def test_top_k_invalido_e_rejeitado_e_mantem_valor_anterior(
    anterior: int, valor: object
) -> None:
    # Feature: pdf-rag-chatbot, Property 13: Validação de Top_K rejeita os demais valores e mantém o anterior
    # **Validates: Requirements 2.6, 2.7**
    with pytest.raises(ErroConfiguracao) as erro:
        validar_top_k(valor)
    assert "1" in str(erro.value) and "20" in str(erro.value)

    with tempfile.TemporaryDirectory() as tmp:
        recuperador = _criar_recuperador(Path(tmp) / "indice", anterior)
        with pytest.raises(ErroConfiguracao):
            recuperador.definir_top_k(valor)
        assert recuperador.top_k == anterior


@st.composite
def _cenario_comparativo(draw):
    """Índice com vários arquivos, pergunta que menciona 2 a 10 deles e Top_K."""
    quantidade = draw(st.integers(min_value=2, max_value=12))
    # nomes de largura fixa: nenhum é substring de outro
    fontes = [f"arquivo{n:02d}x.pdf" for n in range(quantidade)]
    trechos_por_fonte = {
        fonte: draw(st.integers(min_value=1, max_value=4)) for fonte in fontes
    }
    mencionados = draw(
        st.lists(
            st.sampled_from(fontes),
            min_size=2,
            max_size=min(10, quantidade),
            unique=True,
        )
    )
    com_extensao = draw(st.booleans())
    nomes = [m if com_extensao else m[: -len(".pdf")] for m in mencionados]
    pergunta = "Compare " + " e ".join(n.upper() for n in nomes) + " sobre o tema."
    top_k = draw(st.integers(min_value=1, max_value=20))
    return trechos_por_fonte, mencionados, pergunta, top_k


@settings(max_examples=100, deadline=None)
@given(cenario=_cenario_comparativo())
def test_pergunta_comparativa_recupera_trechos_de_cada_arquivo_mencionado(
    cenario,
) -> None:
    # Feature: pdf-rag-chatbot, Property 19: Perguntas comparativas recuperam Trechos de cada arquivo mencionado
    # **Validates: Requirements 3.5**
    trechos_por_fonte, mencionados, pergunta, top_k = cenario
    with tempfile.TemporaryDirectory() as tmp:
        provedor = EmbeddingsFalso()
        embeddings = GeradorDeEmbeddings(provedor)
        indice = IndiceVetorial(Path(tmp) / "indice", provedor.nome_modelo)
        indice.carregar()
        for fonte, quantidade in trechos_por_fonte.items():
            trechos = [
                Trecho(texto=f"{fonte} conteúdo {n}", fonte=fonte, pagina=n + 1)
                for n in range(quantidade)
            ]
            indice.substituir_fonte(
                fonte, trechos, embeddings.gerar([t.texto for t in trechos])
            )

        resultados = Recuperador(embeddings, indice, top_k).recuperar(pergunta)

    fontes_recuperadas = {r.trecho.fonte for r in resultados}
    assert set(mencionados) <= fontes_recuperadas
    assert len(resultados) <= 10


# ---------------------------------------------------------------------------
# Property 30: menção de arquivo exige termo isolado
# ---------------------------------------------------------------------------

_CARACTERES_DE_PALAVRA = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
_SEPARADORES = " \t\n.,;:!?()[]{}\"'-/\\@#%&*+="

_RADICAIS = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",
    min_size=1,
    max_size=8,
)
_EXTENSOES = st.sampled_from([".pdf", ".PDF", ".Pdf"])
_FONTES = st.builds(lambda r, e: r + e, _RADICAIS, _EXTENSOES)
_TEXTO_LIVRE = st.text(
    alphabet=_CARACTERES_DE_PALAVRA + _SEPARADORES, min_size=0, max_size=12
)
_LADO_ISOLADO = st.one_of(
    st.just(""), st.text(alphabet=_SEPARADORES, min_size=1, max_size=3)
)
_LADO_COLADO = st.text(alphabet=_CARACTERES_DE_PALAVRA, min_size=1, max_size=3)
_FORMA = st.sampled_from(["completo", "radical"])
_CAIXA = st.sampled_from(["original", "minuscula", "maiuscula", "alternada"])


def _radical(fonte: str) -> str:
    return fonte[: fonte.rindex(".")]


def _escolher_forma(fonte: str, forma: str) -> str:
    return fonte if forma == "completo" else _radical(fonte)


def _aplicar_caixa(texto: str, caixa: str) -> str:
    if caixa == "minuscula":
        return texto.lower()
    if caixa == "maiuscula":
        return texto.upper()
    if caixa == "alternada":
        return "".join(c.upper() if i % 2 else c.lower() for i, c in enumerate(texto))
    return texto


def _e_caractere_de_palavra(caractere: str) -> bool:
    return caractere == "_" or ("a" <= caractere.lower() <= "z") or caractere.isdigit()


def _oraculo_menciona(pergunta: str, nome: str) -> bool:
    """Varredura posição a posição, sem expressão regular."""
    texto = pergunta.lower()
    alvo = nome.lower()
    inicio = texto.find(alvo)
    while inicio != -1:
        fim = inicio + len(alvo)
        antes_ok = inicio == 0 or not _e_caractere_de_palavra(texto[inicio - 1])
        depois_ok = fim == len(texto) or not _e_caractere_de_palavra(texto[fim])
        if antes_ok and depois_ok:
            return True
        inicio = texto.find(alvo, inicio + 1)
    return False


def _oraculo_fontes(pergunta: str, fontes: list[str]) -> list[str]:
    resultado: list[str] = []
    for fonte in fontes:
        if fonte in resultado:
            continue
        if _oraculo_menciona(pergunta, fonte) or _oraculo_menciona(
            pergunta, _radical(fonte)
        ):
            resultado.append(fonte)
    return resultado


@settings(max_examples=200, deadline=None)
@given(
    fontes=st.lists(_FONTES, min_size=1, max_size=4),
    antes=_TEXTO_LIVRE,
    depois=_TEXTO_LIVRE,
    meio=st.lists(
        st.tuples(st.integers(min_value=0, max_value=3), _FORMA, _CAIXA, _TEXTO_LIVRE),
        max_size=3,
    ),
)
def test_mencao_de_arquivo_concorda_com_oraculo_independente(
    fontes: list[str], antes: str, depois: str, meio: list[tuple[int, str, str, str]]
) -> None:
    # Feature: pdf-rag-chatbot, Property 30: Menção de arquivo exige termo isolado
    # **Validates: Requirements 3.5**
    pergunta = antes
    for indice, forma, caixa, sobra in meio:
        fonte = fontes[indice % len(fontes)]
        pergunta += _aplicar_caixa(_escolher_forma(fonte, forma), caixa) + sobra
    pergunta += depois
    assert fontes_mencionadas(pergunta, fontes) == _oraculo_fontes(pergunta, fontes)


@settings(max_examples=200, deadline=None)
@given(
    fonte=_FONTES,
    forma=_FORMA,
    caixa=_CAIXA,
    antes=_LADO_ISOLADO,
    depois=_LADO_ISOLADO,
)
def test_nome_isolado_por_pontuacao_espaco_inicio_ou_fim_e_detectado(
    fonte: str, forma: str, caixa: str, antes: str, depois: str
) -> None:
    # Feature: pdf-rag-chatbot, Property 30: Nome cercado por pontuação, espaços, início ou fim é detectado
    # **Validates: Requirements 3.5**
    nome = _aplicar_caixa(_escolher_forma(fonte, forma), caixa)
    assert fontes_mencionadas(antes + nome + depois, [fonte]) == [fonte]


@settings(max_examples=200, deadline=None)
@given(
    fonte=_FONTES,
    caixa=_CAIXA,
    antes=st.one_of(st.just(""), _LADO_ISOLADO, _LADO_COLADO),
    depois=st.one_of(st.just(""), _LADO_ISOLADO, _LADO_COLADO),
)
def test_radical_colado_em_letra_digito_ou_sublinhado_nao_e_detectado(
    fonte: str, caixa: str, antes: str, depois: str
) -> None:
    # Feature: pdf-rag-chatbot, Property 30: Nome colado em letra, dígito ou sublinhado não é detectado
    # **Validates: Requirements 3.5**
    colado_antes = antes != "" and _e_caractere_de_palavra(antes[-1])
    colado_depois = depois != "" and _e_caractere_de_palavra(depois[0])
    nome = _aplicar_caixa(_radical(fonte), caixa)
    encontrado = fontes_mencionadas(antes + nome + depois, [fonte])
    if colado_antes or colado_depois:
        # o radical só tem caracteres de palavra, logo nenhuma ocorrência fica isolada
        assert encontrado == []
    else:
        assert encontrado == [fonte]


@pytest.mark.parametrize(
    ("pergunta", "fontes", "esperado"),
    [
        ("compare", ["c.pdf"], []),
        ("compare c.pdf e d.pdf", ["c.pdf", "d.pdf"], ["c.pdf", "d.pdf"]),
        ("o relatorio_a.pdf traz dados", ["a.pdf"], []),
        ("o relatorio_a.pdf traz dados", ["relatorio_a.pdf", "a.pdf"], ["relatorio_a.pdf"]),
        ("veja o A.PDF.", ["a.pdf"], ["a.pdf"]),
        ("veja (Relatorio)", ["relatorio.pdf"], ["relatorio.pdf"]),
        ("relatorio2 e relatorio_ e xrelatorio", ["relatorio.pdf"], []),
        ("relatorio", ["relatorio.pdf"], ["relatorio.pdf"]),
        ("relatorio.pdf2", ["relatorio.pdf"], ["relatorio.pdf"]),  # radical ainda isolado
        ("", ["a.pdf"], []),
    ],
)
def test_mencao_de_arquivo_exemplos_fixos(
    pergunta: str, fontes: list[str], esperado: list[str]
) -> None:
    # Feature: pdf-rag-chatbot, Property 30: Menção de arquivo exige termo isolado (exemplos fixos)
    # **Validates: Requirements 3.5**
    assert fontes_mencionadas(pergunta, fontes) == esperado
