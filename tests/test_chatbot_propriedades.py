"""Testes de propriedade para ``Chatbot``."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pdf_rag_chatbot import mensagens
from pdf_rag_chatbot.chatbot import Chatbot
from pdf_rag_chatbot.embeddings import GeradorDeEmbeddings
from pdf_rag_chatbot.erros import ErroEmbeddings, ErroPergunta, ErroProvedor
from pdf_rag_chatbot.geracao.base import GeradorDeRespostas
from pdf_rag_chatbot.indice_vetorial import IndiceVetorial
from pdf_rag_chatbot.modelos import Trecho
from pdf_rag_chatbot.recuperador import Recuperador

from dobles import EmbeddingsFalso, GeracaoFalsa


class _RecuperadorEspiao(Recuperador):
    """Recuperador real que conta as chamadas a ``recuperar``."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.chamadas = 0

    def recuperar(self, pergunta: str):
        self.chamadas += 1
        return super().recuperar(pergunta)


def _criar_chatbot(diretorio: Path):
    """Chatbot com índice não vazio, para que só a validação possa barrar a pergunta."""
    provedor = EmbeddingsFalso()
    embeddings = GeradorDeEmbeddings(provedor)
    indice = IndiceVetorial(diretorio, provedor.nome_modelo)
    indice.carregar()
    texto = "conteúdo de teste"
    vetores = np.array([provedor.vetor_de(texto)], dtype=np.float32)
    indice.substituir_fonte("doc.pdf", [Trecho(texto, "doc.pdf", 1)], vetores)
    recuperador = _RecuperadorEspiao(embeddings, indice, 4)
    geracao = GeracaoFalsa(["Resposta fundamentada [1]."])
    chatbot = Chatbot(recuperador, GeradorDeRespostas(geracao), indice)
    return chatbot, recuperador, geracao


_ESPACOS = st.text(alphabet=" \t\n\r\u00a0\u2003", max_size=6)
_CARACTERE = st.characters(whitelist_categories=("L", "N"))


@st.composite
def _entradas(draw) -> str:
    """Entradas vazias, só espaços, dentro do limite e acima dele, com bordas."""
    tamanho = draw(
        st.one_of(
            st.integers(min_value=0, max_value=30),
            st.sampled_from([1999, 2000, 2001]),
            st.integers(min_value=1990, max_value=2100),
        )
    )
    nucleo = draw(_CARACTERE) * tamanho
    return draw(_ESPACOS) + nucleo + draw(_ESPACOS)


@settings(max_examples=100, deadline=None)
@given(entrada=st.one_of(_entradas(), st.text(max_size=40)))
def test_validacao_de_pergunta_sem_efeitos_colaterais(entrada: str) -> None:
    # Feature: pdf-rag-chatbot, Property 14: Validação de pergunta sem efeitos colaterais
    # **Validates: Requirements 2.4, 2.11, 2.12, 2.13, 4.5**
    with tempfile.TemporaryDirectory() as tmp:
        chatbot, recuperador, geracao = _criar_chatbot(Path(tmp) / "indice")
        tamanho = len(entrada.strip())

        if 1 <= tamanho <= mensagens.LIMITE_PERGUNTA:
            resposta = chatbot.responder(entrada)
            assert recuperador.chamadas == 1
            assert len(chatbot.historico) == 1
            assert chatbot.historico[0].pergunta == entrada.strip()
            assert resposta.texto
        else:
            with pytest.raises(ErroPergunta) as erro:
                chatbot.responder(entrada)
            if tamanho == 0:
                assert str(erro.value) == mensagens.PERGUNTA_VAZIA
            else:
                assert str(erro.value) == mensagens.PERGUNTA_LONGA
                assert str(mensagens.LIMITE_PERGUNTA) in str(erro.value)
            assert recuperador.chamadas == 0
            assert geracao.chamadas == 0
            assert chatbot.historico == ()


@settings(max_examples=100, deadline=None)
@given(
    nucleo=st.text(alphabet=_CARACTERE, min_size=1, max_size=mensagens.LIMITE_PERGUNTA),
    antes=_ESPACOS,
    depois=_ESPACOS,
)
def test_indice_vazio_nao_aciona_recuperacao_nem_modelo(
    nucleo: str, antes: str, depois: str
) -> None:
    # Feature: pdf-rag-chatbot, Property 15: Índice vazio não aciona recuperação nem modelo
    # **Validates: Requirements 2.8**
    with tempfile.TemporaryDirectory() as tmp:
        provedor = EmbeddingsFalso()
        embeddings = GeradorDeEmbeddings(provedor)
        indice = IndiceVetorial(Path(tmp) / "indice", provedor.nome_modelo)
        indice.carregar()
        assert indice.esta_vazio()
        recuperador = _RecuperadorEspiao(embeddings, indice, 4)
        geracao = GeracaoFalsa(["Resposta fundamentada [1]."])
        chatbot = Chatbot(recuperador, GeradorDeRespostas(geracao), indice)

        resposta = chatbot.responder(antes + nucleo + depois)

        assert resposta.texto == mensagens.INDICE_VAZIO
        assert "Nenhum PDF foi carregado" in resposta.texto
        assert "carregar" in resposta.texto
        assert recuperador.chamadas == 0
        assert geracao.chamadas == 0


class _RecuperadorVazio(_RecuperadorEspiao):
    """Recuperador que não encontra nenhum Trecho."""

    def recuperar(self, pergunta: str):
        self.chamadas += 1
        return []


_PALAVRAS = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs")), min_size=1, max_size=80
).filter(lambda t: t.strip() != "" and t.strip() != "SEM_RESPOSTA")
_MARCADORES_INVALIDOS = st.lists(
    st.sampled_from(["[0]", "[2]", "[3]", "[99]", "[1000]"]), max_size=3
)


@st.composite
def _respostas_sem_fundamento(draw):
    """Devolve (texto do modelo, mensagem esperada, usar_recuperador_vazio)."""
    caso = draw(st.sampled_from(["sem_marcador", "sentinela", "sem_trechos"]))
    if caso == "sentinela":
        espacos = st.text(alphabet=" \t\n", max_size=3)
        texto = draw(espacos) + "SEM_RESPOSTA" + draw(espacos)
        return texto, mensagens.NAO_ENCONTRADA, False
    if caso == "sem_trechos":
        return draw(_PALAVRAS) + " [1]", mensagens.NAO_ENCONTRADA, True
    texto = draw(_PALAVRAS) + " " + " ".join(draw(_MARCADORES_INVALIDOS))
    return texto, mensagens.NAO_FUNDAMENTADA, False


@settings(max_examples=100, deadline=None)
@given(caso=_respostas_sem_fundamento(), pergunta=_PALAVRAS)
def test_resposta_sem_fundamento_nao_e_exibida(caso, pergunta: str) -> None:
    # Feature: pdf-rag-chatbot, Property 18: Resposta sem fundamento não é exibida
    # **Validates: Requirements 3.4, 3.7**
    texto_modelo, esperada, sem_trechos = caso
    with tempfile.TemporaryDirectory() as tmp:
        chatbot, recuperador, geracao = _criar_chatbot(Path(tmp) / "indice")
        geracao.respostas = [texto_modelo]
        if sem_trechos:
            recuperador.__class__ = _RecuperadorVazio

        resposta = chatbot.responder(pergunta)

        assert resposta.texto == esperada
        assert resposta.citacoes == ()
        # Textos curtos podem ser substring da mensagem padrão por acaso;
        # a igualdade exata acima já garante que o texto do modelo não é exibido.
        if texto_modelo.strip() not in esperada:
            assert texto_modelo.strip() not in resposta.texto
        if sem_trechos:
            assert geracao.chamadas == 0

        # A sessão continua ativa: uma nova pergunta é processada normalmente.
        assert len(chatbot.historico) == 1
        segunda = chatbot.responder(pergunta)
        assert segunda.texto == esperada
        assert len(chatbot.historico) == 2


@settings(max_examples=100, deadline=None)
@given(
    sufixos=st.lists(
        st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=20),
        min_size=0,
        max_size=12,
    )
)
def test_janela_de_historico(sufixos: list[str]) -> None:
    # Feature: pdf-rag-chatbot, Property 21: Janela de histórico
    # **Validates: Requirements 4.2, 4.3**
    perguntas = [f"pergunta {i} {s}" for i, s in enumerate(sufixos)]
    with tempfile.TemporaryDirectory() as tmp:
        chatbot, _, geracao = _criar_chatbot(Path(tmp) / "indice")

        for n, pergunta in enumerate(perguntas):
            chatbot.responder(pergunta)

            # Contexto enviado ao gerador: as min(5, n) interações anteriores, em ordem.
            esperadas = perguntas[max(0, n - Chatbot.JANELA_HISTORICO) : n]
            enviado = geracao.prompts[-1]
            usuario = [m["content"] for m in enviado if m["role"] == "user"]
            assistente = [m for m in enviado if m["role"] == "assistant"]
            assert usuario[:-1] == esperadas
            assert len(assistente) == len(esperadas)
            assert pergunta in usuario[-1]

        # Histórico da sessão: todas as n interações em ordem cronológica.
        assert [i.pergunta for i in chatbot.historico] == perguntas


class _GeracaoQueFalhaNaChamada(GeracaoFalsa):
    """Geração que falha somente na chamada de número ``chamada_falha`` (a partir de 1)."""

    def __init__(self, respostas, chamada_falha: int, erro: Exception | None, timeout: bool):
        super().__init__(respostas)
        self.chamada_falha = chamada_falha
        self._erro_injetado = erro
        self._timeout_injetado = timeout
        self.ativa = False

    def completar(self, mensagens, timeout_s):
        if self.ativa and len(self.prompts) + 1 == self.chamada_falha:
            self.prompts.append(list(mensagens))
            self.timeouts.append(timeout_s)
            if self._timeout_injetado:
                raise ErroProvedor(f"tempo limite de {timeout_s:g} s excedido (simulado)")
            raise self._erro_injetado
        return super().completar(mensagens, timeout_s)


_ERROS_GERACAO = st.sampled_from(
    [
        ErroProvedor("erro HTTP 500"),
        ErroProvedor("resposta inválida do provedor"),
        ErroProvedor("falha de conexão"),
    ]
)


@settings(max_examples=100, deadline=None)
@given(
    ponto=st.sampled_from(["embeddings", "geracao", "reforco"]),
    erro=_ERROS_GERACAO,
    timeout=st.booleans(),
    pergunta=_PALAVRAS,
    interacoes_previas=st.integers(min_value=0, max_value=3),
)
def test_falha_do_provedor_nao_deixa_rastros(
    ponto: str, erro: Exception, timeout: bool, pergunta: str, interacoes_previas: int
) -> None:
    # Feature: pdf-rag-chatbot, Property 20: Falha do provedor não deixa rastros
    # **Validates: Requirements 3.6, 3.8, 3.9, 4.8, 2.9**
    with tempfile.TemporaryDirectory() as tmp:
        provedor = EmbeddingsFalso()
        embeddings = GeradorDeEmbeddings(provedor)
        indice = IndiceVetorial(Path(tmp) / "indice", provedor.nome_modelo)
        indice.carregar()
        for nome in ("a.pdf", "b.pdf"):
            texto = f"conteúdo de {nome}"
            vetores = np.array([provedor.vetor_de(texto)], dtype=np.float32)
            indice.substituir_fonte(nome, [Trecho(texto, nome, 1)], vetores)
        recuperador = _RecuperadorEspiao(embeddings, indice, 4)

        # No ponto "reforco", a pergunta cita dois arquivos e a 1ª resposta cita só um,
        # o que aciona a segunda chamada ao gerador.
        pergunta_final = f"{pergunta} a.pdf b.pdf" if ponto == "reforco" else pergunta
        chamada_falha = {"embeddings": 0, "geracao": 1, "reforco": 2}[ponto]
        geracao = _GeracaoQueFalhaNaChamada(
            ["Resposta fundamentada [1]."], chamada_falha, erro, timeout
        )
        chatbot = Chatbot(recuperador, GeradorDeRespostas(geracao), indice)

        for n in range(interacoes_previas):
            chatbot.responder(f"pergunta prévia {n}")
        geracao.chamada_falha = chamada_falha + len(geracao.prompts)
        geracao.ativa = True
        historico_antes = chatbot.historico
        chamadas_recuperacao = recuperador.chamadas

        if ponto == "embeddings":
            provedor.falhar = True
            esperado = ErroEmbeddings
        else:
            esperado = ErroProvedor

        with pytest.raises(esperado) as capturado:
            chatbot.responder(pergunta_final)

        # Mensagem de erro presente, sem resposta parcial nem Citações no erro.
        assert str(capturado.value)
        assert "Resposta fundamentada" not in str(capturado.value)
        assert chatbot.historico == historico_antes
        assert recuperador.chamadas == chamadas_recuperacao + 1

        # Nova chamada com a mesma pergunta refaz a recuperação do zero.
        provedor.falhar = False
        geracao.ativa = False
        resposta = chatbot.responder(pergunta_final)
        assert recuperador.chamadas == chamadas_recuperacao + 2
        assert resposta.texto
        assert len(chatbot.historico) == len(historico_antes) + 1
        assert chatbot.historico[:-1] == historico_antes
        assert chatbot.historico[-1].pergunta == pergunta_final.strip()
