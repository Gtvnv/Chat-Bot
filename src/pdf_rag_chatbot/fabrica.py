"""Fábrica de componentes: monta provedores e objetos a partir da Configuracao."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from pdf_rag_chatbot.config import Configuracao
from pdf_rag_chatbot.embeddings.base import GeradorDeEmbeddings, ProvedorEmbeddings
from pdf_rag_chatbot.embeddings.ollama import OllamaEmbeddings
from pdf_rag_chatbot.embeddings.openai import OpenAIEmbeddings
from pdf_rag_chatbot.erros import ErroChaveApi
from pdf_rag_chatbot.geracao.base import GeradorDeRespostas, ProvedorGeracao
from pdf_rag_chatbot.geracao.ollama import OllamaGeracao
from pdf_rag_chatbot.geracao.openai import OpenAIGeracao
from pdf_rag_chatbot.indice_vetorial import IndiceVetorial
from pdf_rag_chatbot.ingestor import Ingestor, LeitorPdfPypdf
from pdf_rag_chatbot.recuperador import Recuperador

if TYPE_CHECKING:  # evita dependência circular e import antecipado
    from pdf_rag_chatbot.chatbot import Chatbot

NOME_VARIAVEL_CHAVE_API = "OPENAI_API_KEY"


@dataclass
class Componentes:
    """Conjunto de objetos da aplicação, todos compartilhando os mesmos embeddings."""

    configuracao: Configuracao
    cliente_http: httpx.Client
    embeddings: GeradorDeEmbeddings
    gerador: GeradorDeRespostas
    indice: IndiceVetorial
    ingestor: Ingestor
    recuperador: Recuperador
    chatbot: "Chatbot"

    def fechar(self) -> None:
        self.cliente_http.close()


def criar_cliente_http(config: Configuracao) -> httpx.Client:
    """Cria o cliente HTTP com os timeouts da configuração.

    Leitura/geração usa `timeout_geracao_s`; a conexão usa o timeout local.
    """
    return httpx.Client(
        timeout=httpx.Timeout(
            config.timeout_geracao_s, connect=config.timeout_conexao_local_s
        )
    )


def _timeout_local(config: Configuracao) -> httpx.Timeout:
    # Passado a cada requisição, pois um timeout numérico por chamada
    # substituiria também o tempo de conexão do cliente.
    return httpx.Timeout(config.timeout_geracao_s, connect=config.timeout_conexao_local_s)


def _chave_api(config: Configuracao) -> str:
    if not config.chave_api or not config.chave_api.strip():
        raise ErroChaveApi(NOME_VARIAVEL_CHAVE_API)
    return config.chave_api


def criar_provedor_embeddings(
    config: Configuracao, cliente: httpx.Client
) -> ProvedorEmbeddings:
    if config.provedor_embeddings == "local":
        return OllamaEmbeddings(
            cliente,
            config.ollama_host,
            config.modelo_embeddings_local,
            timeout_s=_timeout_local(config),  # type: ignore[arg-type]
        )
    return OpenAIEmbeddings(
        cliente,
        _chave_api(config),
        config.modelo_embeddings_externo,
        timeout_s=config.timeout_geracao_s,
    )


def criar_provedor_geracao(config: Configuracao, cliente: httpx.Client) -> ProvedorGeracao:
    if config.provedor_geracao == "local":
        return OllamaGeracao(
            cliente,
            config.ollama_host,
            config.modelo_geracao_local,
            timeout_s=_timeout_local(config),  # type: ignore[arg-type]
        )
    return OpenAIGeracao(
        cliente,
        _chave_api(config),
        config.modelo_geracao_externo,
        timeout_s=config.timeout_geracao_s,
    )


def criar_gerador_de_embeddings(
    config: Configuracao, cliente: httpx.Client | None = None
) -> GeradorDeEmbeddings:
    cliente = cliente or criar_cliente_http(config)
    return GeradorDeEmbeddings(criar_provedor_embeddings(config, cliente))


def criar_gerador_de_respostas(
    config: Configuracao, cliente: httpx.Client | None = None
) -> GeradorDeRespostas:
    cliente = cliente or criar_cliente_http(config)
    provedor = criar_provedor_geracao(config, cliente)
    # O provedor local recebe o Timeout completo (conexão + leitura); o
    # GeradorDeRespostas o repassa a cada chamada.
    timeout: float | httpx.Timeout = (
        _timeout_local(config) if config.provedor_geracao == "local" else config.timeout_geracao_s
    )
    return GeradorDeRespostas(provedor, timeout_s=timeout)  # type: ignore[arg-type]


def criar_componentes(
    config: Configuracao, cliente: httpx.Client | None = None
) -> Componentes:
    """Cria todos os componentes; Ingestor e Recuperador usam o mesmo GeradorDeEmbeddings."""
    from pdf_rag_chatbot.chatbot import Chatbot  # import tardio (tarefa 11.1)

    cliente = cliente or criar_cliente_http(config)
    embeddings = criar_gerador_de_embeddings(config, cliente)
    gerador = criar_gerador_de_respostas(config, cliente)
    indice = IndiceVetorial(config.diretorio_indice, embeddings.nome_modelo)
    ingestor = Ingestor(
        LeitorPdfPypdf(),
        embeddings,
        indice,
        config.tamanho_trecho,
        config.sobreposicao,
    )
    recuperador = Recuperador(embeddings, indice, config.top_k)
    chatbot = Chatbot(recuperador, gerador, indice)
    return Componentes(
        configuracao=config,
        cliente_http=cliente,
        embeddings=embeddings,
        gerador=gerador,
        indice=indice,
        ingestor=ingestor,
        recuperador=recuperador,
        chatbot=chatbot,
    )
