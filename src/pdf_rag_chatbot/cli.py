"""CLI: argumentos, sequência de inicialização e laço interativo."""

from __future__ import annotations

import argparse
import logging
import os
import shlex
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import httpx

from pdf_rag_chatbot import mensagens
from pdf_rag_chatbot.chatbot import Comando, reconhecer_comando
from pdf_rag_chatbot.citacoes import formatar_lista_citacoes
from pdf_rag_chatbot.config import Configuracao, carregar_configuracao
from pdf_rag_chatbot.erros import (
    ErroChatbot,
    ErroProvedorLocalIndisponivel,
)
from pdf_rag_chatbot.fabrica import Componentes, criar_cliente_http, criar_componentes
from pdf_rag_chatbot.log_seguro import instalar_filtro_segredos

logger = logging.getLogger(__name__)

Entrada = Callable[[str], str]
Saida = Callable[[str], None]

PROMPT = "> "


# ---------------------------------------------------------------------------
# Verificação do provedor local
# ---------------------------------------------------------------------------
def _modelo_instalado(esperado: str, instalados: Sequence[str]) -> bool:
    candidatos = {esperado}
    if ":" not in esperado:
        candidatos.add(f"{esperado}:latest")
    return any(nome in candidatos for nome in instalados)


def verificar_disponibilidade_local(config: Configuracao, cliente: httpx.Client) -> None:
    """Confere o Ollama (`GET /api/tags`) e os modelos dos componentes locais.

    Não há fallback para o provedor externo: qualquer falha lança
    `ErroProvedorLocalIndisponivel`. Sem componente local, não faz requisição.
    """
    componentes: list[tuple[str, str]] = []
    if config.provedor_embeddings == "local":
        componentes.append(("embeddings", config.modelo_embeddings_local))
    if config.provedor_geracao == "local":
        componentes.append(("geração", config.modelo_geracao_local))
    if not componentes:
        return

    url = f"{config.ollama_host.rstrip('/')}/api/tags"
    try:
        resposta = cliente.get(url, timeout=httpx.Timeout(config.timeout_conexao_local_s))
        resposta.raise_for_status()
        dados = resposta.json()
        instalados = [str(m.get("name", "")) for m in dados.get("models", [])]
    except (httpx.HTTPError, ValueError, AttributeError, TypeError):
        raise ErroProvedorLocalIndisponivel(
            " e ".join(c for c, _ in componentes),
            ", ".join(m for _, m in componentes),
            mensagens.ollama_inacessivel(config.ollama_host),
        ) from None

    for componente, modelo in componentes:
        if not _modelo_instalado(modelo, instalados):
            raise ErroProvedorLocalIndisponivel(
                componente, modelo, mensagens.modelo_local_ausente(modelo)
            )


# ---------------------------------------------------------------------------
# Comandos de sessão
# ---------------------------------------------------------------------------
def _dividir_caminhos(texto: str) -> list[str]:
    """Divide com `shlex.split`; no Windows preserva barras invertidas."""
    if os.name == "nt":
        partes = shlex.split(texto, posix=False)
        return [
            p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'" else p
            for p in partes
        ]
    return shlex.split(texto)


def _carregar(componentes: Componentes, argumentos: str, saida: Saida) -> None:
    try:
        caminhos = _dividir_caminhos(argumentos)
    except ValueError:
        saida(mensagens.ERRO_ASPAS)
        return
    if not caminhos:
        saida(mensagens.USO_CARREGAR)
        return
    _carregar_lista(componentes, caminhos, saida)


def _carregar_lista(componentes: Componentes, caminhos: Sequence[str], saida: Saida) -> None:
    resumo = componentes.ingestor.ingerir([Path(c) for c in caminhos])
    for linha in resumo.mensagens:
        saida(linha)
    saida(
        mensagens.resumo_ingestao(
            resumo.sucessos, resumo.falhas_ou_ignorados, resumo.total_trechos
        )
    )


def _documentos(componentes: Componentes, saida: Saida) -> None:
    fontes = sorted(componentes.indice.listar_fontes())
    if not fontes:
        saida(mensagens.NENHUM_DOCUMENTO_INDEXADO)
        return
    for fonte in fontes:
        saida(fonte)


def _top_k(componentes: Componentes, argumentos: str, saida: Saida) -> None:
    partes = argumentos.split()
    if len(partes) != 1:
        saida(mensagens.USO_TOPK)
        return
    try:
        valor: object = int(partes[0])
    except ValueError:
        valor = partes[0]  # a validação rejeita com a faixa aceita
    componentes.recuperador.definir_top_k(valor)
    saida(mensagens.top_k_atualizado(componentes.recuperador.top_k))


def _argumentos_do_comando(entrada: str) -> str:
    partes = entrada.strip().split(maxsplit=1)
    return partes[1] if len(partes) > 1 else ""


def _responder(componentes: Componentes, entrada: str, saida: Saida) -> None:
    resposta = componentes.chatbot.responder(entrada)
    saida(resposta.texto)
    if resposta.citacoes:
        saida(formatar_lista_citacoes(resposta.citacoes))


def _tratar_entrada(componentes: Componentes, entrada: str, saida: Saida) -> bool:
    """Processa uma entrada. Devolve True quando a sessão deve terminar."""
    comando = reconhecer_comando(entrada)
    if comando is Comando.SAIR:
        componentes.chatbot.limpar_historico()
        return True
    if comando is Comando.DOCUMENTOS:
        _documentos(componentes, saida)
    elif comando is Comando.CARREGAR:
        _carregar(componentes, _argumentos_do_comando(entrada), saida)
    elif comando is Comando.TOPK:
        _top_k(componentes, _argumentos_do_comando(entrada), saida)
    else:
        _responder(componentes, entrada, saida)
    return False


def laco_interativo(componentes: Componentes, entrada: Entrada, saida: Saida) -> int:
    """Laço da sessão. `ErroChatbot` (inclusive `ErroChaveApi`) não encerra a sessão."""
    saida(mensagens.BOAS_VINDAS)
    while True:
        try:
            texto = entrada(PROMPT)
        except (EOFError, KeyboardInterrupt):
            componentes.chatbot.limpar_historico()
            return 0
        try:
            if _tratar_entrada(componentes, texto, saida):
                return 0
        except ErroChatbot as erro:
            saida(str(erro))
        except Exception:  # noqa: BLE001 - qualquer falha inesperada vira erro genérico
            logger.error("Erro inesperado durante a sessão.", exc_info=True)
            saida(mensagens.ERRO_INESPERADO)


# ---------------------------------------------------------------------------
# Inicialização e ponto de entrada
# ---------------------------------------------------------------------------
def _criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf_rag_chatbot", description="Chatbot RAG sobre PDFs."
    )
    sub = parser.add_subparsers(dest="comando")
    sub.add_parser("chat", help="inicia a sessão interativa (padrão)")
    carregar = sub.add_parser("carregar", help="indexa arquivos ou diretórios e sai")
    carregar.add_argument("caminhos", nargs="+", help="arquivos .pdf/.txt ou diretórios")
    return parser


def iniciar(
    saida: Saida,
    env: Mapping[str, str] | None = None,
    arquivo_env: Path | None = Path(".env"),
    cliente_http: httpx.Client | None = None,
) -> tuple[Componentes, bool] | None:
    """Sequência de inicialização. Devolve (componentes, cliente_proprio) ou None em falha."""
    try:
        config = carregar_configuracao(env, arquivo_env)
    except ErroChatbot as erro:
        saida(str(erro))
        return None

    instalar_filtro_segredos([config.chave_api] if config.chave_api else [])

    if "externo" in (config.provedor_embeddings, config.provedor_geracao):
        saida(mensagens.aviso_terceiros("OpenAI"))

    proprio = cliente_http is None
    cliente = cliente_http or criar_cliente_http(config)
    try:
        verificar_disponibilidade_local(config, cliente)
        componentes = criar_componentes(config, cliente)
        componentes.indice.carregar()
    except ErroChatbot as erro:
        saida(str(erro))
        if proprio:
            cliente.close()
        return None
    except Exception:  # noqa: BLE001
        logger.error("Erro inesperado na inicialização.", exc_info=True)
        saida(mensagens.ERRO_INESPERADO)
        if proprio:
            cliente.close()
        return None
    return componentes, proprio


def executar(
    argv: Sequence[str] | None = None,
    entrada: Entrada = input,
    saida: Saida = print,
    env: Mapping[str, str] | None = None,
    arquivo_env: Path | None = Path(".env"),
    cliente_http: httpx.Client | None = None,
) -> int:
    """Executa a CLI e devolve o código de saída. Entrada e saída são injetáveis."""
    args = _criar_parser().parse_args(list(argv) if argv is not None else None)

    iniciado = iniciar(saida, env, arquivo_env, cliente_http)
    if iniciado is None:
        return 1
    componentes, proprio = iniciado
    try:
        if args.comando == "carregar":
            try:
                _carregar_lista(componentes, args.caminhos, saida)
            except ErroChatbot as erro:
                saida(str(erro))
                return 1
            except Exception:  # noqa: BLE001
                logger.error("Erro inesperado ao carregar.", exc_info=True)
                saida(mensagens.ERRO_INESPERADO)
                return 1
            return 0
        return laco_interativo(componentes, entrada, saida)
    finally:
        if proprio:
            componentes.fechar()


def main(argv: Sequence[str] | None = None) -> int:
    return executar(argv)
