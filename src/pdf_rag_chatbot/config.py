"""Configuração do PDF RAG Chatbot: leitura do ambiente e validação de parâmetros."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

from dotenv import dotenv_values

from pdf_rag_chatbot.erros import ErroChaveApi, ErroConfiguracao

NOME_VARIAVEL_CHAVE_API = "OPENAI_API_KEY"
VALORES_PROVEDOR = ("local", "externo")

TAMANHO_MINIMO = 100
TAMANHO_MAXIMO = 10_000
TOP_K_MINIMO = 1
TOP_K_MAXIMO = 20

TAMANHO_PADRAO = 1000
SOBREPOSICAO_PADRAO = 200
TOP_K_PADRAO = 4

Provedor = Literal["local", "externo"]


@dataclass(frozen=True)
class Configuracao:
    """Configuração imutável da aplicação. A chave de API nunca aparece no repr."""

    provedor_embeddings: Provedor
    provedor_geracao: Provedor
    tamanho_trecho: int = TAMANHO_PADRAO
    sobreposicao: int = SOBREPOSICAO_PADRAO
    top_k: int = TOP_K_PADRAO
    diretorio_indice: Path = Path("data/index")
    ollama_host: str = "http://localhost:11434"
    modelo_embeddings_local: str = "nomic-embed-text"
    modelo_geracao_local: str = "llama3.2"
    modelo_embeddings_externo: str = "text-embedding-3-small"
    modelo_geracao_externo: str = "gpt-4o-mini"
    chave_api: str | None = field(default=None, repr=False)
    timeout_geracao_s: float = 60.0
    timeout_conexao_local_s: float = 10.0


def validar_parametros_chunking(tamanho: object, sobreposicao: object) -> None:
    """Valida tamanho (100 a 10.000) e sobreposição (0 a 50% do tamanho).

    Levanta ErroConfiguracao citando o parâmetro inválido e a faixa aceita.
    """
    if not _eh_inteiro(tamanho) or not TAMANHO_MINIMO <= tamanho <= TAMANHO_MAXIMO:
        raise ErroConfiguracao(
            f"Parâmetro TAMANHO_TRECHO inválido: deve ser um inteiro entre "
            f"{TAMANHO_MINIMO} e {TAMANHO_MAXIMO} caracteres."
        )
    if not _eh_inteiro(sobreposicao) or not 0 <= sobreposicao <= tamanho // 2:
        raise ErroConfiguracao(
            f"Parâmetro SOBREPOSICAO inválido: deve ser um inteiro entre 0 e "
            f"{tamanho // 2} caracteres (50% do tamanho do trecho)."
        )


def validar_top_k(valor: object) -> int:
    """Valida Top_K (inteiro entre 1 e 20) e devolve o valor como int."""
    if not _eh_inteiro(valor) or not TOP_K_MINIMO <= valor <= TOP_K_MAXIMO:
        raise ErroConfiguracao(
            f"Parâmetro TOP_K inválido: deve ser um inteiro entre "
            f"{TOP_K_MINIMO} e {TOP_K_MAXIMO}."
        )
    return valor


def carregar_configuracao(
    env: Mapping[str, str] | None = None,
    arquivo_env: Path | None = Path(".env"),
) -> Configuracao:
    """Carrega a configuração do `.env` mesclado com o ambiente.

    Quando o mesmo nome existe nas duas fontes, o ambiente prevalece.
    Se `env` for None, usa `os.environ`.
    """
    valores: dict[str, str] = {}
    if arquivo_env is not None and Path(arquivo_env).is_file():
        valores.update(
            {k: v for k, v in dotenv_values(arquivo_env).items() if v is not None}
        )
    valores.update(os.environ if env is None else env)

    provedor_embeddings = _ler_provedor(valores, "PROVEDOR_EMBEDDINGS")
    provedor_geracao = _ler_provedor(valores, "PROVEDOR_GERACAO")

    tamanho = _ler_inteiro(valores, "TAMANHO_TRECHO", TAMANHO_PADRAO)
    sobreposicao = _ler_inteiro(valores, "SOBREPOSICAO", SOBREPOSICAO_PADRAO)
    validar_parametros_chunking(tamanho, sobreposicao)
    top_k = validar_top_k(_ler_inteiro(valores, "TOP_K", TOP_K_PADRAO))

    chave_api: str | None = None
    if "externo" in (provedor_embeddings, provedor_geracao):
        chave_api = (valores.get(NOME_VARIAVEL_CHAVE_API) or "").strip()
        if not chave_api:
            raise ErroChaveApi(NOME_VARIAVEL_CHAVE_API)

    padrao = Configuracao(provedor_embeddings, provedor_geracao)
    return Configuracao(
        provedor_embeddings=provedor_embeddings,
        provedor_geracao=provedor_geracao,
        tamanho_trecho=tamanho,
        sobreposicao=sobreposicao,
        top_k=top_k,
        diretorio_indice=Path(
            _ler_texto(valores, "DIRETORIO_INDICE", str(padrao.diretorio_indice))
        ),
        ollama_host=_ler_texto(valores, "OLLAMA_HOST", padrao.ollama_host),
        modelo_embeddings_local=_ler_texto(
            valores, "MODELO_EMBEDDINGS_LOCAL", padrao.modelo_embeddings_local
        ),
        modelo_geracao_local=_ler_texto(
            valores, "MODELO_GERACAO_LOCAL", padrao.modelo_geracao_local
        ),
        modelo_embeddings_externo=_ler_texto(
            valores, "MODELO_EMBEDDINGS_EXTERNO", padrao.modelo_embeddings_externo
        ),
        modelo_geracao_externo=_ler_texto(
            valores, "MODELO_GERACAO_EXTERNO", padrao.modelo_geracao_externo
        ),
        chave_api=chave_api,
    )


def _eh_inteiro(valor: object) -> bool:
    return isinstance(valor, int) and not isinstance(valor, bool)


def _ler_provedor(valores: Mapping[str, str], nome: str) -> Provedor:
    bruto = (valores.get(nome) or "").strip()
    if bruto not in VALORES_PROVEDOR:
        aceitos = ", ".join(VALORES_PROVEDOR)
        raise ErroConfiguracao(
            f"Variável {nome} ausente ou inválida. Valores aceitos: {aceitos}."
        )
    return bruto  # type: ignore[return-value]


def _ler_texto(valores: Mapping[str, str], nome: str, padrao: str) -> str:
    bruto = (valores.get(nome) or "").strip()
    return bruto or padrao


def _ler_inteiro(valores: Mapping[str, str], nome: str, padrao: int) -> int | str:
    """Lê um inteiro; valor não numérico devolve o texto bruto para a validação rejeitar."""
    bruto = (valores.get(nome) or "").strip()
    if not bruto:
        return padrao
    try:
        return int(bruto)
    except ValueError:
        return bruto
