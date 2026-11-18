"""Testes de segurança e de provedores da CLI (tarefa 14.5).

Requisitos: 5.3, 5.7, 5.8, 6.4, 6.8.
Usam `executar` com entrada e saída injetadas e `httpx.MockTransport`.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from pdf_rag_chatbot.cli import executar

CHAVE = "sk-chave-secreta-de-teste-12345"
HOST_LOCAL = "localhost"
MODELO_EMB = "nomic-embed-text"
MODELO_GER = "llama3.2"


class Transporte:
    """MockTransport que registra hosts e caminhos de todas as requisições."""

    def __init__(self, tratador) -> None:
        self.requisicoes: list[httpx.Request] = []
        self._tratador = tratador
        self.transporte = httpx.MockTransport(self._tratar)

    def _tratar(self, requisicao: httpx.Request) -> httpx.Response:
        self.requisicoes.append(requisicao)
        return self._tratador(requisicao)

    @property
    def hosts(self) -> set[str]:
        return {r.url.host for r in self.requisicoes}

    def cliente(self) -> httpx.Client:
        return httpx.Client(transport=self.transporte)


def _vetores(requisicao: httpx.Request) -> list[list[float]]:
    entradas = json.loads(requisicao.content)["input"]
    return [[1.0, 0.5, 0.25] for _ in entradas]


def _tags(*modelos: str) -> httpx.Response:
    return httpx.Response(200, json={"models": [{"name": m} for m in modelos]})


def _executar(
    entradas: list[str], transporte: Transporte, env: dict[str, str]
) -> tuple[int, list[str]]:
    saidas: list[str] = []
    fila = iter(entradas)
    cliente = transporte.cliente()
    try:
        codigo = executar(
            [],
            entrada=lambda _p: next(fila),
            saida=saidas.append,
            env=env,
            arquivo_env=None,
            cliente_http=cliente,
        )
    finally:
        cliente.close()
    return codigo, saidas


def _documento(tmp_path: Path) -> Path:
    caminho = tmp_path / "doc.txt"
    caminho.write_text("Testes de software reduzem defeitos.\n", encoding="utf-8")
    return caminho


# --------------------------------------------------------------------------
# Requisitos 5.3 e 6.4: erro 401 após a inicialização
# --------------------------------------------------------------------------
def test_erro_401_mantem_sessao_e_cita_variavel_sem_valor(tmp_path: Path) -> None:
    def tratar(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [
                {"index": i, "embedding": v} for i, v in enumerate(_vetores(req))
            ]})
        return httpx.Response(401, json={"error": {"message": f"chave {CHAVE} inválida"}})

    transporte = Transporte(tratar)
    env = {
        "PROVEDOR_EMBEDDINGS": "externo",
        "PROVEDOR_GERACAO": "externo",
        "OPENAI_API_KEY": CHAVE,
        "DIRETORIO_INDICE": str(tmp_path / "indice"),
    }
    doc = _documento(tmp_path)
    codigo, saidas = _executar(
        [f'carregar "{doc}"', "O que reduz defeitos?", "documentos", "sair"],
        transporte,
        env,
    )

    texto = "\n".join(saidas)
    assert "OPENAI_API_KEY" in texto
    assert CHAVE not in texto
    # A sessão continua: o comando seguinte ao erro 401 foi atendido e a saída é 0.
    assert "doc.txt" in saidas
    assert codigo == 0


# --------------------------------------------------------------------------
# Requisito 6.8: provedor local não acessa hosts externos
# --------------------------------------------------------------------------
def test_provedor_local_nao_acessa_hosts_externos(tmp_path: Path) -> None:
    def tratar(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/tags":
            return _tags(f"{MODELO_EMB}:latest", f"{MODELO_GER}:latest")
        if req.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": _vetores(req)})
        if req.url.path == "/api/chat":
            return httpx.Response(
                200, json={"message": {"role": "assistant", "content": "Reduzem defeitos [1]."}}
            )
        return httpx.Response(404)

    transporte = Transporte(tratar)
    env = {
        "PROVEDOR_EMBEDDINGS": "local",
        "PROVEDOR_GERACAO": "local",
        "DIRETORIO_INDICE": str(tmp_path / "indice"),
    }
    doc = _documento(tmp_path)
    codigo, saidas = _executar(
        [f'carregar "{doc}"', "O que reduz defeitos?", "sair"], transporte, env
    )

    caminhos = {r.url.path for r in transporte.requisicoes}
    assert {"/api/tags", "/api/embed", "/api/chat"} <= caminhos
    assert transporte.hosts == {HOST_LOCAL}
    assert codigo == 0
    assert not any("terceiros" in s for s in saidas)


# --------------------------------------------------------------------------
# Requisitos 5.7 e 5.8: falha de componente local
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("provedor_emb", "provedor_ger", "modelos_instalados", "componente", "modelo"),
    [
        ("local", "local", [f"{MODELO_GER}:latest"], "embeddings", MODELO_EMB),
        ("local", "local", [f"{MODELO_EMB}:latest"], "geração", MODELO_GER),
        ("local", "externo", [], "embeddings", MODELO_EMB),
        ("externo", "local", [f"{MODELO_EMB}:latest"], "geração", MODELO_GER),
    ],
)
def test_falha_de_componente_local_encerra_sem_recorrer_ao_externo(
    tmp_path: Path,
    provedor_emb: str,
    provedor_ger: str,
    modelos_instalados: list[str],
    componente: str,
    modelo: str,
) -> None:
    transporte = Transporte(lambda req: _tags(*modelos_instalados))
    env = {
        "PROVEDOR_EMBEDDINGS": provedor_emb,
        "PROVEDOR_GERACAO": provedor_ger,
        "OPENAI_API_KEY": CHAVE,
        "DIRETORIO_INDICE": str(tmp_path / "indice"),
    }
    codigo, saidas = _executar(["sair"], transporte, env)

    texto = "\n".join(saidas)
    assert codigo != 0
    assert componente in texto
    assert modelo in texto
    assert CHAVE not in texto
    assert "api.openai.com" not in transporte.hosts
    assert transporte.hosts <= {HOST_LOCAL}


def test_ollama_inacessivel_encerra_sem_recorrer_ao_externo(tmp_path: Path) -> None:
    def tratar(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("conexão recusada", request=req)

    transporte = Transporte(tratar)
    env = {
        "PROVEDOR_EMBEDDINGS": "local",
        "PROVEDOR_GERACAO": "local",
        "OPENAI_API_KEY": CHAVE,
        "DIRETORIO_INDICE": str(tmp_path / "indice"),
    }
    codigo, saidas = _executar(["sair"], transporte, env)

    texto = "\n".join(saidas)
    assert codigo != 0
    assert "embeddings" in texto and "geração" in texto
    assert MODELO_EMB in texto and MODELO_GER in texto
    assert transporte.hosts == {HOST_LOCAL}
