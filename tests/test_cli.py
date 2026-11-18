"""Testes de exemplo da CLI: laço interativo, códigos de saída e `topk`."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from pdf_rag_chatbot import mensagens
from pdf_rag_chatbot.cli import executar

MODELO_EMBEDDINGS = "nomic-embed-text"
MODELO_GERACAO = "llama3.2"
_MARCADOR_TRECHO = re.compile(r"^\[\d+\] \(", re.MULTILINE)


class OllamaSimulado:
    """Simula /api/tags, /api/embed e /api/chat do Ollama com `httpx.MockTransport`."""

    def __init__(self, modelos: list[str] | None = None) -> None:
        self.modelos = [MODELO_EMBEDDINGS, MODELO_GERACAO] if modelos is None else modelos
        self.requisicoes_chat: list[dict] = []
        self.hosts: list[str] = []

    def tratar(self, requisicao: httpx.Request) -> httpx.Response:
        self.hosts.append(requisicao.url.host)
        caminho = requisicao.url.path
        if caminho == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": f"{m}:latest"} for m in self.modelos]}
            )
        corpo = json.loads(requisicao.content)
        if caminho == "/api/embed":
            vetores = [[1.0, float(len(t) % 5 + 1), 0.5] for t in corpo["input"]]
            return httpx.Response(200, json={"embeddings": vetores})
        if caminho == "/api/chat":
            self.requisicoes_chat.append(corpo)
            numero = len(self.requisicoes_chat)
            return httpx.Response(
                200, json={"message": {"role": "assistant", "content": f"Resposta {numero} [1]"}}
            )
        return httpx.Response(404)

    def cliente(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.tratar))


def _entrada(linhas: list[str]) -> Iterator[str]:
    return iter(linhas)


def _executar(
    tmp_path: Path, linhas: list[str], ollama: OllamaSimulado, **extra_env: str
) -> tuple[int, list[str]]:
    env = {
        "PROVEDOR_EMBEDDINGS": "local",
        "PROVEDOR_GERACAO": "local",
        "DIRETORIO_INDICE": str(tmp_path / "indice"),
        **extra_env,
    }
    entradas = _entrada(linhas)
    saidas: list[str] = []
    cliente = ollama.cliente()
    try:
        codigo = executar(
            [],
            entrada=lambda _prompt: next(entradas),
            saida=saidas.append,
            env=env,
            arquivo_env=None,
            cliente_http=cliente,
        )
    finally:
        cliente.close()
    return codigo, saidas


@pytest.fixture
def documento(tmp_path: Path) -> Path:
    caminho = tmp_path / "notas.txt"
    caminho.write_text(
        "\n".join(f"Sentença número {i} sobre testes de software." for i in range(1, 7)),
        encoding="utf-8",
    )
    return caminho


def test_sair_termina_com_codigo_zero(tmp_path: Path) -> None:
    codigo, saidas = _executar(tmp_path, ["sair"], OllamaSimulado())

    assert codigo == 0
    assert saidas[0] == mensagens.BOAS_VINDAS


def test_sair_aceita_caixa_e_espacos(tmp_path: Path) -> None:
    codigo, _ = _executar(tmp_path, ["  SAIR  "], OllamaSimulado())

    assert codigo == 0


def test_erro_de_configuracao_termina_com_codigo_um_sem_requisicao(tmp_path: Path) -> None:
    ollama = OllamaSimulado()
    entradas = _entrada(["sair"])
    saidas: list[str] = []
    cliente = ollama.cliente()

    codigo = executar(
        [],
        entrada=lambda _p: next(entradas),
        saida=saidas.append,
        env={"PROVEDOR_GERACAO": "local"},  # PROVEDOR_EMBEDDINGS ausente
        arquivo_env=None,
        cliente_http=cliente,
    )
    cliente.close()

    assert codigo == 1
    assert any("PROVEDOR_EMBEDDINGS" in s for s in saidas)
    assert ollama.hosts == []


def test_modelo_local_ausente_termina_com_codigo_um(tmp_path: Path) -> None:
    ollama = OllamaSimulado(modelos=[MODELO_EMBEDDINGS])  # falta o modelo de geração

    codigo, saidas = _executar(tmp_path, ["sair"], ollama)

    assert codigo == 1
    assert any(MODELO_GERACAO in s for s in saidas)


def test_ollama_inacessivel_termina_com_codigo_um(tmp_path: Path) -> None:
    def recusar(_requisicao: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("recusada")

    cliente = httpx.Client(transport=httpx.MockTransport(recusar))
    saidas: list[str] = []
    codigo = executar(
        [],
        entrada=lambda _p: "sair",
        saida=saidas.append,
        env={
            "PROVEDOR_EMBEDDINGS": "local",
            "PROVEDOR_GERACAO": "local",
            "DIRETORIO_INDICE": str(tmp_path / "indice"),
        },
        arquivo_env=None,
        cliente_http=cliente,
    )
    cliente.close()

    assert codigo == 1
    assert any("Ollama" in s for s in saidas)


def test_interacoes_seguem_ordem_cronologica(tmp_path: Path, documento: Path) -> None:
    ollama = OllamaSimulado()
    linhas = [f'carregar "{documento}"', "Primeira pergunta?", "Segunda pergunta?", "sair"]

    codigo, saidas = _executar(tmp_path, linhas, ollama)

    assert codigo == 0
    assert len(ollama.requisicoes_chat) == 2

    # Cada resposta exibida aparece depois da sua pergunta e antes da próxima.
    assert "Resposta 1 [1]" in saidas
    assert "Resposta 2 [1]" in saidas
    assert saidas.index("Resposta 1 [1]") < saidas.index("Resposta 2 [1]")

    # A primeira chamada não tem histórico; a segunda traz a 1ª interação antes da pergunta atual.
    primeira = [m["content"] for m in ollama.requisicoes_chat[0]["messages"]]
    segunda = [m["content"] for m in ollama.requisicoes_chat[1]["messages"]]
    assert "Primeira pergunta?" not in "".join(primeira[:-1])
    assert segunda[-3:-1] == ["Primeira pergunta?", "Resposta 1 [1]"]
    assert segunda[-1].endswith("Pergunta: Segunda pergunta?")


def test_citacoes_aparecem_apos_a_resposta(tmp_path: Path, documento: Path) -> None:
    codigo, saidas = _executar(
        tmp_path, [f'carregar "{documento}"', "Qual é o tema?", "sair"], OllamaSimulado()
    )

    assert codigo == 0
    posicao = saidas.index("Resposta 1 [1]")
    assert saidas[posicao + 1].startswith("Fontes:")
    assert "notas.txt" in saidas[posicao + 1]


def test_documentos_lista_arquivos_indexados(tmp_path: Path, documento: Path) -> None:
    _, saidas = _executar(
        tmp_path, ["documentos", f'carregar "{documento}"', "documentos", "sair"], OllamaSimulado()
    )

    assert mensagens.NENHUM_DOCUMENTO_INDEXADO in saidas
    assert saidas[-1] == "notas.txt"


def test_topk_invalido_mantem_valor_anterior(tmp_path: Path, documento: Path) -> None:
    ollama = OllamaSimulado()
    linhas = [
        f'carregar "{documento}"',
        "topk 3",
        "topk 0",
        "topk 21",
        "topk abc",
        "Qual é o tema?",
        "sair",
    ]

    codigo, saidas = _executar(tmp_path, linhas, ollama)

    assert codigo == 0
    assert mensagens.top_k_atualizado(3) in saidas
    # Os três valores inválidos não confirmam atualização e citam a faixa aceita.
    rejeicoes = [s for s in saidas if "TOP_K inválido" in s]
    assert len(rejeicoes) == 3
    assert all("1 e 20" in s for s in rejeicoes)
    # A recuperação seguinte ainda usa Top_K = 3.
    ultimo_prompt = ollama.requisicoes_chat[0]["messages"][-1]["content"]
    assert len(_MARCADOR_TRECHO.findall(ultimo_prompt)) == 3


def test_topk_valido_altera_quantidade_de_trechos(tmp_path: Path, documento: Path) -> None:
    ollama = OllamaSimulado()

    _executar(tmp_path, [f'carregar "{documento}"', "topk 2", "Pergunta?", "sair"], ollama)

    prompt = ollama.requisicoes_chat[0]["messages"][-1]["content"]
    assert len(_MARCADOR_TRECHO.findall(prompt)) == 2
