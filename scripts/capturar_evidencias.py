"""Executa sessões reais da CLI e grava as transcrições em ``docs/evidencias/``.

Pensado para rodar no contêiner (``docker compose run``), com o Ollama do
compose. Cada sessão chama ``cli.executar`` com entrada simulada e registra
o comando digitado e tudo o que a CLI imprimiu, na ordem.

Uso: ``python scripts/capturar_evidencias.py [diretorio_saida] [nome_da_sessao ...]``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from pdf_rag_chatbot.cli import PROMPT, executar

SESSOES: list[tuple[str, list[str]]] = [
    (
        "01-carregar-pdfs",
        ["carregar inputs inputs/sentencas.txt", "documentos", "sair"],
    ),
    (
        "02-pergunta-com-citacoes",
        [
            "O que o documento de testes diz sobre testes baseados em propriedades?",
            "sair",
        ],
    ),
    (
        "03-pergunta-comparativa",
        [
            "Compare introducao_rag.pdf e arquitetura_de_software.pdf: "
            "o que cada um diz sobre trocar o provedor de IA?",
            "sair",
        ],
    ),
    (
        "04-insights-sentencas",
        [
            "Segundo sentencas.txt, por que cobertura de código alta não garante qualidade?",
            "sair",
        ],
    ),
]


def executar_sessao(comandos: list[str]) -> tuple[int, list[str], float]:
    linhas = ["$ python -m pdf_rag_chatbot"]
    fila = iter(comandos)

    def entrada(prompt: str) -> str:
        comando = next(fila)
        linhas.append(f"{prompt or PROMPT}{comando}")
        return comando

    inicio = time.perf_counter()
    codigo = executar([], entrada=entrada, saida=linhas.append, arquivo_env=None)
    return codigo, linhas, time.perf_counter() - inicio


def main() -> int:
    saida = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/evidencias")
    saida.mkdir(parents=True, exist_ok=True)
    filtro = set(sys.argv[2:])
    falhou = False
    for nome, comandos in SESSOES:
        if filtro and nome not in filtro:
            continue
        codigo, linhas, duracao = executar_sessao(comandos)
        linhas.append(f"[código de saída: {codigo}]")
        (saida / f"{nome}.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")
        print(f"{nome}: código {codigo}, {duracao:.1f} s")
        falhou = falhou or codigo != 0
    return 1 if falhou else 0


if __name__ == "__main__":
    raise SystemExit(main())