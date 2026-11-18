"""Testes de fumaça de entrega: .gitignore, .env.example, dependências, licença,
inputs/, README e varredura do termo proibido.

Requisitos: 6.1, 6.2, 7.1, 7.2, 7.3, 8.4, 8.9, 9.2, 9.3, 9.4.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pdf_rag_chatbot.ingestor import LeitorPdfPypdf

RAIZ = Path(__file__).resolve().parent.parent
README = RAIZ / "README.md"
INPUTS = RAIZ / "inputs"

# O termo proibido é montado em partes para este arquivo não conter o literal.
TERMO_PROIBIDO = re.compile(r"\b" + "D" + "I" + "O" + r"\b", re.IGNORECASE)

# Variáveis lidas por config.py (Requisito 6.2).
VARIAVEIS_ESPERADAS = (
    "PROVEDOR_EMBEDDINGS",
    "PROVEDOR_GERACAO",
    "OPENAI_API_KEY",
    "OLLAMA_HOST",
    "MODELO_EMBEDDINGS_LOCAL",
    "MODELO_GERACAO_LOCAL",
    "MODELO_EMBEDDINGS_EXTERNO",
    "MODELO_GERACAO_EXTERNO",
    "TAMANHO_TRECHO",
    "SOBREPOSICAO",
    "TOP_K",
    "DIRETORIO_INDICE",
)


# --------------------------------------------------------------------- utilidades
def _linhas_sem_comentario(caminho: Path) -> list[str]:
    linhas = caminho.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in linhas if ln.strip() and not ln.strip().startswith("#")]


def _secao(texto: str, titulo: str, nivel: int) -> str:
    """Corpo da seção cujo título é ``titulo`` (até o próximo título de nível <= nivel)."""
    marcador = "#" * nivel
    padrao = re.compile(
        rf"^{marcador} {re.escape(titulo)}\s*$(.*?)(?=^#{{1,{nivel}}} |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    achado = padrao.search(texto)
    assert achado, f"seção '{titulo}' não encontrada no README"
    return achado.group(1)


def _fora_de_blocos_de_codigo(texto: str) -> str:
    return re.sub(r"```.*?```", "", texto, flags=re.DOTALL)


def _frases(texto: str) -> list[str]:
    limpo = _fora_de_blocos_de_codigo(texto)
    return [f for f in re.split(r"(?<=[.!?])\s+", limpo.strip()) if len(f.strip()) > 3]


def _imagens_do_readme() -> list[tuple[str, str]]:
    texto = README.read_text(encoding="utf-8")
    return re.findall(r"!\[([^\]]*)\]\(([^)\s]+)\)", texto)


# ------------------------------------------------------------ 6.1 .gitignore
def test_gitignore_cobre_env_indice_e_ambientes_virtuais() -> None:
    entradas = _linhas_sem_comentario(RAIZ / ".gitignore")
    assert ".env" in entradas
    assert any(e.rstrip("/") == "data" for e in entradas), "índice persistido (data/)"
    assert ".venv/" in entradas or ".venv" in entradas
    assert "venv/" in entradas or "venv" in entradas


# ------------------------------------------------------------ 6.2 .env.example
def test_env_example_tem_todas_as_variaveis() -> None:
    valores: dict[str, str] = {}
    for linha in _linhas_sem_comentario(RAIZ / ".env.example"):
        nome, _, valor = linha.partition("=")
        valores[nome.strip()] = valor.strip()
    faltando = [v for v in VARIAVEIS_ESPERADAS if v not in valores]
    assert not faltando, f"variáveis ausentes no .env.example: {faltando}"


def test_env_example_nao_contem_chave_real() -> None:
    texto = (RAIZ / ".env.example").read_text(encoding="utf-8")
    valor = ""
    for linha in _linhas_sem_comentario(RAIZ / ".env.example"):
        nome, _, v = linha.partition("=")
        if nome.strip() == "OPENAI_API_KEY":
            valor = v.strip()
    assert valor in ("", "sua-chave-aqui")
    assert not re.search(r"sk-[A-Za-z0-9_-]{10,}", texto)


# ------------------------------------------------------------ 9.2 dependências
@pytest.mark.parametrize("arquivo", ["requirements.txt", "requirements-dev.txt"])
def test_dependencias_com_versao_fixada_ou_minima(arquivo: str) -> None:
    pacote = re.compile(r"^[A-Za-z0-9_.\-]+(\[[A-Za-z0-9_,\-]+\])?(==|>=)\d[\w.\-]*$")
    for linha in _linhas_sem_comentario(RAIZ / arquivo):
        if linha.startswith("-r "):
            assert (RAIZ / linha[3:].strip()).is_file()
            continue
        assert pacote.match(linha), f"{arquivo}: dependência sem '==' ou '>=': {linha}"


# ------------------------------------------------------------ 9.3 licença
def test_licenca_osi_com_ano_e_titular() -> None:
    texto = (RAIZ / "LICENSE").read_text(encoding="utf-8")
    assert re.match(r"\s*(MIT License|Apache License|BSD)", texto)
    direitos = re.search(r"Copyright \(c\) (\d{4}) (\S.+)", texto)
    assert direitos, "linha de copyright com ano e titular não encontrada"
    assert direitos.group(2).strip()
    assert "Permission is hereby granted, free of charge" in texto
    assert "THE SOFTWARE IS PROVIDED \"AS IS\"" in texto


# ------------------------------------------------------------ 7.1 a 7.3 inputs
def test_pasta_inputs_existe() -> None:
    assert INPUTS.is_dir()


def test_documento_de_sentencas_valido() -> None:
    caminho = INPUTS / "sentencas.txt"
    texto = caminho.read_bytes().decode("utf-8")  # falha se não for UTF-8
    sentencas = [ln for ln in texto.splitlines() if ln.strip()]
    assert 5 <= len(sentencas) <= 100, f"{len(sentencas)} sentenças"
    longas = [s for s in sentencas if len(s) > 500]
    assert not longas, f"{len(longas)} sentença(s) com mais de 500 caracteres"


def test_pdfs_de_exemplo_legiveis_e_com_texto() -> None:
    pdfs = sorted(p for p in INPUTS.iterdir() if p.suffix.lower() == ".pdf")
    assert 2 <= len(pdfs) <= 10, f"{len(pdfs)} PDFs"
    leitor = LeitorPdfPypdf()
    for pdf in pdfs:
        assert pdf.stat().st_size <= 10 * 1024 * 1024, f"{pdf.name} maior que 10 MB"
        paginas = leitor.ler_paginas(pdf)
        assert any(p.strip() for p in paginas), f"{pdf.name} sem texto extraível"


# ------------------------------------------------------------ 8 README
def test_readme_possui_secoes_obrigatorias() -> None:
    texto = README.read_text(encoding="utf-8")
    titulos = re.findall(r"^#{2,3} (.+?)\s*$", texto, flags=re.MULTILINE)
    for esperado in (
        "Descrição do projeto",
        "Instalação e execução",
        "Modo local gratuito (Ollama)",
        "Modo com API externa (OpenAI)",
        "Prints de tela",
        "Processo de construção",
        "Limitações conhecidas",
        "Possíveis evoluções",
        "Uso responsável",
    ):
        assert esperado in titulos, f"seção ausente no README: {esperado}"
    assert any(t.startswith("Insights") for t in titulos)


def test_readme_descricao_com_100_a_500_palavras() -> None:
    corpo = _secao(README.read_text(encoding="utf-8"), "Descrição do projeto", 2)
    palavras = len(corpo.split())
    assert 100 <= palavras <= 500, f"{palavras} palavras"


def test_readme_processo_com_quatro_etapas_em_ordem_e_duas_frases() -> None:
    texto = README.read_text(encoding="utf-8")
    processo = _secao(texto, "Processo de construção", 2)
    subtitulos = re.findall(r"^### (.+?)\s*$", processo, flags=re.MULTILINE)
    assert subtitulos == ["Ingestão", "Indexação", "Recuperação", "Geração"]
    for etapa in subtitulos:
        corpo = _secao(processo, etapa, 3)
        assert len(_frases(corpo)) >= 2, f"etapa '{etapa}' com menos de 2 frases"


def test_readme_insights_limitacoes_e_evolucoes() -> None:
    texto = README.read_text(encoding="utf-8")
    titulo = next(
        t for t in re.findall(r"^## (.+?)\s*$", texto, flags=re.MULTILINE)
        if t.startswith("Insights")
    )
    insights = re.findall(r"^\d+\. \*\*(.+?)\*\*\s*(\S.*)$", _secao(texto, titulo, 2), re.MULTILINE)
    assert len(insights) >= 3, f"{len(insights)} insights"
    for secao in ("Limitações conhecidas", "Possíveis evoluções"):
        itens = re.findall(r"^- \S", _secao(texto, secao, 2), flags=re.MULTILINE)
        assert len(itens) >= 3, f"{secao}: {len(itens)} itens"


def test_readme_tem_ao_menos_3_imagens_com_texto_alternativo_e_formato_valido() -> None:
    imagens = _imagens_do_readme()
    assert len(imagens) >= 3, f"{len(imagens)} imagens referenciadas"
    for alt, caminho in imagens:
        assert alt.strip(), f"texto alternativo vazio para {caminho}"
        assert caminho.lower().endswith((".png", ".jpg", ".jpeg")), caminho
        assert not re.match(r"^(https?:|/|[A-Za-z]:)", caminho), f"caminho não relativo: {caminho}"


def test_imagens_do_readme_existem() -> None:
    inexistentes = [c for _, c in _imagens_do_readme() if not (RAIZ / c).is_file()]
    assert not inexistentes, f"imagens referenciadas no README não existem: {inexistentes}"


def test_arquivos_referenciados_no_readme_existem() -> None:
    texto = _fora_de_blocos_de_codigo(README.read_text(encoding="utf-8"))
    referencias: set[str] = set()
    for alvo in re.findall(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)", texto):
        if not re.match(r"^(https?:|mailto:|#)", alvo):
            referencias.add(alvo)
    for token in re.findall(r"`([^`\s]+)`", texto):
        if re.match(r"^(inputs|scripts|docs|src|tests)/[\w./\-]+\.\w+$", token):
            referencias.add(token)
    # as imagens têm teste próprio
    imagens = {c for _, c in _imagens_do_readme()}
    inexistentes = sorted(r for r in referencias - imagens if not (RAIZ / r).exists())
    assert not inexistentes, f"arquivos referenciados no README não existem: {inexistentes}"


# ------------------------------------------------------------ 9.4 termo proibido
EXTENSOES = {".md", ".py", ".txt", ".toml", ".cfg", ".ini", ".example", ".json", ".yml", ".yaml", ".rst"}
NOMES = {".gitignore", "LICENSE"}
PASTAS_EXCLUIDAS = {".git", ".venv", "venv", ".hypothesis", ".pytest_cache", "__pycache__", ".kiro", ".idea", ".vscode"}


def test_varredura_do_termo_proibido() -> None:
    este = Path(__file__).resolve()
    ocorrencias: list[str] = []
    for arquivo in RAIZ.rglob("*"):
        relativo = arquivo.relative_to(RAIZ)
        if not arquivo.is_file() or arquivo.resolve() == este:
            continue
        if any(p in PASTAS_EXCLUIDAS or p.endswith(".egg-info") for p in relativo.parts):
            continue
        if arquivo.suffix.lower() not in EXTENSOES and arquivo.name not in NOMES and arquivo.name != ".env.example":
            continue
        for numero, linha in enumerate(arquivo.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if TERMO_PROIBIDO.search(linha):
                ocorrencias.append(f"{relativo}:{numero}")
    assert not ocorrencias, f"termo proibido encontrado em: {ocorrencias}"
