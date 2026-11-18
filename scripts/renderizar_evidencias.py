"""Converte as transcrições de ``docs/evidencias/*.txt`` em imagens PNG.

As imagens imitam uma janela de terminal e são gravadas em ``docs/img/``.
O texto é exatamente o que a CLI imprimiu; só a quebra de linhas longas é
ajustada para caber na largura da imagem.

Uso: ``python scripts/renderizar_evidencias.py``
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RAIZ = Path(__file__).resolve().parent.parent
ORIGEM = RAIZ / "docs" / "evidencias"
DESTINO = RAIZ / "docs" / "img"

LARGURA_COLUNAS = 100
TAMANHO_FONTE = 17
MARGEM = 24
BARRA = 38
FUNDO = (24, 26, 33)
TEXTO = (220, 223, 228)
COMANDO = (126, 231, 135)
CITACAO = (121, 192, 255)
AVISO = (255, 203, 107)
CANDIDATAS = [
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/cour.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]


def _fonte() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for caminho in CANDIDATAS:
        if Path(caminho).is_file():
            return ImageFont.truetype(caminho, TAMANHO_FONTE)
    return ImageFont.load_default()


def _cor(linha: str, em_fontes: bool) -> tuple[int, int, int]:
    if linha.startswith(("$ ", "> ")):
        return COMANDO
    if em_fontes or linha.startswith("Fontes:"):
        return CITACAO
    if linha.startswith(("Aviso", "Atenção", "[código")):
        return AVISO
    return TEXTO


def _linhas_quebradas(texto: str) -> list[tuple[str, tuple[int, int, int]]]:
    resultado: list[tuple[str, tuple[int, int, int]]] = []
    em_fontes = False
    for bruta in texto.splitlines():
        if bruta.startswith("Fontes:"):
            em_fontes = True
        elif bruta.startswith(("> ", "$ ", "[código")):
            em_fontes = False
        cor = _cor(bruta, em_fontes)
        partes = textwrap.wrap(bruta, LARGURA_COLUNAS, subsequent_indent="  ") or [""]
        resultado.extend((p, cor) for p in partes)
    return resultado


def renderizar(origem: Path, destino: Path) -> None:
    fonte = _fonte()
    linhas = _linhas_quebradas(origem.read_text(encoding="utf-8"))
    caixa = fonte.getbbox("M" * LARGURA_COLUNAS)
    largura_texto = caixa[2] - caixa[0]
    altura_linha = int(TAMANHO_FONTE * 1.45)
    largura = largura_texto + 2 * MARGEM
    altura = BARRA + MARGEM * 2 + altura_linha * len(linhas)

    imagem = Image.new("RGB", (largura, altura), FUNDO)
    desenho = ImageDraw.Draw(imagem)
    desenho.rectangle([0, 0, largura, BARRA], fill=(44, 47, 56))
    for i, cor in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        x = 18 + i * 22
        desenho.ellipse([x, 12, x + 13, 25], fill=cor)
    desenho.text((90, 10), f"pdf_rag_chatbot (Docker) - {origem.stem}", fill=(170, 174, 182), font=fonte)

    y = BARRA + MARGEM
    for texto, cor in linhas:
        desenho.text((MARGEM, y), texto, fill=cor, font=fonte)
        y += altura_linha
    imagem.save(destino, optimize=True)


def main() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    for origem in sorted(ORIGEM.glob("*.txt")):
        destino = DESTINO / f"{origem.stem}.png"
        renderizar(origem, destino)
        print(f"Gerado: {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()