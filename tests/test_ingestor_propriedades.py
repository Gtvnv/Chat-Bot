"""Testes de propriedade para ``Ingestor``."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dobles import EmbeddingsFalso, LeitorPdfFalso
from pdf_rag_chatbot.config import TAMANHO_MAXIMO, TAMANHO_MINIMO
from pdf_rag_chatbot.embeddings.base import GeradorDeEmbeddings
from pdf_rag_chatbot.erros import ErroConfiguracao, ErroLimiteCaminhos
from pdf_rag_chatbot.indice_vetorial import IndiceVetorial
from pdf_rag_chatbot.ingestor import Ingestor


@st.composite
def _parametros_chunking_invalidos(draw):
    """Gera pares ``(tamanho, sobreposicao, parametro_invalido)`` fora das faixas."""
    if draw(st.booleans()):
        # Tamanho fora de [100, 10000]; a sobreposição pode ser qualquer inteiro.
        tamanho = draw(
            st.one_of(
                st.integers(min_value=-20_000, max_value=TAMANHO_MINIMO - 1),
                st.integers(min_value=TAMANHO_MAXIMO + 1, max_value=50_000),
            )
        )
        sobreposicao = draw(st.integers(min_value=-20_000, max_value=50_000))
        return tamanho, sobreposicao, "TAMANHO_TRECHO"
    # Tamanho válido; sobreposição negativa ou acima de 50% do tamanho.
    tamanho = draw(st.integers(min_value=TAMANHO_MINIMO, max_value=TAMANHO_MAXIMO))
    sobreposicao = draw(
        st.one_of(
            st.integers(min_value=-20_000, max_value=-1),
            st.integers(min_value=tamanho // 2 + 1, max_value=tamanho + 20_000),
        )
    )
    return tamanho, sobreposicao, "SOBREPOSICAO"


# Feature: pdf-rag-chatbot, Property 2: Parâmetros de chunking inválidos são rejeitados sem processar arquivos
# **Validates: Requirements 1.3**
@settings(max_examples=100, deadline=None)
@given(_parametros_chunking_invalidos())
def test_parametros_chunking_invalidos_sao_rejeitados_sem_processar_arquivos(dados):
    tamanho, sobreposicao, parametro = dados
    with tempfile.TemporaryDirectory() as tmp:
        diretorio = Path(tmp) / "index"
        provedor = EmbeddingsFalso()
        embeddings = GeradorDeEmbeddings(provedor)
        indice = IndiceVetorial(diretorio, provedor.nome_modelo)
        indice.carregar()
        leitor = LeitorPdfFalso(paginas_padrao=["Texto de exemplo."])

        with pytest.raises(ErroConfiguracao) as info:
            Ingestor(leitor, embeddings, indice, tamanho, sobreposicao)

        mensagem = str(info.value)
        assert parametro in mensagem
        if parametro == "TAMANHO_TRECHO":
            assert f"{TAMANHO_MINIMO} e {TAMANHO_MAXIMO}" in mensagem
        else:
            assert f"entre 0 e {tamanho // 2}" in mensagem

        # Nenhum arquivo foi aberto, nenhum embedding foi pedido e o índice não mudou.
        assert leitor.lidos == []
        assert provedor.chamadas == 0
        assert len(indice) == 0
        assert indice.esta_vazio()
        assert not diretorio.exists()


class _IndiceCapturador(IndiceVetorial):
    """Índice que guarda os Trechos recebidos de ``Ingestor``, na ordem de entrega."""

    def __init__(self, diretorio: Path, nome_modelo: str) -> None:
        super().__init__(diretorio, nome_modelo)
        self.recebidos: list = []

    def substituir_fonte(self, fonte, trechos, vetores):
        self.recebidos = list(trechos)
        super().substituir_fonte(fonte, trechos, vetores)


_texto_de_pagina = st.one_of(
    st.just(""),
    st.text(alphabet=" \n\t", max_size=5),  # página só com espaços em branco
    st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"), max_codepoint=0x024F
        ),
        min_size=1,
        max_size=400,
    ),
)

_nome_de_arquivo = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-", min_size=1, max_size=20
).map(lambda base: f"{base}.pdf")


# Feature: pdf-rag-chatbot, Property 3: Trechos carregam arquivo e página de origem em ordem
# **Validates: Requirements 1.1, 1.4, 2.2**
@settings(max_examples=100, deadline=None)
@given(
    paginas=st.lists(_texto_de_pagina, min_size=1, max_size=12).filter(
        lambda ps: any(p.strip() for p in ps)
    ),
    nome=_nome_de_arquivo,
    tamanho=st.integers(min_value=TAMANHO_MINIMO, max_value=500),
    fracao=st.floats(min_value=0.0, max_value=0.5),
)
def test_trechos_carregam_arquivo_e_pagina_de_origem_em_ordem(
    paginas, nome, tamanho, fracao
):
    sobreposicao = int(tamanho * fracao)
    with tempfile.TemporaryDirectory() as tmp:
        provedor = EmbeddingsFalso()
        embeddings = GeradorDeEmbeddings(provedor)
        indice = _IndiceCapturador(Path(tmp) / "index", provedor.nome_modelo)
        indice.carregar()
        leitor = LeitorPdfFalso(paginas_por_arquivo={nome: paginas})
        ingestor = Ingestor(leitor, embeddings, indice, tamanho, sobreposicao)

        resumo = ingestor.ingerir([Path(tmp) / nome])

        assert resumo.sucessos == 1
        trechos = indice.recebidos
        assert trechos, "ao menos um Trecho deve ser gerado"
        numeros = [t.pagina for t in trechos]
        for trecho in trechos:
            assert trecho.fonte == nome
            assert isinstance(trecho.pagina, int)
            assert 1 <= trecho.pagina <= len(paginas)
            # A página indicada contém texto extraível (páginas vazias não geram Trechos).
            assert paginas[trecho.pagina - 1].strip()
        assert numeros == sorted(numeros)


# Feature: pdf-rag-chatbot, Property 4: Mais de 50 caminhos rejeita a solicitação inteira
# **Validates: Requirements 1.12, 1.13, 1.14**
@settings(max_examples=100, deadline=None)
@given(
    quantidade=st.integers(min_value=Ingestor.LIMITE_CAMINHOS + 1, max_value=300),
    com_diretorio=st.booleans(),
)
def test_mais_de_50_caminhos_rejeita_a_solicitacao_inteira(quantidade, com_diretorio):
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        provedor = EmbeddingsFalso()
        embeddings = GeradorDeEmbeddings(provedor)
        indice = IndiceVetorial(base / "index", provedor.nome_modelo)
        indice.carregar()
        leitor = LeitorPdfFalso(paginas_padrao=["Texto de exemplo."])
        ingestor = Ingestor(leitor, embeddings, indice, 1000, 200)

        # Estado anterior: uma fonte já indexada.
        ingestor.ingerir([base / "anterior.pdf"])
        fontes_antes = indice.listar_fontes()
        total_antes = len(indice)
        chamadas_antes = provedor.chamadas
        lidos_antes = list(leitor.lidos)
        assert total_antes > 0

        caminhos = [base / f"doc_{i}.pdf" for i in range(quantidade)]
        if com_diretorio:
            caminhos[0] = base  # diretório existente entre os caminhos

        with pytest.raises(ErroLimiteCaminhos) as info:
            ingestor.ingerir(caminhos)

        erro = info.value
        assert erro.limite == 50
        assert erro.recebidos == quantidade
        mensagem = str(erro)
        assert "50" in mensagem
        assert str(quantidade) in mensagem

        # Nenhum arquivo aberto, nenhum embedding pedido e índice inalterado.
        assert leitor.lidos == lidos_antes
        assert provedor.chamadas == chamadas_antes
        assert indice.listar_fontes() == fontes_antes
        assert len(indice) == total_antes


_base_de_nome = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-", min_size=1, max_size=12
).map(lambda base: f"f_{base}")

_extensao_pdf = st.sampled_from([".pdf", ".PDF", ".Pdf", ".pDf", ".pdF", ".PdF"])
_extensao_outra = st.sampled_from(
    ["", ".txt", ".doc", ".pdfx", ".pd", ".xpdf", ".png", ".pdf.bak"]
)


# Feature: pdf-rag-chatbot, Property 5: Descoberta de PDFs em diretório
# **Validates: Requirements 1.5**
@settings(max_examples=100, deadline=None)
@given(
    pdfs=st.lists(
        st.tuples(_base_de_nome, _extensao_pdf),
        max_size=8,
        unique_by=lambda item: item[0],
    ),
    outros=st.lists(
        st.tuples(_base_de_nome, _extensao_outra),
        max_size=8,
        unique_by=lambda item: item[0],
    ),
    subdiretorios=st.lists(
        st.tuples(_base_de_nome, st.booleans(), st.lists(_extensao_pdf, max_size=3)),
        max_size=4,
        unique_by=lambda item: item[0],
    ),
)
def test_descoberta_de_pdfs_em_diretorio(pdfs, outros, subdiretorios):
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp) / "entrada"
        raiz.mkdir()

        # Garante nomes distintos (sem diferenciar maiúsculas) entre todos os itens.
        usados: set[str] = set()

        def _livre(base: str) -> bool:
            if base.lower() in usados:
                return False
            usados.add(base.lower())
            return True

        esperados: set[str] = set()
        for base, extensao in pdfs:
            if _livre(base):
                nome = f"{base}{extensao}"
                (raiz / nome).write_bytes(b"%PDF-falso")
                esperados.add(nome)
        for base, extensao in outros:
            if _livre(base):
                (raiz / f"{base}{extensao}").write_bytes(b"conteudo")
        for base, termina_em_pdf, internos in subdiretorios:
            if _livre(base):
                # Subdiretório (às vezes com nome terminado em .pdf) com PDFs dentro.
                sub = raiz / (f"{base}.pdf" if termina_em_pdf else base)
                sub.mkdir()
                for i, extensao in enumerate(internos):
                    (sub / f"interno_{i}{extensao}").write_bytes(b"%PDF-falso")

        provedor = EmbeddingsFalso()
        embeddings = GeradorDeEmbeddings(provedor)
        indice = IndiceVetorial(Path(tmp) / "index", provedor.nome_modelo)
        indice.carregar()
        leitor = LeitorPdfFalso(paginas_padrao=["Texto de exemplo."])
        ingestor = Ingestor(leitor, embeddings, indice, 1000, 200)

        resumo = ingestor.ingerir([raiz])

        # Exatamente os PDFs diretos foram abertos; nada de subdiretórios nem outros formatos.
        assert sorted(leitor.lidos) == sorted(esperados)
        assert set(indice.listar_fontes()) == esperados
        assert resumo.sucessos == len(esperados)


class _IndiceRegistrador(IndiceVetorial):
    """Índice que registra quantos Trechos cada ``substituir_fonte`` recebeu."""

    def __init__(self, diretorio: Path, nome_modelo: str) -> None:
        super().__init__(diretorio, nome_modelo)
        self.contagens: dict[str, int] = {}

    def substituir_fonte(self, fonte, trechos, vetores):
        super().substituir_fonte(fonte, trechos, vetores)
        self.contagens[fonte] = len(trechos)


_pagina_com_texto = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), max_codepoint=0x024F),
    min_size=1,
    max_size=300,
)
_categoria = st.sampled_from(["valido", "inexistente", "ilegivel", "sem_texto"])


# Feature: pdf-rag-chatbot, Property 6: Falhas isoladas e resumo consistente
# **Validates: Requirements 1.7, 1.8, 1.11**
@settings(max_examples=100, deadline=None)
@given(
    itens=st.lists(
        st.tuples(_categoria, st.lists(_pagina_com_texto, min_size=1, max_size=4)),
        min_size=1,
        max_size=10,
    ),
    tamanho=st.integers(min_value=TAMANHO_MINIMO, max_value=300),
)
def test_falhas_isoladas_e_resumo_consistente(itens, tamanho):
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        provedor = EmbeddingsFalso()
        embeddings = GeradorDeEmbeddings(provedor)
        indice = _IndiceRegistrador(base / "index", provedor.nome_modelo)
        indice.carregar()

        nomes = [f"doc_{i}.pdf" for i in range(len(itens))]
        por_arquivo: dict[str, list[str]] = {}
        invalidos: list[str] = []
        for nome, (categoria, paginas) in zip(nomes, itens):
            if categoria == "valido":
                por_arquivo[nome] = paginas
            elif categoria == "sem_texto":
                por_arquivo[nome] = ["", "  \n"]
            elif categoria == "ilegivel":
                invalidos.append(nome)
            # "inexistente": não registrado e sem páginas padrão -> ErroPdf

        # Estado anterior: todo arquivo não válido já tinha uma versão indexada.
        nao_validos = [n for n, (c, _) in zip(nomes, itens) if c != "valido"]
        anterior = Ingestor(
            LeitorPdfFalso(paginas_padrao=["Versão anterior."]),
            embeddings,
            indice,
            tamanho,
            0,
        )
        anterior.ingerir([base / n for n in nao_validos])
        antes = len(indice)
        assert antes == len(nao_validos)
        indice.contagens.clear()

        leitor = LeitorPdfFalso(paginas_por_arquivo=por_arquivo, arquivos_invalidos=invalidos)
        ingestor = Ingestor(leitor, embeddings, indice, tamanho, 0)

        resumo = ingestor.ingerir([base / n for n in nomes])

        validos = [n for n, (c, _) in zip(nomes, itens) if c == "valido"]
        falhos = [n for n, (c, _) in zip(nomes, itens) if c in ("inexistente", "ilegivel")]

        # Arquivos válidos foram indexados; os demais não alteraram o índice.
        assert resumo.sucessos == len(validos)
        assert set(indice.contagens) == set(validos)
        for nome in nao_validos:
            resultados = indice.buscar(
                provedor_vetor(provedor), len(indice) + 1, fontes=[nome]
            )
            assert [r.trecho.texto for r in resultados] == ["Versão anterior."]

        # Cada falha é reportada com nome e motivo (sem texto extraível é ignorado).
        assert {f.nome for f in resumo.falhas} == set(falhos)
        assert all(f.motivo.strip() for f in resumo.falhas)

        # Resumo consistente.
        assert resumo.sucessos + resumo.falhas_ou_ignorados == len(nomes)
        assert resumo.total_trechos == sum(indice.contagens.values())
        assert len(indice) == antes + resumo.total_trechos


def provedor_vetor(provedor: EmbeddingsFalso):
    import numpy as np

    return np.asarray(provedor.vetor_de("consulta"), dtype=np.float32)


def _trechos_da_fonte(indice: IndiceVetorial, provedor: EmbeddingsFalso, fonte: str):
    """Trechos e similaridades de uma fonte, em ordem canônica."""
    resultados = indice.buscar(
        provedor_vetor(provedor), len(indice) + 1, fontes=[fonte]
    )
    return sorted(
        (r.trecho.texto, r.trecho.pagina, round(float(r.similaridade), 5))
        for r in resultados
    )


_paginas_com_texto = st.lists(_pagina_com_texto, min_size=1, max_size=5)


# Feature: pdf-rag-chatbot, Property 7: Reindexar substitui os Trechos do arquivo
# **Validates: Requirements 1.9**
@settings(max_examples=100, deadline=None)
@given(
    conteudo_a=_paginas_com_texto,
    conteudo_b=_paginas_com_texto,
    conteudo_outro=_paginas_com_texto,
    tamanho=st.integers(min_value=TAMANHO_MINIMO, max_value=300),
)
def test_reindexar_substitui_os_trechos_do_arquivo(
    conteudo_a, conteudo_b, conteudo_outro, tamanho
):
    nome = "alvo.pdf"
    outro = "outro.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        provedor = EmbeddingsFalso()
        embeddings = GeradorDeEmbeddings(provedor)

        # Referência: índice que só conhece o conteúdo B (indexado uma vez).
        indice_ref = IndiceVetorial(base / "ref", provedor.nome_modelo)
        indice_ref.carregar()
        Ingestor(
            LeitorPdfFalso(paginas_por_arquivo={nome: conteudo_b}),
            embeddings,
            indice_ref,
            tamanho,
            0,
        ).ingerir([base / nome])
        esperado = _trechos_da_fonte(indice_ref, provedor, nome)

        # Índice testado: A (e outro arquivo), depois B, depois B de novo.
        indice = IndiceVetorial(base / "index", provedor.nome_modelo)
        indice.carregar()
        leitor = LeitorPdfFalso(
            paginas_por_arquivo={nome: conteudo_a, outro: conteudo_outro}
        )
        ingestor = Ingestor(leitor, embeddings, indice, tamanho, 0)
        ingestor.ingerir([base / nome, base / outro])
        do_outro = _trechos_da_fonte(indice, provedor, outro)

        leitor.paginas_por_arquivo[nome] = conteudo_b
        ingestor.ingerir([base / nome])

        # Só os Trechos de B permanecem para o arquivo; nenhum de A.
        assert indice.listar_fontes() == sorted([nome, outro])
        assert _trechos_da_fonte(indice, provedor, nome) == esperado
        # Outras fontes não são afetadas.
        assert _trechos_da_fonte(indice, provedor, outro) == do_outro

        # Idempotência: indexar o mesmo conteúdo de novo mantém o mesmo estado.
        total = len(indice)
        ingestor.ingerir([base / nome])
        assert len(indice) == total
        assert _trechos_da_fonte(indice, provedor, nome) == esperado
        assert _trechos_da_fonte(indice, provedor, outro) == do_outro


def _estado_em_disco(diretorio: Path) -> dict[str, bytes]:
    """Nome e conteúdo de todos os arquivos do diretório do índice."""
    return {p.name: p.read_bytes() for p in sorted(diretorio.iterdir())}


def _estado_em_memoria(indice: IndiceVetorial):
    """Cópia comparável dos Trechos e vetores guardados em memória."""
    return (
        [(t.texto, t.fonte, t.pagina, t.tipo) for t in indice._trechos],
        indice._vetores.copy(),
    )


_modo_de_falha = st.sampled_from(["extracao", "embedding", "gravacao_vetores", "troca_arquivos"])


# Feature: pdf-rag-chatbot, Property 8: Falha na substituição preserva o estado anterior
# **Validates: Requirements 1.10**
@settings(max_examples=100, deadline=None)
@given(
    conteudo_anterior=_paginas_com_texto,
    conteudo_novo=_paginas_com_texto,
    conteudo_outro=_paginas_com_texto,
    modo=_modo_de_falha,
    tamanho=st.integers(min_value=TAMANHO_MINIMO, max_value=300),
)
def test_falha_na_substituicao_preserva_o_estado_anterior(
    conteudo_anterior, conteudo_novo, conteudo_outro, modo, tamanho
):
    nome = "alvo.pdf"
    outro = "outro.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        diretorio = base / "index"
        provedor = EmbeddingsFalso()
        embeddings = GeradorDeEmbeddings(provedor)
        indice = IndiceVetorial(diretorio, provedor.nome_modelo)
        indice.carregar()
        leitor = LeitorPdfFalso(
            paginas_por_arquivo={nome: conteudo_anterior, outro: conteudo_outro}
        )
        ingestor = Ingestor(leitor, embeddings, indice, tamanho, 0)

        # Estado anterior: o arquivo (e outro) já indexados.
        resumo_inicial = ingestor.ingerir([base / nome, base / outro])
        assert resumo_inicial.sucessos == 2

        disco_antes = _estado_em_disco(diretorio)
        memoria_antes = _estado_em_memoria(indice)
        fontes_antes = indice.listar_fontes()
        alvo_antes = _trechos_da_fonte(indice, provedor, nome)

        # Nova versão do arquivo, com uma falha injetada em uma das etapas.
        leitor.paginas_por_arquivo[nome] = conteudo_novo
        contexto = None
        if modo == "extracao":
            leitor.arquivos_invalidos.add(nome)
        elif modo == "embedding":
            provedor.falhar = True
        elif modo == "gravacao_vetores":
            contexto = patch(
                "pdf_rag_chatbot.indice_vetorial.np.save",
                side_effect=OSError("falha injetada ao gravar vetores"),
            )
        else:  # troca_arquivos: falha na primeira troca atômica, antes de qualquer troca
            contexto = patch(
                "pdf_rag_chatbot.indice_vetorial.os.replace",
                side_effect=OSError("falha injetada na troca de arquivos"),
            )

        if contexto is None:
            resumo = ingestor.ingerir([base / nome])
        else:
            with contexto:
                resumo = ingestor.ingerir([base / nome])

        # A falha é reportada e o arquivo não conta como sucesso.
        assert resumo.sucessos == 0
        assert {f.nome for f in resumo.falhas} == {nome}

        # Memória idêntica ao estado anterior.
        trechos_depois, vetores_depois = _estado_em_memoria(indice)
        assert trechos_depois == memoria_antes[0]
        assert np.array_equal(vetores_depois, memoria_antes[1])
        assert indice.listar_fontes() == fontes_antes
        assert _trechos_da_fonte(indice, provedor, nome) == alvo_antes

        # Disco idêntico (sem arquivos temporários residuais) e recarga equivalente.
        assert _estado_em_disco(diretorio) == disco_antes
        recarregado = IndiceVetorial(diretorio, provedor.nome_modelo)
        recarregado.carregar()
        trechos_recarregados, vetores_recarregados = _estado_em_memoria(recarregado)
        assert trechos_recarregados == memoria_antes[0]
        assert np.array_equal(vetores_recarregados, memoria_antes[1])


# Linhas sem quebras de linha (Cc, Zl, Zp) nem substitutos, para que a numeração
# esperada coincida com a de ``str.splitlines``.
_linha_de_sentenca = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc", "Zl", "Zp")),
    max_size=60,
)
_separador_de_linha = st.sampled_from(["\n", "\r\n", "\r"])


# Feature: pdf-rag-chatbot, Property 27: Documento de sentenças gera um item por linha não vazia
# **Validates: Requirements 7.4, 7.5**
@settings(max_examples=100, deadline=None)
@given(
    linhas=st.lists(_linha_de_sentenca, min_size=1, max_size=30).filter(
        lambda ls: any(linha.strip() for linha in ls)
    ),
    separador=_separador_de_linha,
    termina_com_quebra=st.booleans(),
    base_nome=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-", min_size=1, max_size=15
    ),
)
def test_documento_de_sentencas_gera_um_item_por_linha_nao_vazia(
    linhas, separador, termina_com_quebra, base_nome
):
    nome = f"{base_nome}.txt"
    conteudo = separador.join(linhas) + (separador if termina_com_quebra else "")
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        provedor = EmbeddingsFalso()
        embeddings = GeradorDeEmbeddings(provedor)
        indice = _IndiceCapturador(base / "index", provedor.nome_modelo)
        indice.carregar()
        leitor = LeitorPdfFalso(paginas_padrao=["Texto de exemplo."])
        (base / nome).write_bytes(conteudo.encode("utf-8"))

        resumo = Ingestor(leitor, embeddings, indice, 1000, 200).ingerir([base / nome])

        esperados = [
            (numero, linha.strip())
            for numero, linha in enumerate(linhas, start=1)
            if linha.strip()
        ]
        assert resumo.sucessos == 1
        assert resumo.total_trechos == len(esperados)
        assert [(t.pagina, t.texto) for t in indice.recebidos] == esperados
        assert all(t.fonte == nome and t.tipo == "texto" for t in indice.recebidos)
        assert len(indice) == len(esperados)
        assert indice.listar_fontes() == [nome]
        assert leitor.lidos == []  # .txt não passa pelo leitor de PDF


_conteudo_invalido_de_sentencas = st.one_of(
    # Não UTF-8: o byte 0xFF nunca é válido em UTF-8.
    st.tuples(st.binary(max_size=40), st.binary(max_size=40)).map(
        lambda partes: partes[0] + b"\xff" + partes[1]
    ),
    # Sem linhas não vazias: vazio ou só espaços em branco e quebras de linha.
    st.text(alphabet=" \t\r\n", max_size=40).map(lambda t: t.encode("utf-8")),
)


# Feature: pdf-rag-chatbot, Property 27: Documento de sentenças gera um item por linha não vazia
# **Validates: Requirements 7.4, 7.5**
@settings(max_examples=100, deadline=None)
@given(
    conteudo=_conteudo_invalido_de_sentencas,
    base_nome=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-", min_size=1, max_size=15
    ),
)
def test_documento_de_sentencas_invalido_nao_indexa_e_reporta_nome_e_motivo(
    conteudo, base_nome
):
    nome = f"{base_nome}.txt"
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        provedor = EmbeddingsFalso()
        embeddings = GeradorDeEmbeddings(provedor)
        indice = _IndiceCapturador(base / "index", provedor.nome_modelo)
        indice.carregar()
        leitor = LeitorPdfFalso(paginas_padrao=["Texto de exemplo."])
        (base / nome).write_bytes(conteudo)

        resumo = Ingestor(leitor, embeddings, indice, 1000, 200).ingerir([base / nome])

        # Nada é indexado e nenhum embedding é pedido.
        assert resumo.sucessos == 0
        assert resumo.total_trechos == 0
        assert indice.recebidos == []
        assert indice.esta_vazio()
        assert provedor.chamadas == 0

        # A falha identifica o arquivo e traz um motivo.
        assert [f.nome for f in resumo.falhas] == [nome]
        assert resumo.falhas[0].motivo.strip()
        assert any(nome in m for m in resumo.mensagens)
