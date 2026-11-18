"""Testes de exemplo do Gerador_de_Respostas (prompt e limite de resposta)."""

from __future__ import annotations

from pdf_rag_chatbot.geracao.base import GeradorDeRespostas, limitar_resposta
from pdf_rag_chatbot.geracao.prompt import (
    JANELA_HISTORICO,
    MENSAGEM_SISTEMA,
    SENTINELA_SEM_RESPOSTA,
    montar_prompt,
)
from pdf_rag_chatbot.modelos import Interacao, ResultadoBusca, Trecho
from dobles import GeracaoFalsa


def _resultado(texto: str = "conteudo", fonte: str = "a.pdf", pagina: int = 1) -> ResultadoBusca:
    return ResultadoBusca(
        trecho=Trecho(texto=texto, fonte=fonte, pagina=pagina, tipo="pdf"),
        similaridade=0.9,
    )


def test_prompt_exige_portugues():
    mensagens = montar_prompt("Qual o tema?", [_resultado()], [])
    sistema = mensagens[0]
    assert sistema["role"] == "system"
    assert "português" in sistema["content"]


def test_prompt_exige_aspas_em_citacoes_literais_no_idioma_original():
    conteudo = montar_prompt("Pergunta", [_resultado()], [])[0]["content"]
    assert "aspas" in conteudo
    assert "idioma original" in conteudo
    assert "literalmente" in conteudo


def test_prompt_instrui_o_sentinela():
    conteudo = montar_prompt("Pergunta", [_resultado()], [])[0]["content"]
    assert SENTINELA_SEM_RESPOSTA == "SEM_RESPOSTA"
    assert SENTINELA_SEM_RESPOSTA in conteudo
    assert conteudo == MENSAGEM_SISTEMA


def test_prompt_numera_trechos_e_inclui_pergunta():
    trechos = [_resultado("alfa", "x.pdf", 3), _resultado("beta", "y.pdf", 7)]
    final = montar_prompt("Minha pergunta?", trechos, [])[-1]
    assert final["role"] == "user"
    assert "[1] (x.pdf, p. 3): alfa" in final["content"]
    assert "[2] (y.pdf, p. 7): beta" in final["content"]
    assert "Minha pergunta?" in final["content"]


def test_historico_entra_em_ordem_cronologica():
    historico = [Interacao(pergunta=f"p{i}", resposta=f"r{i}") for i in range(3)]
    mensagens = montar_prompt("nova", [_resultado()], historico)
    papeis = [m["role"] for m in mensagens]
    assert papeis == ["system", "user", "assistant", "user", "assistant", "user", "assistant", "user"]
    conteudos = [m["content"] for m in mensagens[1:-1]]
    assert conteudos == ["p0", "r0", "p1", "r1", "p2", "r2"]


def test_historico_limitado_as_cinco_interacoes_mais_recentes():
    historico = [Interacao(pergunta=f"p{i}", resposta=f"r{i}") for i in range(8)]
    mensagens = montar_prompt("nova", [_resultado()], historico)
    perguntas = [m["content"] for m in mensagens[1:-1] if m["role"] == "user"]
    assert JANELA_HISTORICO == 5
    assert perguntas == ["p3", "p4", "p5", "p6", "p7"]


def test_gerador_envia_prompt_ao_provedor_e_devolve_resposta():
    provedor = GeracaoFalsa(["Resposta [1]"])
    resposta = GeradorDeRespostas(provedor).gerar("Pergunta?", [_resultado()], [])
    assert resposta == "Resposta [1]"
    assert provedor.chamadas == 1
    assert provedor.prompts[0][0]["role"] == "system"


def test_limitar_resposta_mantem_texto_no_limite_e_corta_acima():
    exato = "a" * 2000
    assert limitar_resposta(exato) == exato

    texto = ("palavra " * 400).strip()  # 3199 caracteres
    cortado = limitar_resposta(texto)
    assert len(cortado) <= 2000
    assert cortado.split(" ")[-1] == "palavra"
