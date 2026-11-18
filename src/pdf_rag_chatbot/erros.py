"""Hierarquia de exceções do PDF RAG Chatbot."""

from __future__ import annotations


class ErroChatbot(Exception):
    """Erro base de todas as falhas previstas da aplicação."""


class ErroConfiguracao(ErroChatbot):
    """Provedor inválido, tamanho, sobreposição ou top_k fora da faixa."""


class ErroChaveApi(ErroConfiguracao):
    """Chave de API ausente, vazia ou rejeitada.

    Guarda apenas o nome da variável de ambiente, nunca o valor da chave.
    """

    def __init__(self, nome_variavel: str = "OPENAI_API_KEY") -> None:
        self.nome_variavel = nome_variavel
        super().__init__(
            f"Chave de API ausente, vazia ou inválida. "
            f"Verifique a variável {nome_variavel}."
        )


class ErroLimiteCaminhos(ErroChatbot):
    """Mais caminhos do que o limite permitido em uma única solicitação."""

    def __init__(self, recebidos: int, limite: int = 50) -> None:
        self.recebidos = recebidos
        self.limite = limite
        super().__init__(
            f"Limite de {limite} caminhos por solicitação excedido: "
            f"{recebidos} caminhos recebidos."
        )


class ErroPdf(ErroChatbot):
    """Arquivo inexistente ou ilegível."""


class ErroEmbeddings(ErroChatbot):
    """Falha ao gerar embeddings."""


class ErroProvedor(ErroChatbot):
    """Falha, resposta inválida ou timeout na geração de respostas."""


class ErroIndiceIncompativel(ErroChatbot):
    """O índice foi criado com outro modelo de embeddings."""


class ErroIndiceCorrompido(ErroChatbot):
    """Arquivos do índice ilegíveis ou inconsistentes."""


class ErroPergunta(ErroChatbot):
    """Pergunta vazia ou acima de 2.000 caracteres."""


class ErroProvedorLocalIndisponivel(ErroChatbot):
    """Componente local indisponível; informa componente, modelo e instruções."""

    def __init__(self, componente: str, modelo: str, instrucoes: str) -> None:
        self.componente = componente
        self.modelo = modelo
        self.instrucoes = instrucoes
        super().__init__(
            f"Provedor local indisponível para {componente} "
            f"(modelo esperado: {modelo}). {instrucoes}"
        )
