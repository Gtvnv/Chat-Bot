"""Mensagens ao usuário, centralizadas em português."""

from __future__ import annotations

LIMITE_PERGUNTA = 2000

PERGUNTA_VAZIA = "A pergunta não pode ser vazia. Digite uma nova pergunta."

PERGUNTA_LONGA = (
    f"A pergunta excede o limite de {LIMITE_PERGUNTA} caracteres. "
    "Reduza o texto e tente novamente."
)

INDICE_VAZIO = (
    "Nenhum PDF foi carregado. Use o comando "
    "'carregar <caminhos>' para indexar arquivos ou diretórios antes de perguntar."
)

NAO_ENCONTRADA = "A resposta não foi encontrada nos documentos carregados."

NAO_FUNDAMENTADA = (
    "A resposta não pôde ser fundamentada nos documentos carregados."
)

FALHA_COMUNICACAO = (
    "Falha de comunicação com o provedor. Nenhuma resposta foi gerada; "
    "tente novamente."
)

FALHA_BUSCA = "Não foi possível realizar a busca. Tente novamente."

NENHUM_DOCUMENTO_INDEXADO = "Nenhum documento está indexado."


def aviso_terceiros(provedor: str = "OpenAI") -> str:
    """Aviso exibido antes de qualquer requisição a um provedor externo."""
    return (
        f"Atenção: trechos dos documentos e perguntas serão enviados a um "
        f"serviço de terceiros ({provedor})."
    )


AVISO_TERCEIROS = aviso_terceiros("OpenAI")


def aviso_arquivos_sem_citacao(arquivos: list[str]) -> str:
    """Aviso de pergunta comparativa com arquivos mencionados sem Citação."""
    lista = ", ".join(arquivos)
    return f"Aviso: não foi possível citar trechos dos seguintes arquivos: {lista}."


REFORCO_COMPARATIVA = (
    "Cite ao menos um Trecho de cada um dos seguintes arquivos, "
    "usando os marcadores [n]: {arquivos}."
)


def nenhum_pdf_encontrado(diretorio: str) -> str:
    return f"Nenhum PDF encontrado no diretório '{diretorio}'."


def sem_texto_extraivel(arquivo: str) -> str:
    return f"O arquivo '{arquivo}' não tem texto extraível e foi ignorado."


def formato_nao_suportado(arquivo: str) -> str:
    return f"Formato não suportado (use .pdf ou .txt): '{arquivo}'."


def arquivo_inexistente(caminho: str) -> str:
    return f"Caminho inexistente: '{caminho}'."


def falha_arquivo(arquivo: str, motivo: str) -> str:
    return f"Falha ao processar '{arquivo}': {motivo}"


def arquivo_indexado(arquivo: str, trechos: int) -> str:
    return f"'{arquivo}' indexado com {trechos} trecho(s)."


# --- CLI -------------------------------------------------------------------

BOAS_VINDAS = (
    "PDF RAG Chatbot. Comandos: carregar <caminhos>, documentos, topk <n>, sair. "
    "Qualquer outra entrada é tratada como pergunta."
)

ERRO_INESPERADO = (
    "Ocorreu um erro inesperado. Os detalhes foram registrados no log; tente novamente."
)

USO_CARREGAR = "Uso: carregar <caminho> [<caminho> ...] (use aspas para caminhos com espaços)."

ERRO_ASPAS = "Não foi possível interpretar os caminhos: aspas não fechadas."

USO_TOPK = "Uso: topk <n>, com n inteiro entre 1 e 20."

TITULO_DOCUMENTOS = "Documentos indexados:"


def top_k_atualizado(valor: int) -> str:
    return f"Top_K atualizado para {valor}."


def resumo_ingestao(sucessos: int, falhas_ou_ignorados: int, total_trechos: int) -> str:
    return (
        f"Resumo da ingestão: {sucessos} arquivo(s) indexado(s), "
        f"{falhas_ou_ignorados} falha(s) ou ignorado(s), "
        f"{total_trechos} trecho(s) gerado(s)."
    )


def ollama_inacessivel(host: str) -> str:
    return f"Não foi possível acessar o Ollama em {host}. Inicie-o com 'ollama serve'."


def modelo_local_ausente(modelo: str) -> str:
    return f"O modelo '{modelo}' não está instalado. Instale-o com 'ollama pull {modelo}'."
