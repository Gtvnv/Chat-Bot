# Imagem da aplicação PDF RAG Chatbot (CLI).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY inputs/ ./inputs/
RUN pip install --no-deps .

# Usuário sem privilégios para executar a aplicação.
RUN useradd --create-home chatbot && mkdir -p /app/data /app/docs && chown -R chatbot /app
USER chatbot

ENTRYPOINT ["python", "-m", "pdf_rag_chatbot"]
CMD ["chat"]