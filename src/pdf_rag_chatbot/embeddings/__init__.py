"""Geração de embeddings: interface, gerador em lotes e provedores."""

from pdf_rag_chatbot.embeddings.base import GeradorDeEmbeddings, ProvedorEmbeddings

__all__ = ["GeradorDeEmbeddings", "ProvedorEmbeddings"]
