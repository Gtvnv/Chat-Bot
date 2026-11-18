"""Tipos compartilhados entre os provedores e o gerador de respostas."""

from __future__ import annotations

import httpx

# Tempo limite aceito pelos provedores: segundos (número) ou um httpx.Timeout completo.
TempoLimite = float | httpx.Timeout
