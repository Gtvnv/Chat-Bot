"""Log seguro: oculta segredos (como chaves de API) das entradas de log.

Requisito 6.3: o valor de qualquer chave de API não pode aparecer nos logs.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Collection

MASCARA = "***"


class FiltroSegredos(logging.Filter):
    """Substitui qualquer segredo por ``***`` na mensagem, nos argumentos e na exceção."""

    def __init__(self, segredos: Collection[str]) -> None:
        super().__init__()
        # Ignora vazios/apenas espaços (substituir "" corromperia a mensagem).
        # Ordem decrescente de tamanho evita que um segredo contido em outro
        # deixe sobras do segredo maior.
        validos = {s for s in segredos if isinstance(s, str) and s.strip()}
        self._segredos = sorted(validos, key=len, reverse=True)

    def ocultar(self, texto: str) -> str:
        """Devolve o texto com todas as ocorrências dos segredos mascaradas."""
        for segredo in self._segredos:
            if segredo in texto:
                texto = texto.replace(segredo, MASCARA)
        return texto

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._segredos:
            return True

        # Mensagem formatada (já com os argumentos aplicados). Depois de mascarar,
        # os argumentos são descartados para não serem formatados de novo.
        try:
            mensagem = record.getMessage()
        except Exception:  # noqa: BLE001 - argumentos inválidos não devem quebrar o log
            mensagem = f"{record.msg} {record.args}"
        record.msg = self.ocultar(mensagem)
        record.args = None

        # Texto da exceção: gera aqui para poder mascarar antes da formatação.
        if record.exc_info and not record.exc_text:
            record.exc_text = "".join(traceback.format_exception(*record.exc_info)).rstrip("\n")
        if record.exc_text:
            record.exc_text = self.ocultar(record.exc_text)

        if record.stack_info:
            record.stack_info = self.ocultar(record.stack_info)
        return True


def _todos_os_handlers() -> list[logging.Handler]:
    loggers = [logging.getLogger()]
    loggers += [
        logger
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    ]
    vistos: dict[int, logging.Handler] = {}
    for logger in loggers:
        for handler in logger.handlers:
            vistos[id(handler)] = handler
    return list(vistos.values())


def instalar_filtro_segredos(segredos: Collection[str]) -> FiltroSegredos:
    """Instala um ``FiltroSegredos`` em todos os handlers de log existentes.

    Percorre o logger raiz e todos os loggers já criados. Handlers que já
    possuem um ``FiltroSegredos`` o recebem atualizado, sem duplicar filtros.
    Devolve o filtro instalado.
    """
    filtro = FiltroSegredos(segredos)
    for handler in _todos_os_handlers():
        for existente in list(handler.filters):
            if isinstance(existente, FiltroSegredos):
                handler.removeFilter(existente)
        handler.addFilter(filtro)
    return filtro
