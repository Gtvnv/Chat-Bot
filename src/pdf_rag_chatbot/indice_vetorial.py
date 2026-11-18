"""Índice vetorial em numpy, persistido em disco (Indice_Vetorial).

Armazenamento em ``diretorio`` (formato 2):

- ``atual.json``: manifesto ``{"versao": "<id>", "versao_formato": 2}``.
- ``versoes/<id>/vectors.npy``: matriz ``float32`` ``(n, d)`` normalizada (norma 1).
- ``versoes/<id>/chunks.json``: Trechos na mesma ordem das linhas da matriz.
- ``versoes/<id>/meta.json``: ``modelo_embeddings``, ``dimensao`` e ``versao_formato``.

A gravação cria uma versão nova e troca o manifesto com um único ``os.replace``.
O formato 1 (três arquivos na raiz) ainda é lido e migrado na próxima gravação.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pdf_rag_chatbot.erros import (
    ErroEmbeddings,
    ErroIndiceCorrompido,
    ErroIndiceIncompativel,
)
from pdf_rag_chatbot.modelos import ResultadoBusca, Trecho

VERSAO_FORMATO = 2
VERSAO_FORMATO_LEGADO = 1
ARQUIVO_MANIFESTO = "atual.json"
DIRETORIO_VERSOES = "versoes"
ARQUIVO_VETORES = "vectors.npy"
ARQUIVO_TRECHOS = "chunks.json"
ARQUIVO_META = "meta.json"
_TIPOS_VALIDOS = ("pdf", "texto")


def _normalizar_linhas(matriz: np.ndarray) -> np.ndarray:
    """Normaliza cada linha para norma 1; rejeita vetor nulo ou não finito."""
    matriz = np.asarray(matriz, dtype=np.float32)
    if not np.all(np.isfinite(matriz)):
        raise ErroEmbeddings("Vetor de embedding contém valores não finitos.")
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    if np.any(normas == 0):
        raise ErroEmbeddings("Vetor de embedding nulo não pode ser indexado.")
    return (matriz / normas).astype(np.float32)


class IndiceVetorial:
    """Índice de Trechos com busca exata por similaridade de cosseno."""

    def __init__(self, diretorio: Path, nome_modelo: str) -> None:
        self._diretorio = Path(diretorio)
        self._nome_modelo = nome_modelo
        self._trechos: list[Trecho] = []
        self._vetores: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._dimensao: int | None = None

    # ------------------------------------------------------------------
    # Consulta de estado
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._trechos)

    def esta_vazio(self) -> bool:
        return len(self._trechos) == 0

    def listar_fontes(self) -> list[str]:
        """Nomes das fontes em ordem alfabética, sem repetição."""
        return sorted({t.fonte for t in self._trechos})

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------
    def carregar(self) -> None:
        """Carrega o índice do disco. Diretório inexistente começa vazio.

        Lê ``atual.json`` e a versão apontada (formato 2). Sem manifesto, aceita
        os três arquivos na raiz (formato 1). Versões não referenciadas são
        ignoradas.
        """
        vazio = np.empty((0, 0), dtype=np.float32)
        if not self._diretorio.exists():
            self._definir_estado([], vazio, None)
            return

        caminho_manifesto = self._diretorio / ARQUIVO_MANIFESTO
        if caminho_manifesto.exists():
            manifesto = self._ler_json(caminho_manifesto)
            if not isinstance(manifesto, dict):
                raise ErroIndiceCorrompido("atual.json inválido.")
            versao = manifesto.get("versao")
            if (
                not isinstance(versao, str)
                or not versao
                or versao in (".", "..")
                or "/" in versao
                or "\\" in versao
                or manifesto.get("versao_formato") != VERSAO_FORMATO
            ):
                raise ErroIndiceCorrompido("atual.json com campos ausentes ou inválidos.")
            pasta = self._diretorio / DIRETORIO_VERSOES / versao
            if not pasta.is_dir():
                raise ErroIndiceCorrompido(
                    "O manifesto aponta para uma versão inexistente do índice. "
                    "Reindexe os PDFs."
                )
            self._carregar_pasta(pasta, VERSAO_FORMATO)
            return

        nomes = (ARQUIVO_META, ARQUIVO_TRECHOS, ARQUIVO_VETORES)
        existentes = [(self._diretorio / n).exists() for n in nomes]
        if not any(existentes):
            self._definir_estado([], vazio, None)
            return
        if not all(existentes):
            raise ErroIndiceCorrompido(
                "Arquivos do índice incompletos. Reindexe os PDFs."
            )
        self._carregar_pasta(self._diretorio, VERSAO_FORMATO_LEGADO)

    def _carregar_pasta(self, pasta: Path, versao_esperada: int) -> None:
        caminho_meta = pasta / ARQUIVO_META
        caminho_trechos = pasta / ARQUIVO_TRECHOS
        caminho_vetores = pasta / ARQUIVO_VETORES
        if not all(p.exists() for p in (caminho_meta, caminho_trechos, caminho_vetores)):
            raise ErroIndiceCorrompido(
                "Arquivos do índice incompletos. Reindexe os PDFs."
            )

        meta = self._ler_json(caminho_meta)
        if not isinstance(meta, dict):
            raise ErroIndiceCorrompido("meta.json inválido.")
        modelo = meta.get("modelo_embeddings")
        dimensao = meta.get("dimensao")
        if not isinstance(modelo, str) or not isinstance(dimensao, int) or isinstance(
            dimensao, bool
        ):
            raise ErroIndiceCorrompido("meta.json com campos ausentes ou inválidos.")
        if meta.get("versao_formato") != versao_esperada:
            raise ErroIndiceCorrompido("Versão de formato do índice não suportada.")
        if modelo != self._nome_modelo:
            raise ErroIndiceIncompativel(
                f"O índice foi criado com o modelo '{modelo}', mas o modelo atual é "
                f"'{self._nome_modelo}'. Reindexe os PDFs com o modelo atual."
            )

        bruto = self._ler_json(caminho_trechos)
        if not isinstance(bruto, list):
            raise ErroIndiceCorrompido("chunks.json inválido.")
        trechos = [self._trecho_de_json(item) for item in bruto]

        try:
            vetores = np.load(caminho_vetores, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ErroIndiceCorrompido(f"vectors.npy ilegível: {exc}") from exc

        if vetores.ndim != 2 or vetores.dtype != np.float32:
            raise ErroIndiceCorrompido("vectors.npy com formato ou tipo inesperado.")
        if vetores.shape[0] != len(trechos):
            raise ErroIndiceCorrompido(
                "Número de vetores diferente do número de Trechos."
            )
        if vetores.shape[1] != dimensao:
            raise ErroIndiceCorrompido("Dimensão dos vetores diferente de meta.json.")
        if not np.all(np.isfinite(vetores)):
            raise ErroIndiceCorrompido("vectors.npy contém valores não finitos.")

        self._definir_estado(trechos, vetores, dimensao)
    # ------------------------------------------------------------------
    # Escrita
    # ------------------------------------------------------------------
    def substituir_fonte(
        self, fonte: str, trechos: Sequence[Trecho], vetores: np.ndarray
    ) -> None:
        """Substitui todos os Trechos de ``fonte`` de forma tudo-ou-nada.

        Lista vazia de Trechos remove a fonte. O novo estado é montado por
        completo, gravado em uma versão nova e publicado com a troca
        atômica do manifesto; a memória só é atualizada depois disso.
        """
        trechos = list(trechos)
        vetores = np.asarray(vetores, dtype=np.float32)

        for t in trechos:
            if t.fonte != fonte:
                raise ValueError(
                    f"Trecho com fonte '{t.fonte}' diferente de '{fonte}'."
                )
            if t.pagina < 1:
                raise ValueError("A página do Trecho deve ser maior ou igual a 1.")

        manter = [i for i, t in enumerate(self._trechos) if t.fonte != fonte]

        if not trechos:
            if len(manter) == len(self._trechos):
                return  # fonte inexistente: nada a remover
            novos_trechos = [self._trechos[i] for i in manter]
            novos_vetores = self._vetores[manter]
            dimensao = self._dimensao
        else:
            if vetores.ndim != 2 or vetores.shape[0] != len(trechos):
                raise ErroEmbeddings(
                    "A quantidade de vetores deve ser igual à de Trechos."
                )
            dimensao = vetores.shape[1]
            if dimensao == 0:
                raise ErroEmbeddings("Vetores de embedding sem dimensões.")
            if manter and self._dimensao is not None and dimensao != self._dimensao:
                raise ErroEmbeddings(
                    f"Dimensão {dimensao} incompatível com a do índice "
                    f"({self._dimensao})."
                )
            normalizados = _normalizar_linhas(vetores)
            novos_trechos = [self._trechos[i] for i in manter] + trechos
            base = self._vetores[manter] if manter else np.empty((0, dimensao), np.float32)
            novos_vetores = np.vstack([base, normalizados]).astype(np.float32)

        self._persistir(novos_trechos, novos_vetores, dimensao)
        self._definir_estado(novos_trechos, novos_vetores, dimensao)

    # ------------------------------------------------------------------
    # Busca
    # ------------------------------------------------------------------
    def buscar(
        self,
        vetor: np.ndarray,
        k: int,
        fontes: Collection[str] | None = None,
    ) -> list[ResultadoBusca]:
        """Devolve até ``k`` Trechos mais similares, em ordem decrescente.

        Empates são resolvidos pela posição de inserção. ``fontes`` restringe
        a busca aos arquivos informados.
        """
        if k < 1:
            raise ValueError("k deve ser maior ou igual a 1.")
        consulta = np.asarray(vetor, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(consulta)) or not np.any(consulta):
            raise ErroEmbeddings("Vetor de consulta nulo ou inválido.")
        if self.esta_vazio():
            return []
        if consulta.shape[0] != self._dimensao:
            raise ErroEmbeddings(
                f"Dimensão da consulta ({consulta.shape[0]}) diferente da do "
                f"índice ({self._dimensao})."
            )
        consulta = consulta / np.linalg.norm(consulta)

        if fontes is None:
            posicoes = np.arange(len(self._trechos))
            matriz = self._vetores
        else:
            permitidas = set(fontes)
            posicoes = np.array(
                [i for i, t in enumerate(self._trechos) if t.fonte in permitidas],
                dtype=np.int64,
            )
            if posicoes.size == 0:
                return []
            matriz = self._vetores[posicoes]

        similaridades = matriz @ consulta
        n = similaridades.shape[0]
        k_efetivo = min(k, n)

        if k_efetivo < n:
            escolhidos = np.argpartition(-similaridades, k_efetivo - 1)[:k_efetivo]
            limiar = similaridades[escolhidos].min()
            # inclui todos os empatados no limiar para o desempate ser determinístico
            candidatos = np.flatnonzero(similaridades >= limiar)
        else:
            candidatos = np.arange(n)

        # lexsort: última chave é a primária; empate resolvido pela posição
        ordem = np.lexsort((posicoes[candidatos], -similaridades[candidatos]))
        finais = candidatos[ordem][:k_efetivo]

        return [
            ResultadoBusca(
                trecho=self._trechos[int(posicoes[i])],
                similaridade=float(similaridades[i]),
            )
            for i in finais
        ]

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------
    def _definir_estado(
        self, trechos: list[Trecho], vetores: np.ndarray, dimensao: int | None
    ) -> None:
        self._trechos = trechos
        self._vetores = vetores
        self._dimensao = dimensao

    @staticmethod
    def _ler_json(caminho: Path) -> Any:
        try:
            return json.loads(caminho.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ErroIndiceCorrompido(f"{caminho.name} ilegível: {exc}") from exc

    @staticmethod
    def _trecho_de_json(item: Any) -> Trecho:
        if not isinstance(item, dict):
            raise ErroIndiceCorrompido("Item de chunks.json inválido.")
        texto = item.get("texto")
        fonte = item.get("fonte")
        pagina = item.get("pagina")
        tipo = item.get("tipo", "pdf")
        if (
            not isinstance(texto, str)
            or not isinstance(fonte, str)
            or not isinstance(pagina, int)
            or isinstance(pagina, bool)
            or pagina < 1
            or tipo not in _TIPOS_VALIDOS
        ):
            raise ErroIndiceCorrompido("Item de chunks.json com campos inválidos.")
        return Trecho(texto=texto, fonte=fonte, pagina=pagina, tipo=tipo)

    def _persistir(
        self, trechos: list[Trecho], vetores: np.ndarray, dimensao: int | None
    ) -> None:
        """Grava uma versão nova e publica com um único ``os.replace`` do manifesto."""
        if dimensao is None:
            dimensao = vetores.shape[1]
        if vetores.shape[0] == 0:
            vetores = np.empty((0, dimensao), dtype=np.float32)

        conteudo_trechos = json.dumps(
            [
                {"texto": t.texto, "fonte": t.fonte, "pagina": t.pagina, "tipo": t.tipo}
                for t in trechos
            ],
            ensure_ascii=False,
        )
        conteudo_meta = json.dumps(
            {
                "modelo_embeddings": self._nome_modelo,
                "dimensao": int(dimensao),
                "versao_formato": VERSAO_FORMATO,
            },
            ensure_ascii=False,
        )

        agora = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        versao = f"{agora}-{uuid.uuid4().hex[:8]}"
        raiz_versoes = self._diretorio / DIRETORIO_VERSOES
        pasta = raiz_versoes / versao
        manifesto = self._diretorio / ARQUIVO_MANIFESTO
        manifesto_tmp = self._diretorio / f"{ARQUIVO_MANIFESTO}.{os.getpid()}.tmp"
        versoes_existiam = raiz_versoes.exists()
        diretorio_existia = self._diretorio.exists()

        try:
            pasta.mkdir(parents=True)
            with open(pasta / ARQUIVO_VETORES, "wb") as f:
                np.save(f, vetores.astype(np.float32), allow_pickle=False)
                f.flush()
                os.fsync(f.fileno())
            for nome, conteudo in (
                (ARQUIVO_TRECHOS, conteudo_trechos),
                (ARQUIVO_META, conteudo_meta),
            ):
                with open(pasta / nome, "w", encoding="utf-8") as f:
                    f.write(conteudo)
                    f.flush()
                    os.fsync(f.fileno())
            self._sincronizar_diretorio(pasta)

            with open(manifesto_tmp, "w", encoding="utf-8") as f:
                json.dump({"versao": versao, "versao_formato": VERSAO_FORMATO}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(manifesto_tmp, manifesto)
        except BaseException:
            # Falha antes da troca do manifesto: o estado anterior segue intacto.
            try:
                manifesto_tmp.unlink(missing_ok=True)
            except OSError:
                pass
            shutil.rmtree(pasta, ignore_errors=True)
            if not versoes_existiam:
                self._remover_se_vazio(raiz_versoes)
            if not diretorio_existia:
                self._remover_se_vazio(self._diretorio)
            raise

        self._sincronizar_diretorio(self._diretorio)
        self._limpar_versoes_antigas(versao)

    def _limpar_versoes_antigas(self, versao_atual: str) -> None:
        """Remove versões não atuais e arquivos do formato 1; ignora falhas."""
        try:
            for item in (self._diretorio / DIRETORIO_VERSOES).iterdir():
                if item.name != versao_atual:
                    shutil.rmtree(item, ignore_errors=True)
        except OSError:
            pass
        for nome in (ARQUIVO_VETORES, ARQUIVO_TRECHOS, ARQUIVO_META):
            try:
                (self._diretorio / nome).unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _remover_se_vazio(caminho: Path) -> None:
        try:
            caminho.rmdir()
        except OSError:
            pass

    @staticmethod
    def _sincronizar_diretorio(caminho: Path) -> None:
        """``fsync`` do diretório onde o sistema permite (POSIX); no Windows é ignorado."""
        if os.name == "nt":
            return
        try:
            fd = os.open(caminho, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)