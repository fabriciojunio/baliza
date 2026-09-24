"""De onde vêm os quadros: imagem, vídeo ou câmera.

Estacionamento muda devagar, então o padrão não é processar tudo. Uma
leitura a cada poucos segundos descreve o pátio tão bem quanto trinta por
segundo e deixa a máquina livre.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FonteIndisponivel(RuntimeError):
    pass


class Fonte:
    nome = "fonte"
    fps = 0.0
    total_quadros = 0

    def quadros(self) -> Iterator[tuple[int, np.ndarray]]:  # pragma: sem cobertura
        raise NotImplementedError

    def fechar(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.fechar()


class FonteImagem(Fonte):
    def __init__(self, caminho: str | Path, repetir: int = 1):
        self.caminho = Path(caminho)
        self.repetir = max(1, repetir)
        self.nome = self.caminho.stem
        quadro = cv2.imread(str(self.caminho))
        if quadro is None:
            raise FonteIndisponivel(f"nao consegui abrir a imagem: {caminho}")
        self._quadro = quadro
        self.total_quadros = self.repetir

    def quadros(self) -> Iterator[tuple[int, np.ndarray]]:
        for indice in range(self.repetir):
            yield indice, self._quadro.copy()


class FontePasta(Fonte):
    """Uma pasta de fotos em ordem, que é como a base PKLot fotografa o pátio."""

    def __init__(self, pasta: str | Path, limite: int | None = None):
        self.pasta = Path(pasta)
        arquivos = sorted(
            p for p in self.pasta.iterdir() if p.suffix.lower() in EXTENSOES_IMAGEM
        )
        if not arquivos:
            raise FonteIndisponivel(f"nenhuma imagem em {pasta}")
        if limite is not None:
            arquivos = arquivos[:limite]
        self.arquivos = arquivos
        self.nome = self.pasta.name
        self.total_quadros = len(arquivos)

    def quadros(self) -> Iterator[tuple[int, np.ndarray]]:
        for indice, arquivo in enumerate(self.arquivos):
            quadro = cv2.imread(str(arquivo))
            if quadro is not None:
                yield indice, quadro


class FonteVideo(Fonte):
    def __init__(self, caminho: str | Path, intervalo_s: float = 0.0,
                 limite: int | None = None):
        self.caminho = Path(caminho)
        self.limite = limite
        captura = cv2.VideoCapture(str(self.caminho))
        if not captura.isOpened():
            raise FonteIndisponivel(f"nao consegui abrir o video: {caminho}")
        self._captura = captura
        self.nome = self.caminho.stem
        self.fps = captura.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_quadros = int(captura.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.passo = max(1, int(round(self.fps * intervalo_s))) if intervalo_s > 0 else 1

    def quadros(self) -> Iterator[tuple[int, np.ndarray]]:
        indice = 0
        entregues = 0
        while True:
            ok, quadro = self._captura.read()
            if not ok:
                break
            if indice % self.passo == 0:
                yield indice, quadro
                entregues += 1
                if self.limite is not None and entregues >= self.limite:
                    break
            indice += 1

    def fechar(self) -> None:
        self._captura.release()


class FonteCamera(Fonte):
    """Câmera local ou RTSP.

    No Windows vale tentar mais de um backend: o primeiro que abre nem sempre
    é o que entrega quadro de verdade, então só aceitamos o que devolver um.
    """

    BACKENDS = (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY)

    def __init__(self, alvo: int | str = 0, intervalo_s: float = 0.0):
        self.alvo = alvo
        captura = None
        for backend in self.BACKENDS:
            tentativa = cv2.VideoCapture(alvo, backend)
            if tentativa.isOpened():
                ok, _ = tentativa.read()
                if ok:
                    captura = tentativa
                    break
            tentativa.release()
        if captura is None:
            raise FonteIndisponivel(f"nenhum backend entregou quadro da camera {alvo}")
        self._captura = captura
        self.nome = f"camera{alvo}"
        self.fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
        self.intervalo_s = intervalo_s

    def quadros(self) -> Iterator[tuple[int, np.ndarray]]:
        import time

        indice = 0
        proximo = 0.0
        while True:
            ok, quadro = self._captura.read()
            if not ok:
                break
            agora = time.perf_counter()
            if agora >= proximo:
                proximo = agora + self.intervalo_s
                yield indice, quadro
            indice += 1

    def fechar(self) -> None:
        self._captura.release()


def abrir_fonte(alvo: str | int, intervalo_s: float = 0.0, limite: int | None = None) -> Fonte:
    """Descobre sozinha o que é o alvo: índice de câmera, URL, pasta ou arquivo."""
    if isinstance(alvo, int) or (isinstance(alvo, str) and alvo.isdigit()):
        return FonteCamera(int(alvo), intervalo_s)
    texto = str(alvo)
    if texto.startswith(("rtsp://", "http://", "https://")):
        return FonteCamera(texto, intervalo_s)
    caminho = Path(texto)
    if caminho.is_dir():
        return FontePasta(caminho, limite)
    if not caminho.exists():
        raise FonteIndisponivel(f"nao existe: {caminho}")
    if caminho.suffix.lower() in EXTENSOES_IMAGEM:
        return FonteImagem(caminho)
    return FonteVideo(caminho, intervalo_s, limite)
