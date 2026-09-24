import numpy as np
import pytest

from baliza.mapa import Mapa
from baliza.tipos import Deteccao, Vaga


def vaga_retangular(identificador, x, y, largura=10, altura=10, setor="A"):
    return Vaga(
        id=identificador,
        contorno=[(x, y), (x + largura, y), (x + largura, y + altura), (x, y + altura)],
        setor=setor,
    )


@pytest.fixture
def mapa_de_quatro():
    """Quatro vagas em fila, dois setores."""
    return Mapa(
        camera="teste",
        vagas=[
            vaga_retangular("1", 0, 0, setor="A"),
            vaga_retangular("2", 20, 0, setor="A"),
            vaga_retangular("3", 40, 0, setor="B"),
            vaga_retangular("4", 60, 0, setor="B"),
        ],
        largura=100,
        altura=50,
    )


@pytest.fixture
def carro_na_vaga_1():
    return [Deteccao((0, 0, 10, 10), "carro", 0.9)]


@pytest.fixture
def quadro():
    return np.full((50, 100, 3), 120, dtype=np.uint8)


class DetectorFalso:
    """Devolve o que mandarem, sem carregar modelo nenhum."""

    def __init__(self, deteccoes=None, ms=7.5):
        self.deteccoes = deteccoes or []
        self.ms_ultima_inferencia = ms
        self.chamadas = 0

    def detectar(self, quadro):
        self.chamadas += 1
        return list(self.deteccoes)


@pytest.fixture
def detector_falso():
    return DetectorFalso
