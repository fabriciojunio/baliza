"""Os quatro tipos que circulam entre os módulos."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .geometria import Poligono, centro, envolvente


class Estado(str, Enum):
    LIVRE = "livre"
    OCUPADA = "ocupada"
    SEM_LEITURA = "sem_leitura"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Vaga:
    """Uma vaga desenhada no mapa. Não muda durante a execução."""

    id: str
    contorno: Poligono
    setor: str = "A"

    @property
    def centro(self):
        return centro(self.contorno)

    @property
    def envolvente(self):
        return envolvente(self.contorno)


@dataclass(frozen=True)
class Deteccao:
    """O que o detector devolveu para um quadro."""

    caixa: tuple[float, float, float, float]
    classe: str
    confianca: float


@dataclass
class Leitura:
    """A decisão sobre uma vaga em um instante."""

    vaga_id: str
    estado: Estado
    confianca: float = 0.0
    cobertura: float = 0.0
    classe_detectada: str | None = None

    @property
    def ocupada(self) -> bool:
        return self.estado is Estado.OCUPADA


@dataclass
class Resultado:
    """O retorno de um quadro inteiro."""

    leituras: list[Leitura] = field(default_factory=list)
    ms_inferencia: float = 0.0

    @property
    def livres(self) -> int:
        return sum(1 for l in self.leituras if l.estado is Estado.LIVRE)

    @property
    def ocupadas(self) -> int:
        return sum(1 for l in self.leituras if l.estado is Estado.OCUPADA)

    @property
    def sem_leitura(self) -> int:
        return sum(1 for l in self.leituras if l.estado is Estado.SEM_LEITURA)

    @property
    def total(self) -> int:
        return len(self.leituras)

    def por_setor(self, vagas: dict[str, Vaga]) -> dict[str, tuple[int, int]]:
        """setor -> (livres, total). Vaga sem leitura não conta como livre."""
        contagem: dict[str, list[int]] = {}
        for leitura in self.leituras:
            vaga = vagas.get(leitura.vaga_id)
            setor = vaga.setor if vaga else "?"
            par = contagem.setdefault(setor, [0, 0])
            par[1] += 1
            if leitura.estado is Estado.LIVRE:
                par[0] += 1
        return {setor: (par[0], par[1]) for setor, par in sorted(contagem.items())}
