"""Baliza: ocupação de vagas de estacionamento a partir de câmera fixa."""

from .mapa import Mapa
from .tipos import Deteccao, Estado, Leitura, Resultado, Vaga

__version__ = "1.0.0"

__all__ = ["Mapa", "Vaga", "Deteccao", "Leitura", "Resultado", "Estado", "__version__"]
