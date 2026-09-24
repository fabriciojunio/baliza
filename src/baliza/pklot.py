"""Leitura da base PKLot.

Cada foto vem acompanhada de um XML com o contorno de todas as vagas e o
rótulo de ocupação de cada uma. É de lá que sai o mapa de vagas de cada
câmera e é contra ele que o sistema é medido.

Referência: ALMEIDA, P. et al. PKLot: a robust dataset for parking lot
classification. Expert Systems with Applications, v. 42, n. 11, 2015.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .mapa import Mapa
from .tipos import Estado, Vaga

ESTACIONAMENTOS = ("UFPR04", "UFPR05", "PUCPR")
CLIMAS = ("Sunny", "Cloudy", "Rainy")


@dataclass(frozen=True)
class Foto:
    """Uma foto da base, com o XML ao lado."""

    imagem: Path
    anotacao: Path
    estacionamento: str
    clima: str
    dia: str

    @property
    def nome(self) -> str:
        return self.imagem.stem


class AnotacaoInvalida(ValueError):
    pass


def _texto_ponto(elemento) -> tuple[float, float]:
    return float(elemento.get("x")), float(elemento.get("y"))


def ler_anotacao(caminho: str | Path) -> list[tuple[str, list[tuple[float, float]], Estado]]:
    """Devolve (id da vaga, contorno, estado verdadeiro) para cada vaga do XML.

    Vaga sem o atributo `occupied` é descartada: existe um punhado delas na
    base e adivinhar o rótulo contaminaria a medição.
    """
    caminho = Path(caminho)
    try:
        raiz = ET.parse(caminho).getroot()
    except ET.ParseError as erro:
        raise AnotacaoInvalida(f"{caminho.name}: XML quebrado ({erro})") from erro

    vagas = []
    for espaco in raiz.findall("space"):
        identificador = espaco.get("id")
        ocupado = espaco.get("occupied")
        contorno_no = espaco.find("contour")
        if identificador is None or ocupado is None or contorno_no is None:
            continue
        contorno = [_texto_ponto(p) for p in contorno_no.findall("point")]
        if len(contorno) < 3:
            continue
        estado = Estado.OCUPADA if ocupado.strip() == "1" else Estado.LIVRE
        vagas.append((identificador, contorno, estado))
    return vagas


def mapa_de_anotacao(caminho: str | Path, camera: str, largura: int = 0, altura: int = 0) -> Mapa:
    vagas = [Vaga(id=i, contorno=c) for i, c, _ in ler_anotacao(caminho)]
    if not vagas:
        raise AnotacaoInvalida(f"{caminho}: nenhuma vaga utilizavel")
    return Mapa(camera=camera, vagas=vagas, largura=largura, altura=altura)


def verdade(caminho: str | Path) -> dict[str, Estado]:
    return {i: e for i, _, e in ler_anotacao(caminho)}


def _raiz_das_fotos(raiz: Path) -> Path:
    """Aceita tanto a pasta extraída quanto a pasta PKLot/PKLot de dentro dela."""
    for candidata in (raiz / "PKLot" / "PKLot", raiz / "PKLot", raiz):
        if any((candidata / nome).is_dir() for nome in ESTACIONAMENTOS):
            return candidata
    raise FileNotFoundError(f"nao achei UFPR04/UFPR05/PUCPR dentro de {raiz}")


def listar_fotos(
    raiz: str | Path,
    estacionamentos: tuple[str, ...] = ESTACIONAMENTOS,
    climas: tuple[str, ...] = CLIMAS,
    dias: set[str] | None = None,
    limite_por_dia: int | None = None,
) -> list[Foto]:
    """Varre a base e devolve as fotos que têm imagem e XML.

    `limite_por_dia` existe porque a base fotografa de 5 em 5 minutos: para
    medir, uma foto a cada N já cobre o dia inteiro sem repetir quase o mesmo
    quadro dezenas de vezes.
    """
    base = _raiz_das_fotos(Path(raiz))
    fotos: list[Foto] = []
    for estacionamento in estacionamentos:
        pasta_estacionamento = base / estacionamento
        if not pasta_estacionamento.is_dir():
            continue
        for clima in climas:
            pasta_clima = pasta_estacionamento / clima
            if not pasta_clima.is_dir():
                continue
            for pasta_dia in sorted(p for p in pasta_clima.iterdir() if p.is_dir()):
                if dias is not None and pasta_dia.name not in dias:
                    continue
                do_dia = []
                for imagem in sorted(pasta_dia.glob("*.jpg")):
                    anotacao = imagem.with_suffix(".xml")
                    if anotacao.exists():
                        do_dia.append(
                            Foto(imagem, anotacao, estacionamento, clima, pasta_dia.name)
                        )
                if limite_por_dia is not None and len(do_dia) > limite_por_dia:
                    passo = len(do_dia) / limite_por_dia
                    do_dia = [do_dia[int(i * passo)] for i in range(limite_por_dia)]
                fotos.extend(do_dia)
    return fotos


def dias_disponiveis(raiz: str | Path, estacionamento: str) -> list[str]:
    base = _raiz_das_fotos(Path(raiz)) / estacionamento
    dias = set()
    for clima in CLIMAS:
        pasta = base / clima
        if pasta.is_dir():
            dias.update(p.name for p in pasta.iterdir() if p.is_dir())
    return sorted(dias)
