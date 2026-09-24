"""Anotação do quadro: contorno de cada vaga, placar e legenda.

O OpenCV só desenha ASCII nas fontes internas, então todo texto que vai para
a imagem passa por `_sem_acento`. Sem isso, "Câmera" vira "C??mera" na tela.
"""

from __future__ import annotations

import unicodedata

import cv2
import numpy as np

from .mapa import Mapa
from .tipos import Estado, Resultado

# BGR, que é a ordem do OpenCV.
COR = {
    Estado.LIVRE: (80, 175, 70),
    Estado.OCUPADA: (55, 55, 200),
    Estado.SEM_LEITURA: (60, 160, 220),
}
BRANCO = (255, 255, 255)
PRETO = (25, 25, 25)
FONTE = cv2.FONT_HERSHEY_SIMPLEX


def _sem_acento(texto: str) -> str:
    normalizado = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in normalizado if not unicodedata.combining(c))


def _escrever(quadro, texto, posicao, escala=0.5, cor=BRANCO, espessura=1, fundo=None):
    texto = _sem_acento(texto)
    if fundo is not None:
        (largura, altura), base = cv2.getTextSize(texto, FONTE, escala, espessura)
        x, y = posicao
        cv2.rectangle(
            quadro,
            (x - 3, y - altura - 4),
            (x + largura + 3, y + base + 1),
            fundo,
            cv2.FILLED,
        )
    cv2.putText(quadro, texto, posicao, FONTE, escala, cor, espessura, cv2.LINE_AA)


def anotar(
    quadro,
    mapa: Mapa,
    resultado: Resultado,
    mostrar_id: bool = True,
    preencher: bool = True,
) -> np.ndarray:
    """Devolve uma cópia do quadro com as vagas desenhadas."""
    saida = quadro.copy()
    camada = saida.copy() if preencher else None
    por_id = mapa.por_id

    for leitura in resultado.leituras:
        vaga = por_id.get(leitura.vaga_id)
        if vaga is None:
            continue
        pontos = np.array([[int(x), int(y)] for x, y in vaga.contorno], dtype=np.int32)
        cor = COR[leitura.estado]
        if camada is not None:
            cv2.fillPoly(camada, [pontos], cor)
        cv2.polylines(saida, [pontos], True, cor, 2, cv2.LINE_AA)

    if camada is not None:
        cv2.addWeighted(camada, 0.25, saida, 0.75, 0, saida)

    if mostrar_id:
        for leitura in resultado.leituras:
            vaga = por_id.get(leitura.vaga_id)
            if vaga is None:
                continue
            x, y = vaga.centro
            _escrever(saida, vaga.id, (int(x) - 8, int(y) + 4), 0.35, BRANCO, 1)

    _placar(saida, mapa, resultado)
    return saida


def _placar(quadro, mapa: Mapa, resultado: Resultado) -> None:
    altura, largura = quadro.shape[:2]
    caixa_altura = 62
    cv2.rectangle(quadro, (0, 0), (largura, caixa_altura), PRETO, cv2.FILLED)

    livres = resultado.livres
    _escrever(quadro, "BALIZA", (12, 24), 0.62, BRANCO, 2)
    _escrever(quadro, f"camera {mapa.camera}", (110, 24), 0.45, (190, 190, 190), 1)

    texto = f"{livres} LIVRES de {resultado.total}"
    _escrever(quadro, texto, (12, 48), 0.6, COR[Estado.LIVRE], 2)

    if resultado.sem_leitura:
        _escrever(
            quadro,
            f"{resultado.sem_leitura} sem leitura",
            (240, 48),
            0.45,
            COR[Estado.SEM_LEITURA],
            1,
        )

    por_setor = resultado.por_setor(mapa.por_id)
    if len(por_setor) > 1:
        partes = " | ".join(f"{setor}: {l}/{t}" for setor, (l, t) in por_setor.items())
        # A largura vem do próprio OpenCV: estimar por número de caracteres
        # corta o último setor fora da tela quando ha três ou mais.
        (largura_texto, _), _ = cv2.getTextSize(_sem_acento(partes), FONTE, 0.45, 1)
        _escrever(quadro, partes, (max(12, largura - 12 - largura_texto), 24), 0.45, BRANCO, 1)

    if resultado.ms_inferencia:
        _escrever(
            quadro,
            f"{resultado.ms_inferencia:.0f} ms",
            (largura - 70, 48),
            0.45,
            (190, 190, 190),
            1,
        )


def legenda(quadro) -> np.ndarray:
    """Faixa de rodapé explicando as cores. Usada na demonstração."""
    altura, largura = quadro.shape[:2]
    faixa = 26
    saida = quadro.copy()
    cv2.rectangle(saida, (0, altura - faixa), (largura, altura), PRETO, cv2.FILLED)
    x = 12
    for estado, rotulo in (
        (Estado.LIVRE, "livre"),
        (Estado.OCUPADA, "ocupada"),
        (Estado.SEM_LEITURA, "sem leitura"),
    ):
        cv2.rectangle(
            saida, (x, altura - faixa + 8), (x + 14, altura - faixa + 20), COR[estado], cv2.FILLED
        )
        _escrever(saida, rotulo, (x + 20, altura - faixa + 19), 0.42, BRANCO, 1)
        x += 30 + 9 * len(rotulo)
    return saida
