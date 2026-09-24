"""Área de interseção entre a vaga e a caixa que o detector devolveu.

A vaga do PKLot vem como um quadrilátero girado, e a caixa do YOLO vem
alinhada aos eixos. Recortar um contra o outro com Sutherland-Hodgman dá a
área exata, e é barato: os dois polígonos são convexos e têm quatro lados.
"""

from __future__ import annotations

Ponto = tuple[float, float]
Poligono = list[Ponto]
Caixa = tuple[float, float, float, float]  # x1, y1, x2, y2


def area(poligono: Poligono) -> float:
    """Área pela fórmula do cadarço. Sempre positiva, em qualquer sentido."""
    n = len(poligono)
    if n < 3:
        return 0.0
    soma = 0.0
    for i in range(n):
        x1, y1 = poligono[i]
        x2, y2 = poligono[(i + 1) % n]
        soma += x1 * y2 - x2 * y1
    return abs(soma) / 2.0


def caixa_para_poligono(caixa: Caixa) -> Poligono:
    x1, y1, x2, y2 = caixa
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _lado(ponto: Ponto, a: Ponto, b: Ponto) -> float:
    """Positivo quando o ponto está à esquerda da reta a->b."""
    return (b[0] - a[0]) * (ponto[1] - a[1]) - (b[1] - a[1]) * (ponto[0] - a[0])


def _interseccao_reta(p: Ponto, q: Ponto, a: Ponto, b: Ponto) -> Ponto:
    x1, y1 = p
    x2, y2 = q
    x3, y3 = a
    x4, y4 = b
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if d == 0:
        return q
    t1 = x1 * y2 - y1 * x2
    t2 = x3 * y4 - y3 * x4
    return ((t1 * (x3 - x4) - (x1 - x2) * t2) / d,
            (t1 * (y3 - y4) - (y1 - y2) * t2) / d)


def _sentido_anti_horario(poligono: Poligono) -> Poligono:
    soma = 0.0
    n = len(poligono)
    for i in range(n):
        x1, y1 = poligono[i]
        x2, y2 = poligono[(i + 1) % n]
        soma += x1 * y2 - x2 * y1
    return poligono if soma >= 0 else list(reversed(poligono))


def recortar(sujeito: Poligono, janela: Poligono) -> Poligono:
    """Sutherland-Hodgman. A janela precisa ser convexa; a vaga e a caixa são."""
    if len(sujeito) < 3 or len(janela) < 3:
        return []
    saida = _sentido_anti_horario(list(sujeito))
    janela = _sentido_anti_horario(list(janela))

    for i in range(len(janela)):
        a = janela[i]
        b = janela[(i + 1) % len(janela)]
        entrada, saida = saida, []
        if not entrada:
            break
        anterior = entrada[-1]
        for atual in entrada:
            dentro_atual = _lado(atual, a, b) >= 0
            dentro_anterior = _lado(anterior, a, b) >= 0
            if dentro_atual:
                if not dentro_anterior:
                    saida.append(_interseccao_reta(anterior, atual, a, b))
                saida.append(atual)
            elif dentro_anterior:
                saida.append(_interseccao_reta(anterior, atual, a, b))
            anterior = atual
    return saida


def area_interseccao(poligono: Poligono, caixa: Caixa) -> float:
    return area(recortar(poligono, caixa_para_poligono(caixa)))


def fracao_coberta(poligono: Poligono, caixa: Caixa) -> float:
    """Quanto da vaga a caixa cobre, de 0 a 1.

    É essa razão, e não a IoU, que decide ocupação. A IoU pune o caminhão
    que cobre a vaga inteira e ainda sobra, e caminhão estacionado ocupa a
    vaga do mesmo jeito.
    """
    denominador = area(poligono)
    if denominador <= 0:
        return 0.0
    return min(1.0, area_interseccao(poligono, caixa) / denominador)


def iou(poligono: Poligono, caixa: Caixa) -> float:
    interseccao = area_interseccao(poligono, caixa)
    uniao = area(poligono) + area(caixa_para_poligono(caixa)) - interseccao
    return interseccao / uniao if uniao > 0 else 0.0


def envolvente(poligono: Poligono) -> Caixa:
    xs = [p[0] for p in poligono]
    ys = [p[1] for p in poligono]
    return (min(xs), min(ys), max(xs), max(ys))


def centro(poligono: Poligono) -> Ponto:
    n = len(poligono)
    return (sum(p[0] for p in poligono) / n, sum(p[1] for p in poligono) / n)
