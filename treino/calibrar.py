"""Escolhe o limiar de cobertura olhando o conjunto de validação.

A cobertura de cada vaga é calculada uma vez só e guardada; depois os
limiares são varridos em cima do que já foi medido. Sem isso, cada limiar
exigiria rodar o detector de novo na base inteira.

O critério de escolha não é a acurácia máxima. É a maior acurácia entre os
limiares que mantêm o falso livre abaixo do teto, porque mandar o motorista
para uma vaga ocupada é o erro que estraga a confiança no painel.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2

from baliza import pklot
from baliza.deteccao import abrir_detector
from baliza.geometria import fracao_coberta
from baliza.tipos import Estado

import divisao

TETO_FALSO_LIVRE = 0.03


def coletar(raiz: str, conjunto: str, amostras: int, pesos: str, tamanho: int,
            confianca: float, semente: int = 42) -> list[tuple[float, bool]]:
    """(cobertura maxima da vaga, vaga realmente ocupada) para cada vaga vista."""
    fotos = divisao.conjunto(raiz, conjunto, limite_por_dia=4)
    if len(fotos) > amostras:
        random.Random(semente).shuffle(fotos)
        fotos = fotos[:amostras]

    detector = abrir_detector("veiculos", pesos, confianca=confianca, tamanho=tamanho)
    pares: list[tuple[float, bool]] = []
    for indice, foto in enumerate(fotos):
        quadro = cv2.imread(str(foto.imagem))
        if quadro is None:
            continue
        try:
            mapa = pklot.mapa_de_anotacao(foto.anotacao, foto.estacionamento)
        except pklot.AnotacaoInvalida:
            # Um punhado de XMLs da base vem sem vaga nenhuma marcada.
            continue
        verdadeiro = pklot.verdade(foto.anotacao)
        deteccoes = detector.detectar(quadro)
        for vaga in mapa:
            real = verdadeiro.get(vaga.id)
            if real is None:
                continue
            melhor = 0.0
            for deteccao in deteccoes:
                cobertura = fracao_coberta(vaga.contorno, deteccao.caixa)
                if cobertura > melhor:
                    melhor = cobertura
            pares.append((melhor, real is Estado.OCUPADA))
        if indice and indice % 25 == 0:
            print(f"  {indice}/{len(fotos)} fotos", flush=True)
    return pares


def varrer(pares: list[tuple[float, bool]], limiares: list[float]) -> list[dict]:
    ocupadas = sum(1 for _, o in pares if o)
    livres = len(pares) - ocupadas
    linhas = []
    for limiar in limiares:
        vo_po = vo_pl = vl_po = vl_pl = 0
        for cobertura, real_ocupada in pares:
            predito_ocupada = cobertura >= limiar
            if real_ocupada and predito_ocupada:
                vo_po += 1
            elif real_ocupada:
                vo_pl += 1
            elif predito_ocupada:
                vl_po += 1
            else:
                vl_pl += 1
        linhas.append({
            "limiar": round(limiar, 3),
            "acuracia": round((vo_po + vl_pl) / len(pares), 4),
            "falso_livre": round(vo_pl / ocupadas, 4) if ocupadas else None,
            "falso_ocupado": round(vl_po / livres, 4) if livres else None,
        })
    return linhas


def main() -> int:
    parser = argparse.ArgumentParser(description="calibra o limiar de cobertura")
    parser.add_argument("--raiz", default="dados/PKLot")
    parser.add_argument("--conjunto", default="validacao")
    parser.add_argument("--amostras", type=int, default=120)
    parser.add_argument("--pesos", default="modelos/yolo11n.pt")
    parser.add_argument("--tamanho", type=int, default=1280)
    parser.add_argument("--confianca", type=float, default=0.25)
    parser.add_argument("--saida", default=None)
    args = parser.parse_args()

    pares = coletar(args.raiz, args.conjunto, args.amostras, args.pesos,
                    args.tamanho, args.confianca)
    print(f"\n{len(pares)} vagas coletadas em {args.conjunto}")

    limiares = [i / 100 for i in range(5, 81, 5)]
    linhas = varrer(pares, limiares)

    print(f"\n{'limiar':>8} {'acuracia':>9} {'falso livre':>12} {'falso ocupado':>14}")
    for linha in linhas:
        print(f"{linha['limiar']:>8.2f} {linha['acuracia']:>9.4f}"
              f" {linha['falso_livre']:>12.4f} {linha['falso_ocupado']:>14.4f}")

    aceitaveis = [l for l in linhas if (l["falso_livre"] or 1) <= TETO_FALSO_LIVRE]
    escolhido = max(aceitaveis or linhas, key=lambda l: l["acuracia"])
    print(f"\nescolhido: limiar {escolhido['limiar']:.2f}"
          f" (acuracia {escolhido['acuracia']:.4f},"
          f" falso livre {escolhido['falso_livre']:.4f})")
    if not aceitaveis:
        print(f"nenhum limiar ficou abaixo do teto de {TETO_FALSO_LIVRE:.0%} de falso livre")

    if args.saida:
        Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
        Path(args.saida).write_text(
            json.dumps({"varredura": linhas, "escolhido": escolhido,
                        "amostras": args.amostras, "vagas": len(pares),
                        "conjunto": args.conjunto},
                       indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"gravado em {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
