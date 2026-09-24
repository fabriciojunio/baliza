"""Gera as figuras da documentação: quadros anotados, lado a lado.

    python treino/figuras.py --rotulo coco
    python treino/figuras.py --rotulo treinado --modo vagas --pesos modelos/vagas.pt

Guarda em docs/figuras/<apelido>_<rotulo>.jpg e imprime o acerto daquele
quadro, que é o que vai na legenda da figura.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from baliza import pklot
from baliza.desenho import anotar, legenda
from baliza.deteccao import abrir_detector
from baliza.mapa import dividir_em_setores
from baliza.ocupacao import decidir_por_vaga, decidir_por_veiculo
from baliza.tipos import Estado

SAIDA = Path("docs/figuras")

# Um quadro de cada câmera, no meio do dia, que é quando o pátio está cheio.
ALVOS = {
    "ufpr04": ("UFPR04", "Sunny"),
    "ufpr05": ("UFPR05", "Cloudy"),
    "pucpr": ("PUCPR", "Sunny"),
}


def escolher(raiz: str, estacionamento: str, clima: str):
    """Um quadro com o patio pela metade.

    Patio vazio as seis da manha ou lotado ao meio-dia rende figura bonita e
    sem informacao: qualquer sistema acerta tudo quando a resposta e sempre a
    mesma. O que interessa e o quadro em que livre e ocupada convivem.
    """
    fotos = pklot.listar_fotos(raiz, estacionamentos=(estacionamento,), climas=(clima,),
                               limite_por_dia=6)
    melhor = None
    melhor_distancia = 1.0
    for foto in fotos:
        try:
            verdadeiro = pklot.verdade(foto.anotacao)
            pklot.mapa_de_anotacao(foto.anotacao, estacionamento)
        except pklot.AnotacaoInvalida:
            continue
        if not verdadeiro:
            continue
        ocupacao = sum(1 for e in verdadeiro.values() if e is Estado.OCUPADA) / len(verdadeiro)
        distancia = abs(ocupacao - 0.5)
        if distancia < melhor_distancia:
            melhor_distancia, melhor = distancia, foto
    return melhor


def main() -> int:
    parser = argparse.ArgumentParser(description="gera as figuras da documentacao")
    parser.add_argument("--raiz", default="dados/PKLot")
    parser.add_argument("--rotulo", default="coco")
    parser.add_argument("--modo", default="veiculos", choices=("veiculos", "vagas"))
    parser.add_argument("--pesos", default="modelos/yolo11n.pt")
    parser.add_argument("--tamanho", type=int, default=1280)
    parser.add_argument("--janelas", default=None)
    parser.add_argument("--limiar", type=float, default=0.30)
    args = parser.parse_args()

    grade = None
    if args.janelas:
        c, l = args.janelas.lower().split("x")
        grade = (int(c), int(l))

    SAIDA.mkdir(parents=True, exist_ok=True)
    detector = abrir_detector(args.modo, args.pesos, tamanho=args.tamanho, janelas=grade)
    relatorio = {}

    for apelido, (estacionamento, clima) in ALVOS.items():
        foto = escolher(args.raiz, estacionamento, clima)
        if foto is None:
            print(f"{apelido}: sem foto utilizavel")
            continue
        quadro = cv2.imread(str(foto.imagem))
        mapa = dividir_em_setores(
            pklot.mapa_de_anotacao(foto.anotacao, estacionamento, 1280, 720), 3
        )
        verdadeiro = pklot.verdade(foto.anotacao)
        deteccoes = detector.detectar(quadro)
        if args.modo == "vagas":
            resultado = decidir_por_vaga(mapa, deteccoes)
        else:
            resultado = decidir_por_veiculo(mapa, deteccoes, limiar=args.limiar)

        certas = sum(1 for l in resultado.leituras
                     if verdadeiro.get(l.vaga_id) is l.estado)
        falso_livre = sum(1 for l in resultado.leituras
                          if l.estado is Estado.LIVRE
                          and verdadeiro.get(l.vaga_id) is Estado.OCUPADA)

        caminho = SAIDA / f"{apelido}_{args.rotulo}.jpg"
        cv2.imwrite(str(caminho), legenda(anotar(quadro, mapa, resultado)),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

        relatorio[apelido] = {
            "foto": foto.nome,
            "estacionamento": estacionamento,
            "clima": clima,
            "vagas": len(mapa),
            "certas": certas,
            "falso_livre": falso_livre,
            "ocupadas_previstas": resultado.ocupadas,
            "ocupadas_reais": sum(1 for e in verdadeiro.values() if e is Estado.OCUPADA),
            "ms": round(detector.ms_ultima_inferencia, 1),
            "arquivo": str(caminho).replace("\\", "/"),
        }
        print(f"{apelido}: {certas}/{len(mapa)} vagas certas, {falso_livre} falso livre"
              f" ({detector.ms_ultima_inferencia:.0f} ms) -> {caminho}")

    caminho = SAIDA / "figuras.json"
    anterior = json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else {}
    anterior[args.rotulo] = relatorio
    caminho.write_text(json.dumps(anterior, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
