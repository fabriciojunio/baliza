"""Converte o PKLot para o formato que o Ultralytics lê.

Cada vaga do XML vira uma caixa alinhada aos eixos com a classe 0
(vaga-livre) ou 1 (vaga-ocupada). A divisão de treino, validação e teste é a
de `divisao.py`, por câmera e por dia.

    python treino/exportar_yolo.py --por-dia 20 --saida dados/yolo
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

from baliza.geometria import envolvente
from baliza.pklot import ler_anotacao
from baliza.tipos import Estado

import divisao

CLASSES = ["vaga-livre", "vaga-ocupada"]


def rotulo_de(anotacao: Path, largura: int, altura: int) -> list[str]:
    linhas = []
    for _, contorno, estado in ler_anotacao(anotacao):
        x1, y1, x2, y2 = envolvente(contorno)
        x1 = max(0.0, min(x1, largura))
        x2 = max(0.0, min(x2, largura))
        y1 = max(0.0, min(y1, altura))
        y2 = max(0.0, min(y2, altura))
        largura_caixa = (x2 - x1) / largura
        altura_caixa = (y2 - y1) / altura
        if largura_caixa <= 0 or altura_caixa <= 0:
            continue
        centro_x = (x1 + x2) / 2 / largura
        centro_y = (y1 + y2) / 2 / altura
        classe = 1 if estado is Estado.OCUPADA else 0
        linhas.append(
            f"{classe} {centro_x:.6f} {centro_y:.6f} {largura_caixa:.6f} {altura_caixa:.6f}"
        )
    return linhas


def exportar(raiz: str, saida: Path, por_dia: int, completo: bool = False) -> dict[str, int]:
    pastas = {"treino": "train", "validacao": "val", "teste": "test"}
    # A validação roda a cada época, então ela vai menor de propósito: o que
    # se quer dela é a curva, não a medição final, que sai no teste.
    limites = {"treino": por_dia, "validacao": max(2, por_dia // 4), "teste": 8}
    contagem = {}
    for nome, pasta in pastas.items():
        destino_img = saida / "images" / pasta
        destino_rot = saida / "labels" / pasta
        destino_img.mkdir(parents=True, exist_ok=True)
        destino_rot.mkdir(parents=True, exist_ok=True)

        escolher = divisao.conjunto_completo if completo else divisao.conjunto
        fotos = escolher(raiz, nome, limite_por_dia=limites[nome])
        escritas = 0
        for foto in fotos:
            imagem = cv2.imread(str(foto.imagem))
            if imagem is None:
                continue
            altura, largura = imagem.shape[:2]
            linhas = rotulo_de(foto.anotacao, largura, altura)
            if not linhas:
                continue
            base = f"{foto.estacionamento}_{foto.clima}_{foto.nome}"
            shutil.copy2(foto.imagem, destino_img / f"{base}.jpg")
            (destino_rot / f"{base}.txt").write_text("\n".join(linhas), encoding="utf-8")
            escritas += 1
            if escritas % 200 == 0:
                print(f"  {nome}: {escritas}", flush=True)
        contagem[nome] = escritas
        print(f"{nome}: {escritas} imagens")

    yaml = saida / "pklot.yaml"
    yaml.write_text(
        "# Gerado por treino/exportar_yolo.py. Divisao por camera e por dia.\n"
        f"path: {saida.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        + "".join(f"  {i}: {nome}\n" for i, nome in enumerate(CLASSES)),
        encoding="utf-8",
    )
    print(f"\n{yaml}")
    return contagem


def main() -> int:
    parser = argparse.ArgumentParser(description="PKLot -> formato YOLO")
    parser.add_argument("--raiz", default="dados/PKLot")
    parser.add_argument("--saida", default="dados/yolo")
    parser.add_argument("--por-dia", type=int, default=20,
                        help="maximo de fotos por dia e por clima")
    parser.add_argument("--completo", action="store_true",
                        help="inclui a UFPR05 no treino, para o modelo da demonstracao")
    args = parser.parse_args()
    exportar(args.raiz, Path(args.saida), args.por_dia, args.completo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
