"""Ajuste fino do YOLO11n no PKLot, duas classes.

    python treino/treinar.py --epocas 40 --tamanho 1280 --lote 4

A GPU da máquina de desenvolvimento tem 4 GB e não tem núcleos tensores, o
que limita o lote muito antes de a teoria limitar. Se estourar a memória, o
script cai de lote sozinho e avisa, em vez de morrer no meio da noite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def treinar(dados: str, pesos: str, epocas: int, tamanho: int, lote: int,
            paciencia: int, nome: str, semente: int = 42) -> dict:
    from ultralytics import YOLO

    modelo = YOLO(pesos)
    tentativas = [lote] + [l for l in (4, 2, 1) if l < lote]
    ultimo_erro = None

    for tentativa in tentativas:
        try:
            resultado = modelo.train(
                data=dados,
                epochs=epocas,
                imgsz=tamanho,
                batch=tentativa,
                patience=paciencia,
                seed=semente,
                device=0,
                workers=2,
                amp=True,
                # Caminho absoluto de propósito: o Ultralytics guarda um
                # runs_dir global e sem isso a saida vai parar em outro projeto.
                project=str(Path("runs").resolve()),
                name=nome,
                exist_ok=True,
                # A câmera é fixa e parafusada: girar muito a imagem ensina
                # uma geometria que nunca vai aparecer em operação.
                degrees=5.0,
                fliplr=0.5,
                flipud=0.0,
                mosaic=1.0,
                hsv_h=0.015,
                hsv_s=0.4,
                hsv_v=0.4,
                plots=True,
            )
            return {
                "lote_usado": tentativa,
                "pasta": str(resultado.save_dir),
                "melhor": str(Path(resultado.save_dir) / "weights" / "best.pt"),
            }
        except RuntimeError as erro:
            if "out of memory" not in str(erro).lower():
                raise
            ultimo_erro = erro
            print(f"\nlote {tentativa} nao coube na VRAM, caindo para o proximo\n")
            import torch

            torch.cuda.empty_cache()

    raise RuntimeError(f"nem com lote 1 coube: {ultimo_erro}")


def main() -> int:
    parser = argparse.ArgumentParser(description="treina o detector de vagas")
    parser.add_argument("--dados", default="dados/yolo/pklot.yaml")
    parser.add_argument("--pesos", default="modelos/yolo11n.pt")
    parser.add_argument("--epocas", type=int, default=40)
    parser.add_argument("--tamanho", type=int, default=1280)
    parser.add_argument("--lote", type=int, default=4)
    parser.add_argument("--paciencia", type=int, default=10)
    parser.add_argument("--nome", default="vagas")
    args = parser.parse_args()

    saida = treinar(args.dados, args.pesos, args.epocas, args.tamanho,
                    args.lote, args.paciencia, args.nome)
    print(json.dumps(saida, indent=2, ensure_ascii=False))

    registro = Path("resultados/treinos.jsonl")
    registro.parent.mkdir(parents=True, exist_ok=True)
    with registro.open("a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps({**vars(args), **saida}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
