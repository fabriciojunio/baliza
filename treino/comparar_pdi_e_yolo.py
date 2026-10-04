"""Compara o detector clássico com o detector de veículos, nas MESMAS imagens.

A comparação só vale se os dois virem exatamente o mesmo conjunto. Números
tirados de amostras diferentes, com protocolos diferentes, não respondem
"quanto se perde sem rede neural"; respondem "quanto dois experimentos
diferentes deram".

Protocolo: as imagens amostradas da UFPR05, câmera que nenhum dos dois viu em
treino. O clássico foi calibrado em PUCPR e UFPR04; o YOLO usa os pesos gerais
do COCO, sem treino nenhum no PKLot.

Uso:
    python treino/comparar_pdi_e_yolo.py --limite-por-clima 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from baliza.geometria import fracao_coberta  # noqa: E402
from baliza.pdi import Calibracao, medir, recortar_vaga  # noqa: E402

CLASSES_VEICULO = {2, 3, 5, 7}


def ler_anotacao(xml: Path):
    try:
        raiz = ET.parse(xml).getroot()
    except ET.ParseError:
        return []
    saida = []
    for space in raiz.findall("space"):
        ocup = space.get("occupied")
        cont = space.find("contour")
        if ocup is None or cont is None:
            continue
        pts = [(int(p.get("x")), int(p.get("y"))) for p in cont.findall("point")]
        if len(pts) >= 4:
            saida.append((pts[:4], int(ocup)))
    return saida


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pklot", type=str, default="dados/PKLot/PKLot")
    ap.add_argument("--camera", type=str, default="UFPR05")
    ap.add_argument("--limite-por-clima", type=int, default=40)
    ap.add_argument("--limiar-cobertura", type=float, default=0.30)
    ap.add_argument("--confianca", type=float, default=0.25)
    # 1280 e o tamanho que treino/avaliar.py usa, e nao o padrao 640 do
    # ultralytics. A diferenca nao e detalhe: em 640 o carro no fundo do patio
    # tem poucos pixels e o detector simplesmente nao o ve, o que derruba a
    # revocacao e tornaria a comparacao injusta com o YOLO.
    ap.add_argument("--tamanho", type=int, default=1280)
    ap.add_argument("--saida", type=str, default="resultados")
    args = ap.parse_args()

    saida = RAIZ / args.saida
    cal_arq = saida / "pdi_calibracao.json"
    if not cal_arq.exists():
        print("rode treino/calibrar_pdi.py antes")
        return 1
    cal = Calibracao.de_dict(json.loads(cal_arq.read_text(encoding="utf-8")))

    cam = RAIZ / args.pklot / args.camera
    rng = np.random.default_rng(7)
    imagens: list[Path] = []
    for clima in sorted(d for d in cam.iterdir() if d.is_dir()):
        xmls = sorted(clima.rglob("*.xml"))
        if not xmls:
            continue
        n = min(args.limite_por_clima, len(xmls))
        imagens += [xmls[i] for i in rng.choice(len(xmls), n, replace=False)]
    print(f"{len(imagens)} imagens de {args.camera}")

    from ultralytics import YOLO

    modelo = YOLO(str(RAIZ / "modelos" / "yolo11n.pt"))

    y_true: list[int] = []
    pred_pdi: list[int] = []
    pred_yolo: list[int] = []
    ms_pdi, ms_yolo = 0.0, 0.0
    medidas_cam: list[np.ndarray] = []
    contornos_por_img: list[list] = []
    rotulos_por_img: list[list] = []

    # primeira passada: medidas do classico (precisa das estatisticas da camera
    # inteira para a padronizacao por camera, que e como ele foi calibrado)
    for x in imagens:
        jpg = x.with_suffix(".jpg")
        quadro = cv2.imread(str(jpg))
        if quadro is None:
            continue
        vagas = ler_anotacao(x)
        if not vagas:
            continue
        t0 = time.perf_counter()
        ms = [medir(recortar_vaga(quadro, c)).como_vetor() for c, _ in vagas]
        ms_pdi += (time.perf_counter() - t0) * 1000.0
        medidas_cam.append(np.array(ms))
        contornos_por_img.append([c for c, _ in vagas])
        rotulos_por_img.append([o for _, o in vagas])

    todas = np.vstack(medidas_cam)
    mu, sd = todas.mean(axis=0), todas.std(axis=0)
    sd[sd == 0] = 1.0

    for bloco, rot in zip(medidas_cam, rotulos_por_img):
        z = (bloco - mu) / sd
        z = (z - cal.medias) / np.where(cal.desvios == 0, 1.0, cal.desvios)
        pontos = z @ cal.pesos
        pred_pdi += [int(p >= cal.limiar) for p in pontos]
        y_true += rot

    # segunda passada: YOLO nas mesmas imagens
    k = 0
    for x in imagens:
        jpg = x.with_suffix(".jpg")
        quadro = cv2.imread(str(jpg))
        if quadro is None:
            continue
        vagas = ler_anotacao(x)
        if not vagas:
            continue
        t0 = time.perf_counter()
        r = modelo.predict(
            quadro, conf=args.confianca, imgsz=args.tamanho, verbose=False,
            device="cpu",
        )[0]
        ms_yolo += (time.perf_counter() - t0) * 1000.0
        caixas = []
        for b in r.boxes:
            if int(b.cls.item()) in CLASSES_VEICULO:
                xy = b.xyxy[0].tolist()
                caixas.append((xy[0], xy[1], xy[2], xy[3]))
        for contorno, _ in vagas:
            cobertura = max(
                (fracao_coberta(contorno, c) for c in caixas), default=0.0
            )
            pred_yolo.append(int(cobertura >= args.limiar_cobertura))
        k += 1

    y = np.array(y_true)
    pp = np.array(pred_pdi)
    py = np.array(pred_yolo)
    n = min(len(y), len(pp), len(py))
    y, pp, py = y[:n], pp[:n], py[:n]

    def metricas(pred):
        vp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        vn = int(((pred == 0) & (y == 0)).sum())
        prec = vp / max(vp + fp, 1)
        rev = vp / max(vp + fn, 1)
        return {
            "acuracia": float((pred == y).mean()),
            "precisao": prec,
            "revocacao": rev,
            "f1": 2 * prec * rev / max(prec + rev, 1e-9),
            "vp": vp, "fp": fp, "fn": fn, "vn": vn,
        }

    m_pdi, m_yolo = metricas(pp), metricas(py)
    print(f"\n{n} vagas comparadas, taxa de ocupacao {y.mean():.3f}")
    print(f"{'':22s} {'acuracia':>9s} {'precisao':>9s} {'revocacao':>10s} "
          f"{'F1':>7s} {'ms/imagem':>10s}")
    print(f"{'classico (PDI)':22s} {m_pdi['acuracia']:9.4f} "
          f"{m_pdi['precisao']:9.4f} {m_pdi['revocacao']:10.4f} "
          f"{m_pdi['f1']:7.4f} {ms_pdi/max(k,1):10.1f}")
    print(f"{'YOLO11n (COCO)':22s} {m_yolo['acuracia']:9.4f} "
          f"{m_yolo['precisao']:9.4f} {m_yolo['revocacao']:10.4f} "
          f"{m_yolo['f1']:7.4f} {ms_yolo/max(k,1):10.1f}")
    print(f"\ndiferenca de acuracia: "
          f"{(m_yolo['acuracia']-m_pdi['acuracia'])*100:+.1f} pontos para o YOLO")
    concorda = float((pp == py).mean())
    print(f"os dois concordam em {concorda:.3f} das vagas")

    (saida / "comparacao_pdi_yolo.json").write_text(
        json.dumps(
            {
                "camera": args.camera,
                "configuracao_yolo": {
                    "pesos": "yolo11n.pt (COCO)",
                    "tamanho": args.tamanho,
                    "confianca": args.confianca,
                    "limiar_cobertura": args.limiar_cobertura,
                    "dispositivo": "cpu",
                },
                "n_imagens": k,
                "n_vagas": int(n),
                "taxa_de_ocupacao": float(y.mean()),
                "classico": m_pdi | {"ms_por_imagem": ms_pdi / max(k, 1)},
                "yolo": m_yolo | {"ms_por_imagem": ms_yolo / max(k, 1)},
                "concordancia": concorda,
                "observacao": (
                    "Mesmas imagens para os dois. O classico foi calibrado em "
                    "PUCPR e UFPR04; o YOLO usa pesos gerais do COCO sem treino "
                    "no PKLot. Nenhum dos dois viu a UFPR05."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nartefato em {saida/'comparacao_pdi_yolo.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
