"""Calibra e mede o detector clássico no PKLot.

Calibra com os rótulos de duas câmeras e mede na terceira. O modo global
usa somente estatísticas de treino e pode ser executado online. O modo câmera
é transdutivo: estima média e desvio usando toda a amostra da câmera de teste,
sem rótulos; não mede a execução online nem pode antecipar esses momentos.

Saídas:
    resultados/pdi_medidas.csv        medidas por vaga, com o rótulo
    resultados/pdi_calibracao.json    os pesos e o limiar escolhidos
    resultados/pdi_resultado.json     acurácia por câmera, clima e modo

Uso:
    python treino/calibrar_pdi.py --limite-por-camera 120
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from baliza.pdi import (  # noqa: E402
    NOMES_DAS_MEDIDAS,
    Calibracao,
    limiar_de_otsu,
    medir,
    recortar_vaga,
)


def ler_anotacao(xml: Path) -> list[tuple[list[tuple[int, int]], int]]:
    """Devolve [(contorno, ocupada)] de um XML do PKLot."""
    try:
        raiz = ET.parse(xml).getroot()
    except ET.ParseError:
        return []
    vagas = []
    for space in raiz.findall("space"):
        ocup = space.get("occupied")
        if ocup is None:
            continue
        contorno_el = space.find("contour")
        if contorno_el is None:
            continue
        pts = [
            (int(p.get("x")), int(p.get("y")))
            for p in contorno_el.findall("point")
        ]
        if len(pts) >= 4:
            vagas.append((pts[:4], int(ocup)))
    return vagas


def coletar(
    raiz_pklot: Path, limite_por_camera: int, semente: int = 0
) -> list[dict]:
    """Amostra imagens de cada câmera e clima e mede todas as vagas."""
    rng = np.random.default_rng(semente)
    linhas: list[dict] = []
    cameras = sorted(d for d in raiz_pklot.iterdir() if d.is_dir())
    for cam in cameras:
        for clima in sorted(d for d in cam.iterdir() if d.is_dir()):
            xmls = sorted(clima.rglob("*.xml"))
            if not xmls:
                continue
            n = min(limite_por_camera, len(xmls))
            escolhidos = [xmls[i] for i in rng.choice(len(xmls), n, replace=False)]
            for x in escolhidos:
                jpg = x.with_suffix(".jpg")
                if not jpg.exists():
                    continue
                quadro = cv2.imread(str(jpg))
                if quadro is None:
                    continue
                for contorno, ocup in ler_anotacao(x):
                    m = medir(recortar_vaga(quadro, contorno))
                    linhas.append(
                        {
                            "camera": cam.name,
                            "clima": clima.name,
                            "arquivo": str(jpg.relative_to(raiz_pklot)),
                            "ocupada": ocup,
                            **m.como_dict(),
                        }
                    )
            print(f"  {cam.name}/{clima.name}: {n} imagens, "
                  f"{len(linhas)} vagas acumuladas", flush=True)
    return linhas


def _acuracia(pont: np.ndarray, y: np.ndarray, limiar: float) -> float:
    return float(((pont >= limiar).astype(int) == y).mean())


def _f1(pont: np.ndarray, y: np.ndarray, limiar: float) -> float:
    pred = (pont >= limiar).astype(int)
    vp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    if vp == 0:
        return 0.0
    prec, rev = vp / (vp + fp), vp / (vp + fn)
    return 2 * prec * rev / (prec + rev)


def _melhor_limiar(
    pont: np.ndarray, y: np.ndarray, criterio: str = "f1"
) -> tuple[float, float]:
    """Varre os limiares candidatos e devolve (limiar, valor do criterio).

    O criterio padrao e F1, nao acuracia. Acuracia escolhe um limiar
    conservador demais quando as classes sao desbalanceadas, e foi o que
    aconteceu na primeira versao: precisao 0,98 e revocacao 0,48, ou seja, o
    detector quase nunca errava ao dizer "ocupada" e deixava metade das vagas
    ocupadas passar como livres.
    """
    cand = np.unique(np.quantile(pont, np.linspace(0.01, 0.99, 199)))
    f = _f1 if criterio == "f1" else _acuracia
    melhor_t, melhor_v = 0.0, -1.0
    for t in cand:
        v = f(pont, y, float(t))
        if v > melhor_v:
            melhor_v, melhor_t = v, float(t)
    return melhor_t, melhor_v


def normalizar_por_imagem(df, cols: list[str]) -> np.ndarray:
    """Padroniza cada medida DENTRO de cada imagem.

    O limiar calibrado numa camera nao transfere para outra porque a escala
    absoluta das medidas depende da camera: altura, distancia focal, tamanho
    da vaga em pixels e exposicao. Dentro de uma mesma imagem, porem, a
    comparacao entre vagas e justa, porque tudo isso e comum.

    Padronizar por imagem transforma a pergunta "esta vaga tem muita
    estrutura?" em "esta vaga tem mais estrutura que as outras deste quadro?",
    que e a pergunta que generaliza. E o mesmo principio da equalizacao
    adaptativa, aplicado ao vetor de medidas em vez de ao pixel.
    """
    g = df.groupby("arquivo")[cols]
    mu = g.transform("mean")
    sd = g.transform("std").replace(0, 1.0).fillna(1.0)
    return ((df[cols] - mu) / sd).to_numpy(float)


def calibrar(treino: np.ndarray, y: np.ndarray) -> Calibracao:
    """Escolhe pesos e limiar por varredura sobre combinações de medidas.

    Varre todos os subconjuntos de até três medidas, com peso +1 ou -1, e fica
    com o que der o maior F1 no treino. É busca exaustiva sobre um
    espaço pequeno e declarado, não ajuste de modelo: dá para escrever as
    constantes na monografia e conferir à mão.
    """
    medias = treino.mean(axis=0)
    desvios = treino.std(axis=0)
    desvios[desvios == 0] = 1.0
    z = (treino - medias) / desvios

    melhor = (np.zeros(5), 0.0, 0.0)
    for k in (1, 2, 3):
        for combo in itertools.combinations(range(5), k):
            for sinais in itertools.product((1.0, -1.0), repeat=k):
                w = np.zeros(5)
                for i, s in zip(combo, sinais):
                    w[i] = s
                pont = z @ w
                t, a = _melhor_limiar(pont, y)
                if a > melhor[2]:
                    melhor = (w.copy(), t, a)
    pesos, limiar, _ = melhor
    return Calibracao(
        medias=medias,
        desvios=desvios,
        pesos=pesos,
        limiar=limiar,
        acuracia_de_treino=_acuracia(z @ pesos, y, limiar),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pklot", type=str, default="dados/PKLot/PKLot")
    ap.add_argument("--limite-por-camera", type=int, default=80)
    ap.add_argument("--teste", type=str, default="UFPR05")
    ap.add_argument("--saida", type=str, default="resultados")
    ap.add_argument(
        "--normalizacao", choices=["global", "imagem", "camera"], default="global",
        help="global: compativel com runtime; camera: experimento transdutivo offline",
    )
    args = ap.parse_args()

    raiz = RAIZ / args.pklot
    if not raiz.exists():
        print(f"PKLot nao encontrado em {raiz}")
        return 1
    saida = RAIZ / args.saida
    saida.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cache = saida / "pdi_medidas.csv"
    if cache.exists():
        import pandas as pd

        df = pd.read_csv(cache)
        print(f"medidas lidas do cache: {len(df)} vagas")
    else:
        print("medindo vagas do PKLot...")
        linhas = coletar(raiz, args.limite_por_camera)
        import pandas as pd

        df = pd.DataFrame(linhas)
        df.to_csv(cache, index=False)
        print(f"{len(df)} vagas medidas em {(time.time()-t0)/60:.1f} min")

    import pandas as pd  # noqa: F811

    cols = list(NOMES_DAS_MEDIDAS)
    treino = df[df["camera"] != args.teste]
    teste = df[df["camera"] == args.teste]
    print(f"\ntreino: {len(treino)} vagas em "
          f"{sorted(treino['camera'].unique())}")
    print(f"teste:  {len(teste)} vagas em {args.teste} (camera nunca vista)")
    print(f"taxa de ocupacao no teste: {teste['ocupada'].mean():.3f}")

    # separacao de cada medida sozinha, so para o relatorio
    print("\n== poder de separacao de cada medida (AUC simples) ==")
    for c in cols:
        a = df[df["ocupada"] == 1][c].to_numpy()
        b = df[df["ocupada"] == 0][c].to_numpy()
        amostra = min(len(a), len(b), 4000)
        rng = np.random.default_rng(0)
        aa = rng.choice(a, amostra, replace=False)
        bb = rng.choice(b, amostra, replace=False)
        auc = float((aa[:, None] > bb[None, :]).mean())
        print(f"  {c:26s} AUC={auc:.3f}")

    modo_norm = args.normalizacao
    if modo_norm == "imagem":
        x_tr = normalizar_por_imagem(treino, cols)
        x_te = normalizar_por_imagem(teste, cols)
    elif modo_norm == "camera":
        # Padroniza as medidas DENTRO de cada camera. A escala absoluta das
        # medidas depende da camera (altura, distancia focal, tamanho da vaga
        # em pixels, exposicao), mas a comparacao entre vagas da MESMA camera
        # e justa. Este experimento e transdutivo: usa todas as medidas do
        # conjunto de teste para estimar momentos, inclusive quadros futuros.
        # Nao usa rotulos, mas nao representa a execucao online.
        def _por_camera(d):
            g = d.groupby("camera")[cols]
            mu = g.transform("mean")
            sd = g.transform("std").replace(0, 1.0).fillna(1.0)
            return ((d[cols] - mu) / sd).to_numpy(float)

        x_tr = _por_camera(treino)
        x_te = _por_camera(teste)
    else:
        x_tr = treino[cols].to_numpy(float)
        x_te = teste[cols].to_numpy(float)
    cal = calibrar(x_tr, treino["ocupada"].to_numpy(int))
    cal.normalizacao = modo_norm
    cal.camera_de_treino = "+".join(sorted(treino["camera"].unique()))
    usados = [cols[i] for i in np.nonzero(cal.pesos)[0]]
    print(f"\n== calibracao escolhida ==")
    print(f"  medidas usadas: {usados}")
    print(f"  pesos:  {dict(zip(cols, cal.pesos.round(2)))}")
    print(f"  limiar: {cal.limiar:.4f}")
    print(f"  acuracia no treino: {cal.acuracia_de_treino:.4f}")

    z_te = (x_te - cal.medias) / cal.desvios
    pont_te = z_te @ cal.pesos
    y_te = teste["ocupada"].to_numpy(int)
    acc_te = _acuracia(pont_te, y_te, cal.limiar)
    pred = (pont_te >= cal.limiar).astype(int)
    vp = int(((pred == 1) & (y_te == 1)).sum())
    fp = int(((pred == 1) & (y_te == 0)).sum())
    fn = int(((pred == 0) & (y_te == 1)).sum())
    vn = int(((pred == 0) & (y_te == 0)).sum())
    prec = vp / max(vp + fp, 1)
    rev = vp / max(vp + fn, 1)
    f1 = 2 * prec * rev / max(prec + rev, 1e-9)
    print(f"\n== CAMERA NUNCA VISTA ({args.teste}) ==")
    print(f"  acuracia={acc_te:.4f}  precisao={prec:.4f}  revocacao={rev:.4f}  "
          f"F1={f1:.4f}")
    print(f"  matriz: VP={vp} FP={fp} FN={fn} VN={vn}")

    por_clima = {}
    teste = teste.assign(_pont=pont_te)
    for clima, sub in teste.groupby("clima"):
        a = _acuracia(sub["_pont"].to_numpy(), sub["ocupada"].to_numpy(int), cal.limiar)
        por_clima[clima] = a
        print(f"  {clima:8s}: acuracia={a:.4f}  ({len(sub)} vagas)")

    # modo Otsu por imagem, sem constante nenhuma
    print("\n== modo Otsu por imagem (sem constante calibrada) ==")
    acertos, total, imagens_degeneradas = 0, 0, 0
    idx = cols.index("densidade_de_borda")
    for _arq, sub in teste.groupby("arquivo"):
        vals = sub[cols].to_numpy(float)[:, idx]
        y = sub["ocupada"].to_numpy(int)
        if y.min() == y.max():
            imagens_degeneradas += 1
        t = limiar_de_otsu(vals)
        acertos += int(((vals >= t).astype(int) == y).sum())
        total += len(y)
    acc_otsu = acertos / max(total, 1)
    print(f"  acuracia={acc_otsu:.4f} em {total} vagas")
    print(f"  imagens com patio todo cheio ou todo vazio: {imagens_degeneradas}")

    (saida / "pdi_calibracao.json").write_text(
        json.dumps(cal.como_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (saida / "pdi_resultado.json").write_text(
        json.dumps(
            {
                "normalizacao": args.normalizacao,
                "protocolo": (
                    f"rotulos de treino: {cal.camera_de_treino}; medido em "
                    f"{args.teste}; normalizacao {modo_norm}"
                ),
                "transdutivo": modo_norm == "camera",
                "compativel_com_runtime": modo_norm == "global",
                "n_treino": int(len(treino)),
                "n_teste": int(len(teste)),
                "taxa_de_ocupacao_no_teste": float(teste["ocupada"].mean()),
                "medidas_usadas": usados,
                "acuracia_treino": cal.acuracia_de_treino,
                "acuracia_teste": acc_te,
                "precisao": prec,
                "revocacao": rev,
                "f1": f1,
                "matriz": {"vp": vp, "fp": fp, "fn": fn, "vn": vn},
                "por_clima": por_clima,
                "acuracia_otsu_por_imagem": acc_otsu,
                "imagens_degeneradas_no_otsu": imagens_degeneradas,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nartefatos em {saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
