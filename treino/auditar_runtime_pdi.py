"""Verifica partições e mede o runtime clássico nos três dias reais da demo.

As quatro condições de imagem derivam das mesmas fotos, portanto não são
amostras independentes. Os vídeos são sequências dessas fotos recomprimidas,
não gravações contínuas. Somente UFPR05 é câmera excluída da calibração.

    python treino/auditar_runtime_pdi.py --calibracao artefatos/auditoria-2026-10-11/pdi_calibracao.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import divisao  # noqa: E402
from baliza.captura import FonteVideo  # noqa: E402
from baliza.deteccao import abrir_detector  # noqa: E402
from baliza.mapa import Mapa  # noqa: E402
from baliza.pklot import listar_fotos, verdade  # noqa: E402
from baliza.sistema import Baliza  # noqa: E402
from baliza.tipos import Estado  # noqa: E402


def variantes(quadro):
    yield "jpeg_original", quadro
    ok, comprimida = cv2.imencode(".jpg", quadro, [cv2.IMWRITE_JPEG_QUALITY, 25])
    if not ok:
        raise RuntimeError("falha ao gerar JPEG de qualidade 25")
    yield "jpeg_qualidade_25", cv2.imdecode(comprimida, cv2.IMREAD_COLOR)
    altura, largura = quadro.shape[:2]
    menor = cv2.resize(quadro, (largura // 2, altura // 2), interpolation=cv2.INTER_AREA)
    yield "resolucao_metade", cv2.resize(menor, (largura, altura))
    yield "intensidade_metade", (quadro.astype(float) * 0.5).astype("uint8")


def acumular(resumo, resultado, rotulos):
    resumo["quadros"] += 1
    resumo["ms_inferencia_total"] += resultado.ms_inferencia
    for leitura in resultado.leituras:
        rotulo = rotulos.get(leitura.vaga_id)
        if rotulo is None:
            resumo["sem_gabarito"] += 1
            continue
        resumo["leituras_com_gabarito"] += 1
        resumo["acertos"] += int(leitura.estado == rotulo)
        resumo["falsos_livres"] += int(rotulo == Estado.OCUPADA and leitura.estado == Estado.LIVRE)
        resumo["falsos_ocupados"] += int(rotulo == Estado.LIVRE and leitura.estado == Estado.OCUPADA)


def vazio():
    return dict(quadros=0, leituras_com_gabarito=0, sem_gabarito=0, acertos=0,
                falsos_livres=0, falsos_ocupados=0, ms_inferencia_total=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibracao", required=True)
    parser.add_argument("--pklot", default="dados/PKLot")
    parser.add_argument("--saida", default="artefatos/auditoria-2026-10-11/runtime_pdi.json")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    raiz_base = RAIZ / args.pklot
    cal = RAIZ / args.calibracao
    detector = abrir_detector("classico", cal)
    if "UFPR05" in detector.calibracao.camera_de_treino.split("+"):
        raise ValueError("UFPR05 aparece na calibracao; nao e camera de teste independente")

    partes = {nome: divisao.conjunto_completo(str(raiz_base), nome, 4)
              for nome in ("treino", "validacao", "teste")}
    grupos = {nome: {(f.estacionamento, f.dia) for f in fotos}
              for nome, fotos in partes.items()}
    for a, b in (("treino", "validacao"), ("treino", "teste"), ("validacao", "teste")):
        if grupos[a] & grupos[b]:
            raise RuntimeError(f"camera/dia compartilhada entre {a} e {b}")
    todas = listar_fotos(raiz_base, limite_por_dia=4)
    if set(todas) != set().union(*(set(fotos) for fotos in partes.values())):
        raise RuntimeError("particao perdeu ou duplicou fotos")
    impares = [f for f in todas if not divisao.dia_par(f)]
    grupo_antigo_a = {(f.estacionamento, f.dia) for f in impares[::2]}
    grupo_antigo_b = {(f.estacionamento, f.dia) for f in impares[1::2]}

    metadados = json.loads((RAIZ / "resultados/validacao_final.json").read_text(encoding="utf-8"))
    saida = {
        "normalizacao": detector.calibracao.normalizacao,
        "camera_de_treino": detector.calibracao.camera_de_treino,
        "calibracao_sha256": hashlib.sha256(cal.read_bytes()).hexdigest(),
        "janela_suavizacao": 1,
        "particoes": {nome: {"fotos": len(partes[nome]), "grupos_camera_dia": len(grupos[nome])}
                      for nome in partes},
        "sobreposicao_camera_dia_antigo_validacao_teste": len(grupo_antigo_a & grupo_antigo_b),
        "sobreposicao_camera_dia_corrigido_validacao_teste": 0,
        "camera_excluida_da_calibracao": "UFPR05",
        "limites": "Fotos, variantes e videos compartilham os mesmos 3 dias; nao sao amostras independentes.",
        "por_camera": {},
    }
    for apelido, meta in metadados["dias"].items():
        chave_dia = (meta["estacionamento"], meta["dia"])
        particao_demo = next((nome for nome, grupo in grupos.items() if chave_dia in grupo), None)
        if particao_demo is None:
            raise RuntimeError(f"{apelido}: dia da demonstracao fora das particoes")
        fotos = sorted(listar_fotos(raiz_base, estacionamentos=(meta["estacionamento"],),
                                   climas=(meta["clima"],), dias={meta["dia"]}), key=lambda f: f.nome)
        if len(fotos) != meta["fotos"]:
            raise RuntimeError(f"{apelido}: numero de fotos difere do manifesto")
        mapa = Mapa.carregar(RAIZ / "demo/mapas" / f"{apelido}.json")
        sistema = Baliza(mapa, detector, janela_suavizacao=1)
        condicoes = {}
        for foto in fotos:
            quadro = cv2.imread(str(foto.imagem))
            if quadro is None:
                raise RuntimeError(f"foto indisponivel: {foto.imagem}")
            rotulos = verdade(foto.anotacao)
            for condicao, img in variantes(quadro):
                resumo = condicoes.setdefault(condicao, vazio())
                acumular(resumo, sistema.processar(img), rotulos)
        fonte = FonteVideo(RAIZ / "demo/patios" / f"{apelido}.mp4")
        try:
            resumo_video = condicoes.setdefault("video_mp4_demo", vazio())
            for indice, quadro in fonte.quadros():
                if indice >= len(fotos):
                    raise RuntimeError(f"{apelido}: video tem mais quadros que fotos")
                acumular(resumo_video, sistema.processar(quadro), verdade(fotos[indice].anotacao))
        finally:
            fonte.fechar()
        if resumo_video["quadros"] != len(fotos):
            raise RuntimeError(f"{apelido}: video e fotos desalinhados")
        for resumo in condicoes.values():
            resumo["acuracia"] = resumo["acertos"] / resumo["leituras_com_gabarito"]
            resumo["ms_por_quadro"] = resumo.pop("ms_inferencia_total") / resumo["quadros"]
        saida["por_camera"][apelido] = {"camera": meta["estacionamento"], "dia": meta["dia"],
                                         "particao_no_protocolo_completo": particao_demo,
                                         "selecao_historica_da_demo": True,
                                         "condicoes": condicoes}
        print(f"{apelido}: {len(fotos)} fotos x 4 condicoes + {resumo_video['quadros']} quadros de video", flush=True)
    destino = RAIZ / args.saida
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"relatorio em {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
