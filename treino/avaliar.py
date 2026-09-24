"""Mede o sistema contra o rótulo verdadeiro do PKLot.

Uso:
    python treino/avaliar.py --conjunto teste --amostras 200
    python treino/avaliar.py --conjunto validacao --detector vagas --pesos runs/.../best.pt

Relata acurácia por vaga, os dois tipos de erro separados e o tempo por
quadro. Os dois erros aparecem separados de propósito: dizer "livre" numa
vaga ocupada manda o motorista ao lugar errado, e dizer "ocupada" numa vaga
livre só esconde uma vaga. O primeiro é o caro.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2

from baliza import pklot
from baliza.deteccao import abrir_detector
from baliza.ocupacao import decidir_por_vaga, decidir_por_veiculo
from baliza.tipos import Estado

import divisao


def matriz_vazia() -> dict[str, int]:
    return {"vo_po": 0, "vo_pl": 0, "vl_po": 0, "vl_pl": 0, "sem_leitura": 0}


def avaliar(
    raiz: str,
    conjunto: str,
    amostras: int | None,
    modo: str,
    pesos: str | None,
    tamanho: int,
    confianca: float,
    limiar: float,
    semente: int = 42,
    por_estacionamento: bool = True,
    janelas: tuple[int, int] | None = None,
    estacionamentos: tuple[str, ...] | None = None,
) -> dict:
    fotos = divisao.conjunto(raiz, conjunto, limite_por_dia=4)
    if estacionamentos:
        fotos = [f for f in fotos if f.estacionamento in estacionamentos]
    if amostras is not None and len(fotos) > amostras:
        random.Random(semente).shuffle(fotos)
        fotos = fotos[:amostras]

    detector = abrir_detector(modo, pesos, confianca=confianca, tamanho=tamanho,
                              janelas=janelas)
    geral = matriz_vazia()
    por_grupo: dict[str, dict[str, int]] = {}
    tempos: list[float] = []
    descartadas = 0

    inicio = time.perf_counter()
    for indice, foto in enumerate(fotos):
        quadro = cv2.imread(str(foto.imagem))
        if quadro is None:
            continue
        # O mapa sai do XML da própria foto: na base a câmera é reposicionada
        # entre um dia e outro, e um mapa fixo por câmera sairia do lugar.
        try:
            mapa = pklot.mapa_de_anotacao(foto.anotacao, foto.estacionamento)
        except pklot.AnotacaoInvalida:
            # 274 dos 12.416 XMLs da base trazem vaga sem o atributo occupied.
            # Adivinhar o rótulo contaminaria a medição, então a foto sai.
            descartadas += 1
            continue
        verdadeiro = pklot.verdade(foto.anotacao)

        deteccoes = detector.detectar(quadro)
        tempos.append(detector.ms_ultima_inferencia)
        if modo == "vagas":
            resultado = decidir_por_vaga(mapa, deteccoes)
        else:
            resultado = decidir_por_veiculo(mapa, deteccoes, limiar=limiar)

        for chave in (None, foto.estacionamento, foto.clima) if por_estacionamento else (None,):
            alvo = geral if chave is None else por_grupo.setdefault(chave, matriz_vazia())
            for leitura in resultado.leituras:
                real = verdadeiro.get(leitura.vaga_id)
                if real is None:
                    continue
                if leitura.estado is Estado.SEM_LEITURA:
                    alvo["sem_leitura"] += 1
                elif real is Estado.OCUPADA and leitura.estado is Estado.OCUPADA:
                    alvo["vo_po"] += 1
                elif real is Estado.OCUPADA and leitura.estado is Estado.LIVRE:
                    alvo["vo_pl"] += 1
                elif real is Estado.LIVRE and leitura.estado is Estado.OCUPADA:
                    alvo["vl_po"] += 1
                else:
                    alvo["vl_pl"] += 1

        if indice and indice % 25 == 0:
            print(f"  {indice}/{len(fotos)} fotos", flush=True)

    duracao = time.perf_counter() - inicio
    saida = {
        "conjunto": conjunto,
        "modo": modo,
        "pesos": pesos or "yolo11n.pt (COCO)",
        "janelas": f"{janelas[0]}x{janelas[1]}" if janelas else "quadro inteiro",
        "tamanho": tamanho,
        "confianca": confianca,
        "limiar_cobertura": limiar if modo == "veiculos" else None,
        "fotos": len(fotos) - descartadas,
        "fotos_descartadas": descartadas,
        "segundos": round(duracao, 1),
        "ms_por_quadro": round(sum(tempos) / len(tempos), 1) if tempos else 0,
        "quadros_por_segundo": round(1000 / (sum(tempos) / len(tempos)), 2) if tempos else 0,
        "geral": metricas(geral),
    }
    if por_estacionamento:
        saida["grupos"] = {chave: metricas(m) for chave, m in sorted(por_grupo.items())}
    return saida


def metricas(m: dict[str, int]) -> dict:
    total = m["vo_po"] + m["vo_pl"] + m["vl_po"] + m["vl_pl"] + m["sem_leitura"]
    decididas = total - m["sem_leitura"]
    ocupadas_reais = m["vo_po"] + m["vo_pl"]
    livres_reais = m["vl_po"] + m["vl_pl"]
    acertos = m["vo_po"] + m["vl_pl"]
    preditas_ocupadas = m["vo_po"] + m["vl_po"]

    def taxa(numerador, denominador):
        return round(numerador / denominador, 4) if denominador else None

    return {
        "vagas_avaliadas": total,
        "matriz": dict(m),
        # Duas acurácias de propósito. A primeira ignora a vaga que o modelo
        # não leu, e e a que costuma aparecer em artigo; a segunda conta a vaga
        # não lida como erro, que e o que o motorista sente quando o painel não
        # sabe responder. Quando as duas se afastam, a diferença e a noticia.
        "acuracia": taxa(acertos, decididas),
        "acuracia_sobre_todas": taxa(acertos, total),
        "falso_livre": taxa(m["vo_pl"], ocupadas_reais),
        "falso_ocupado": taxa(m["vl_po"], livres_reais),
        "precisao_ocupada": taxa(m["vo_po"], preditas_ocupadas),
        "revocacao_ocupada": taxa(m["vo_po"], ocupadas_reais),
        "taxa_sem_leitura": taxa(m["sem_leitura"], total),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="avalia o Baliza contra o PKLot")
    parser.add_argument("--raiz", default="dados/PKLot")
    parser.add_argument("--conjunto", default="teste",
                        choices=("treino", "validacao", "teste"))
    parser.add_argument("--amostras", type=int, default=200)
    parser.add_argument("--detector", dest="modo", default="veiculos",
                        choices=("veiculos", "vagas"))
    parser.add_argument("--pesos", default=None)
    parser.add_argument("--tamanho", type=int, default=1280)
    parser.add_argument("--confianca", type=float, default=0.25)
    parser.add_argument("--limiar", type=float, default=0.30)
    parser.add_argument("--janelas", default=None,
                        help="grade de janelas deslizantes, por exemplo 3x2")
    parser.add_argument("--estacionamentos", default=None,
                        help="filtra por estacionamento, separado por virgula")
    parser.add_argument("--saida", default=None, help="grava o resultado em JSON")
    args = parser.parse_args()

    if args.pesos is None and args.modo == "veiculos":
        args.pesos = "modelos/yolo11n.pt"

    grade = None
    if args.janelas:
        colunas, linhas = args.janelas.lower().split("x")
        grade = (int(colunas), int(linhas))

    filtro = tuple(args.estacionamentos.split(",")) if args.estacionamentos else None

    resultado = avaliar(
        args.raiz, args.conjunto, args.amostras, args.modo, args.pesos,
        args.tamanho, args.confianca, args.limiar,
        janelas=grade, estacionamentos=filtro,
    )
    texto = json.dumps(resultado, indent=2, ensure_ascii=False)
    print(texto)
    if args.saida:
        Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
        Path(args.saida).write_text(texto, encoding="utf-8")
        print(f"\ngravado em {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
