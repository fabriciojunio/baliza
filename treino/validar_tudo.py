"""Roda todo detector disponível contra todo material de demonstração.

    python treino/validar_tudo.py

Três frentes, e cada uma responde uma pergunta diferente:

  demo    os três vídeos da pasta demo, quadro a quadro, contra o rótulo
          verdadeiro da foto que gerou aquele quadro. É o que o professor vai
          ver na tela, então é o que precisa estar certo.
  fotos   as nove fotos avulsas, uma a uma, com o número exato de cada uma.
  base    uma amostra dos dias que nenhum modelo usou em treino, nas três
          câmeras e nos três climas.

O resultado vai para resultados/validacao_final.json e é ele que decide qual
detector fica recomendado no mapa de cada câmera.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

import divisao
import montar_demo
from baliza import pklot
from baliza.deteccao import abrir_detector
from baliza.mapa import Mapa
from baliza.ocupacao import decidir_por_vaga, decidir_por_veiculo
from baliza.tipos import Estado

MODELOS = {
    "veiculos": ("veiculos", "modelos/yolo11n.pt", None),
    "veiculos-janelas": ("veiculos", "modelos/yolo11n.pt", (3, 2)),
    "vagas-experimento": ("vagas", "modelos/vagas-experimento.pt", None),
    "vagas": ("vagas", "modelos/vagas.pt", None),
}


def disponiveis() -> dict[str, tuple]:
    return {nome: cfg for nome, cfg in MODELOS.items() if Path(cfg[1]).exists()}


class Placar:
    def __init__(self):
        self.certas = self.total = self.falso_livre = self.falso_ocupado = 0
        self.ocupadas = self.livres = self.sem_leitura = 0

    def somar(self, leituras, verdade):
        for l in leituras:
            real = verdade.get(l.vaga_id)
            if real is None:
                continue
            self.total += 1
            if real is Estado.OCUPADA:
                self.ocupadas += 1
            else:
                self.livres += 1
            if l.estado is Estado.SEM_LEITURA:
                self.sem_leitura += 1
            elif l.estado is real:
                self.certas += 1
            elif real is Estado.OCUPADA:
                self.falso_livre += 1
            else:
                self.falso_ocupado += 1

    def resumo(self) -> dict:
        def taxa(n, d):
            return round(n / d, 4) if d else None
        return {
            "vagas": self.total,
            "acuracia": taxa(self.certas, self.total),
            "falso_livre": taxa(self.falso_livre, self.ocupadas),
            "falso_ocupado": taxa(self.falso_ocupado, self.livres),
            "sem_leitura": taxa(self.sem_leitura, self.total),
        }


def medir(detector, modo, mapa, fotos, limiar=0.30) -> Placar:
    placar = Placar()
    for foto in fotos:
        try:
            verdade = pklot.verdade(foto.anotacao)
        except pklot.AnotacaoInvalida:
            continue
        if not verdade:
            continue
        quadro = cv2.imread(str(foto.imagem))
        if quadro is None:
            continue
        deteccoes = detector.detectar(quadro)
        alvo = mapa if mapa is not None else pklot.mapa_de_anotacao(
            foto.anotacao, foto.estacionamento)
        resultado = (decidir_por_vaga(alvo, deteccoes) if modo == "vagas"
                     else decidir_por_veiculo(alvo, deteccoes, limiar=limiar))
        placar.somar(resultado.leituras, verdade)
    return placar


def main() -> int:
    parser = argparse.ArgumentParser(description="valida tudo que vai para a apresentacao")
    parser.add_argument("--raiz", default="dados/PKLot")
    parser.add_argument("--amostra-base", type=int, default=40,
                        help="fotos por camera na frente 'base'")
    parser.add_argument("--saida", default="resultados/validacao_final.json")
    args = parser.parse_args()

    modelos = disponiveis()
    print("modelos encontrados:", ", ".join(modelos) or "nenhum")
    saida: dict = {"demo": {}, "fotos": {}, "base": {}}

    # --- os tres videos da demonstracao, quadro a quadro ---
    dias = {}
    for apelido, (estacionamento, clima) in montar_demo.DIAS.items():
        dias[apelido] = montar_demo.escolher_dia(args.raiz, estacionamento, clima)

    for nome, (modo, pesos, janelas) in modelos.items():
        detector = abrir_detector(modo, pesos, tamanho=1280, janelas=janelas)
        for apelido, fotos in dias.items():
            mapa = Mapa.carregar(f"demo/mapas/{apelido}.json")
            placar = medir(detector, modo, mapa, fotos)
            saida["demo"].setdefault(apelido, {})[nome] = placar.resumo()
            r = placar.resumo()
            print(f"  demo {apelido:7s} {nome:18s} acuracia {r['acuracia']*100:5.1f}%"
                  f"  falso livre {(r['falso_livre'] or 0)*100:5.1f}%"
                  f"  sem leitura {(r['sem_leitura'] or 0)*100:5.1f}%", flush=True)

        # --- as nove fotos avulsas ---
        for apelido in dias:
            pasta = Path("demo/fotos") / apelido
            nomes = {p.stem for p in pasta.glob("*.jpg")}
            escolhidas = [f for f in dias[apelido] if f"{apelido}_{f.nome}" in nomes]
            if not escolhidas:
                continue
            mapa = Mapa.carregar(f"demo/mapas/{apelido}.json")
            placar = medir(detector, modo, mapa, escolhidas)
            saida["fotos"].setdefault(apelido, {})[nome] = placar.resumo()

        # --- amostra ampla da base, dias que ninguem treinou ---
        for estacionamento in divisao.TODOS:
            fotos = [f for f in divisao.conjunto_completo(args.raiz, "teste", limite_por_dia=4)
                     if f.estacionamento == estacionamento][: args.amostra_base]
            if not fotos:
                continue
            placar = medir(detector, modo, None, fotos)
            saida["base"].setdefault(estacionamento, {})[nome] = placar.resumo()
            r = placar.resumo()
            print(f"  base {estacionamento:7s} {nome:18s} acuracia {r['acuracia']*100:5.1f}%"
                  f"  falso livre {(r['falso_livre'] or 0)*100:5.1f}%", flush=True)

    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida).write_text(json.dumps(saida, indent=2, ensure_ascii=False),
                                encoding="utf-8")

    # --- quem fica recomendado em cada camera ---
    print("\nrecomendacao por camera (maior acuracia na propria camera):")
    recomendacao = {}
    for apelido, por_modelo in saida["demo"].items():
        melhor = max(por_modelo.items(), key=lambda kv: kv[1]["acuracia"] or 0)
        modo = MODELOS[melhor[0]][0]
        recomendacao[apelido] = modo
        print(f"  {apelido:7s} {melhor[0]:18s} {melhor[1]['acuracia']*100:.1f}%  -> {modo}")
    saida["recomendacao"] = recomendacao
    Path(args.saida).write_text(json.dumps(saida, indent=2, ensure_ascii=False),
                                encoding="utf-8")
    print(f"\n{args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
