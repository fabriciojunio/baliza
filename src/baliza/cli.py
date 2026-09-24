"""Linha de comando do Baliza.

    baliza rodar demo/patios/ufpr04_dia.mp4 --mapa demo/mapas/ufpr04.json
    baliza rodar 0 --mapa meu_patio.json --mostrar
    baliza mapa-do-pklot dados/PKLot ... --saida demo/mapas/ufpr04.json
    baliza demonstracao
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DEMO = RAIZ / "demo"


def _comando_rodar(args) -> int:
    from .captura import FonteIndisponivel, abrir_fonte
    from .deteccao import abrir_detector
    from .mapa import Mapa
    from .registro import Registro
    from .sistema import Baliza

    mapa = Mapa.carregar(args.mapa)
    detector = abrir_detector(
        modo="vagas" if args.detector == "vagas" else "veiculos",
        pesos=args.pesos,
        confianca=args.confianca,
        tamanho=args.tamanho,
        dispositivo=args.dispositivo,
    )
    registro = Registro(args.banco) if args.banco else None
    sistema = Baliza(mapa, detector, limiar=args.limiar, janela_suavizacao=args.janela,
                     registro=registro)

    try:
        fonte = abrir_fonte(args.alvo, intervalo_s=args.intervalo, limite=args.limite)
    except FonteIndisponivel as erro:
        print(f"erro: {erro}", file=sys.stderr)
        return 2

    print(f"camera {mapa.camera}, {len(mapa)} vagas, detector {args.detector}"
          f" em {detector.dispositivo}")
    if args.mostrar:
        print("espaco pausa, q encerra")

    resultados = sistema.rodar(fonte, mostrar=args.mostrar, gravar=args.gravar)
    if not resultados:
        print("nenhum quadro lido", file=sys.stderr)
        return 1

    ultimo = resultados[-1]
    media_ms = sum(r.ms_inferencia for r in resultados) / len(resultados)
    print(f"\n{len(resultados)} quadros processados, {media_ms:.0f} ms por quadro em media")
    print(f"ultimo estado: {ultimo.livres} livres, {ultimo.ocupadas} ocupadas,"
          f" {ultimo.sem_leitura} sem leitura")
    for setor, (livres, total) in ultimo.por_setor(mapa.por_id).items():
        print(f"  setor {setor}: {livres} livres de {total}")

    if registro is not None:
        if args.csv:
            linhas = registro.exportar_csv(args.csv, mapa.camera)
            print(f"{linhas} leituras exportadas para {args.csv}")
        registro.fechar()
    return 0


def _comando_mapa_pklot(args) -> int:
    from . import pklot
    from .mapa import dividir_em_setores

    try:
        fotos = pklot.listar_fotos(
            args.raiz, estacionamentos=(args.estacionamento,), limite_por_dia=1
        )
    except FileNotFoundError as erro:
        print(f"erro: {erro}", file=sys.stderr)
        return 2
    if not fotos:
        print(f"erro: nenhuma foto de {args.estacionamento} em {args.raiz}", file=sys.stderr)
        return 2
    foto = fotos[0]
    mapa = pklot.mapa_de_anotacao(foto.anotacao, args.estacionamento, 1280, 720)
    mapa = dividir_em_setores(mapa, args.setores)
    mapa.salvar(args.saida)
    print(f"{len(mapa)} vagas, setores {', '.join(mapa.setores)} -> {args.saida}")
    return 0


def _comando_demonstracao(args) -> int:
    """Roda o pacote de demonstração que acompanha o repositório."""
    from .captura import abrir_fonte
    from .deteccao import abrir_detector
    from .mapa import Mapa
    from .sistema import Baliza

    pasta = Path(args.pasta) if args.pasta else DEMO
    mapas = sorted((pasta / "mapas").glob("*.json"))
    if not mapas:
        print(f"erro: nenhum mapa em {pasta / 'mapas'}", file=sys.stderr)
        return 2

    print("Demonstracao do Baliza\n")
    for indice, caminho in enumerate(mapas, 1):
        print(f"  {indice}. {caminho.stem}")
    escolha = args.escolha
    if escolha is None:
        try:
            escolha = int(input("\nqual? ") or "1")
        except (ValueError, EOFError):
            escolha = 1
    caminho_mapa = mapas[max(1, min(escolha, len(mapas))) - 1]

    mapa = Mapa.carregar(caminho_mapa)
    video = pasta / "patios" / f"{caminho_mapa.stem}.mp4"
    alvo = str(video) if video.exists() else str(pasta / "patios" / caminho_mapa.stem)

    detector = abrir_detector("veiculos", args.pesos, tamanho=args.tamanho)
    sistema = Baliza(mapa, detector)
    resultados = sistema.rodar(abrir_fonte(alvo), mostrar=not args.sem_janela)
    if resultados:
        ultimo = resultados[-1]
        print(f"\n{ultimo.livres} vagas livres de {ultimo.total}")
    return 0


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="baliza",
        description="Ocupacao de vagas de estacionamento a partir de camera fixa",
    )
    sub = parser.add_subparsers(dest="comando")

    rodar = sub.add_parser("rodar", help="processa video, pasta de fotos, imagem ou camera")
    rodar.add_argument("alvo", help="arquivo, pasta, indice de camera ou URL rtsp")
    rodar.add_argument("--mapa", required=True, help="JSON com o mapa de vagas")
    rodar.add_argument("--detector", choices=("veiculos", "vagas"), default="veiculos")
    rodar.add_argument("--pesos", default=None, help="pesos do YOLO")
    rodar.add_argument("--tamanho", type=int, default=1280, help="lado da entrada do modelo")
    rodar.add_argument("--confianca", type=float, default=0.25)
    rodar.add_argument("--limiar", type=float, default=0.30,
                       help="fracao da vaga coberta para chama-la de ocupada")
    rodar.add_argument("--janela", type=int, default=5, help="quadros do voto de maioria")
    rodar.add_argument("--intervalo", type=float, default=0.0,
                       help="segundos entre leituras, 0 le todo quadro")
    rodar.add_argument("--limite", type=int, default=None, help="maximo de quadros")
    rodar.add_argument("--mostrar", action="store_true", help="abre a janela")
    rodar.add_argument("--gravar", default=None, help="salva o video anotado")
    rodar.add_argument("--banco", default=None, help="arquivo SQLite do historico")
    rodar.add_argument("--csv", default=None, help="exporta o historico ao final")
    rodar.add_argument("--dispositivo", default="auto", help="auto, cpu ou cuda:0")
    rodar.set_defaults(funcao=_comando_rodar)

    mapa = sub.add_parser("mapa-do-pklot", help="gera o mapa de vagas a partir do XML do PKLot")
    mapa.add_argument("raiz", help="pasta onde o PKLot foi extraido")
    mapa.add_argument("--estacionamento", default="UFPR04", help="UFPR04, UFPR05 ou PUCPR")
    mapa.add_argument("--setores", type=int, default=3)
    mapa.add_argument("--saida", required=True)
    mapa.set_defaults(funcao=_comando_mapa_pklot)

    demo = sub.add_parser("demonstracao", help="roda o pacote de demonstracao")
    demo.add_argument("--pasta", default=None)
    demo.add_argument("--escolha", type=int, default=None)
    demo.add_argument("--pesos", default=None)
    demo.add_argument("--tamanho", type=int, default=1280)
    demo.add_argument("--sem-janela", action="store_true")
    demo.set_defaults(funcao=_comando_demonstracao)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = construir_parser()
    # Sem argumento nenhum o programa cai na demonstracao, que e o que quem
    # abriu por engano quer ver. Com argumento errado, o argparse reclama.
    args = parser.parse_args(argv if argv else ["demonstracao"])
    return args.funcao(args)


if __name__ == "__main__":  # pragma: sem cobertura
    raise SystemExit(main())
