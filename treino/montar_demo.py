"""Monta o pacote de demonstração que vai com o repositório.

O PKLot fotografa o mesmo pátio de cinco em cinco minutos ao longo do dia
inteiro. Juntando as fotos de um dia em sequência sai um vídeo em que o
estacionamento enche de manhã e esvazia à tarde, que é exatamente o que se
quer mostrar funcionando.

    python treino/montar_demo.py --parte videos     # so os videos crus
    python treino/montar_demo.py --parte anotado --pesos runs/vagas/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

from baliza import pklot
from baliza.mapa import dividir_em_setores
from baliza.tipos import Estado

DESTINO = Path("demo")

# Um dia representativo de cada câmera. Escolhidos por terem o dia inteiro
# fotografado e por mostrarem o pátio enchendo e esvaziando.
DIAS = {
    "ufpr04": ("UFPR04", "Sunny"),
    "ufpr05": ("UFPR05", "Cloudy"),
    "pucpr": ("PUCPR", "Sunny"),
}


def _variacao(fotos: list[pklot.Foto], amostras: int = 12) -> float:
    """Diferenca entre a maior e a menor ocupacao do dia, de 0 a 1."""
    passo = max(1, len(fotos) // amostras)
    taxas = []
    for foto in fotos[::passo]:
        try:
            verdadeiro = pklot.verdade(foto.anotacao)
        except pklot.AnotacaoInvalida:
            continue
        if verdadeiro:
            ocupadas = sum(1 for e in verdadeiro.values() if e is Estado.OCUPADA)
            taxas.append(ocupadas / len(verdadeiro))
    return max(taxas) - min(taxas) if len(taxas) > 1 else 0.0


def escolher_dia(raiz: str, estacionamento: str, clima: str) -> list[pklot.Foto]:
    """O dia em que o patio mais encheu e esvaziou.

    A primeira versao escolhia o dia com mais fotos, e caiu num 23 de dezembro
    com o estacionamento vazio do comeco ao fim: 156 quadros em que nada
    acontece. O que rende demonstracao e variacao de ocupacao, nao duracao.
    """
    fotos = pklot.listar_fotos(raiz, estacionamentos=(estacionamento,), climas=(clima,))
    por_dia: dict[str, list[pklot.Foto]] = {}
    for foto in fotos:
        por_dia.setdefault(foto.dia, []).append(foto)
    if not por_dia:
        return []

    candidatos = [d for d in por_dia.values() if len(d) >= 60]
    if not candidatos:
        candidatos = list(por_dia.values())
    melhor = max(candidatos, key=_variacao)
    return sorted(melhor, key=lambda f: f.nome)


def montar_video(fotos: list[pklot.Foto], saida: Path, fps: int = 8) -> int:
    if not fotos:
        return 0
    primeiro = cv2.imread(str(fotos[0].imagem))
    altura, largura = primeiro.shape[:2]
    saida.parent.mkdir(parents=True, exist_ok=True)
    escritor = cv2.VideoWriter(
        str(saida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura)
    )
    escritas = 0
    for foto in fotos:
        quadro = cv2.imread(str(foto.imagem))
        if quadro is None:
            continue
        if quadro.shape[:2] != (altura, largura):
            quadro = cv2.resize(quadro, (largura, altura))
        carimbo = foto.nome.replace("_", " ")
        cv2.putText(quadro, carimbo, (12, altura - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        escritor.write(quadro)
        escritas += 1
    escritor.release()
    return escritas


def parte_videos(raiz: str) -> dict:
    relatorio = {}
    for apelido, (estacionamento, clima) in DIAS.items():
        fotos = escolher_dia(raiz, estacionamento, clima)
        if not fotos:
            print(f"{apelido}: nenhuma foto, pulando")
            continue

        # Mapa de vagas: sai do XML da primeira foto do dia.
        mapa = None
        for foto in fotos:
            try:
                mapa = pklot.mapa_de_anotacao(foto.anotacao, estacionamento, 1280, 720)
                break
            except pklot.AnotacaoInvalida:
                continue
        if mapa is None:
            print(f"{apelido}: nenhum XML utilizavel, pulando")
            continue
        mapa = dividir_em_setores(mapa, 3)
        mapa.salvar(DESTINO / "mapas" / f"{apelido}.json")

        quadros = montar_video(fotos, DESTINO / "patios" / f"{apelido}.mp4")

        # Três fotos soltas do mesmo dia, para quem quiser testar imagem avulsa.
        pasta_fotos = DESTINO / "fotos"
        pasta_fotos.mkdir(parents=True, exist_ok=True)
        for indice in (0, len(fotos) // 2, len(fotos) - 1):
            origem = fotos[indice].imagem
            shutil.copy2(origem, pasta_fotos / f"{apelido}_{origem.stem}.jpg")

        relatorio[apelido] = {
            "estacionamento": estacionamento,
            "clima": clima,
            "dia": fotos[0].dia,
            "vagas": len(mapa),
            "setores": mapa.setores,
            "quadros_no_video": quadros,
            "primeira_foto": fotos[0].nome,
            "ultima_foto": fotos[-1].nome,
        }
        print(f"{apelido}: {quadros} quadros, {len(mapa)} vagas, dia {fotos[0].dia}")
    return relatorio


def parte_anotado(pesos: str | None, modo: str, tamanho: int, janelas,
                  limite: int | None, sufixo: str = "", apenas: str | None = None) -> dict:
    """Renderiza o vídeo já com as vagas pintadas, que é o que vai na aula."""
    from baliza.captura import abrir_fonte
    from baliza.deteccao import abrir_detector
    from baliza.mapa import Mapa
    from baliza.registro import Registro
    from baliza.sistema import Baliza

    relatorio = {}
    for caminho_mapa in sorted((DESTINO / "mapas").glob("*.json")):
        apelido = caminho_mapa.stem
        if apenas and apelido != apenas:
            continue
        video = DESTINO / "patios" / f"{apelido}.mp4"
        if not video.exists():
            continue
        mapa = Mapa.carregar(caminho_mapa)
        detector = abrir_detector(modo, pesos, tamanho=tamanho, janelas=janelas)
        banco = DESTINO / "historico" / f"{apelido}{sufixo}.db"
        banco.parent.mkdir(parents=True, exist_ok=True)
        if banco.exists():
            banco.unlink()
        registro = Registro(banco)
        sistema = Baliza(mapa, detector, registro=registro)

        saida = DESTINO / "anotado" / f"{apelido}{sufixo}.mp4"
        saida.parent.mkdir(parents=True, exist_ok=True)
        fonte = abrir_fonte(str(video), limite=limite)
        resultados = sistema.rodar(fonte, gravar=saida)
        registro.exportar_csv(DESTINO / "historico" / f"{apelido}{sufixo}.csv", mapa.camera)
        registro.fechar()

        if resultados:
            ocupacao = [r.ocupadas for r in resultados]
            relatorio[apelido + sufixo] = {
                "quadros": len(resultados),
                "vagas": len(mapa),
                "ms_por_quadro": round(
                    sum(r.ms_inferencia for r in resultados) / len(resultados), 1
                ),
                "ocupadas_min": min(ocupacao),
                "ocupadas_max": max(ocupacao),
                "livres_no_fim": resultados[-1].livres,
            }
            print(f"{apelido}: {len(resultados)} quadros anotados -> {saida}")
    return relatorio


def main() -> int:
    parser = argparse.ArgumentParser(description="monta o pacote de demonstracao")
    parser.add_argument("--raiz", default="dados/PKLot")
    parser.add_argument("--parte", default="videos", choices=("videos", "anotado", "tudo"))
    parser.add_argument("--pesos", default=None)
    parser.add_argument("--modo", default="veiculos", choices=("veiculos", "vagas"))
    parser.add_argument("--tamanho", type=int, default=1280)
    parser.add_argument("--janelas", default=None)
    parser.add_argument("--limite", type=int, default=None)
    parser.add_argument("--sufixo", default="", help="sufixo do arquivo de saida")
    parser.add_argument("--apenas", default=None, help="so um patio, pelo apelido")
    args = parser.parse_args()

    grade = None
    if args.janelas:
        c, l = args.janelas.lower().split("x")
        grade = (int(c), int(l))

    relatorio = {}
    if args.parte in ("videos", "tudo"):
        relatorio["patios"] = parte_videos(args.raiz)
    if args.parte in ("anotado", "tudo"):
        novo = parte_anotado(args.pesos, args.modo, args.tamanho, grade,
                             args.limite, args.sufixo, args.apenas)
        relatorio.setdefault("anotado", {}).update(novo)

    caminho = DESTINO / "relatorio.json"
    anterior = {}
    if caminho.exists():
        anterior = json.loads(caminho.read_text(encoding="utf-8"))
    # Mescla por dentro: rodar so a parte anotada nao pode apagar o que a
    # parte de videos escreveu, nem uma variante de detector apagar a outra.
    for chave, valor in relatorio.items():
        if isinstance(valor, dict) and isinstance(anterior.get(chave), dict):
            anterior[chave].update(valor)
        else:
            anterior[chave] = valor
    caminho.write_text(json.dumps(anterior, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{caminho}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
