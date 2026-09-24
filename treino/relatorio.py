"""Roda a comparação inteira e escreve docs/RESULTADOS.md.

    python treino/relatorio.py --amostras 60

Três configurações, dois conjuntos, quebra por estacionamento e por clima.
É daqui que saem os números da documentação; nenhum deles é digitado à mão.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import avaliar

CONFIGURACOES = [
    {
        "nome": "Veículos, quadro inteiro",
        "modo": "veiculos",
        "pesos": "modelos/yolo11n.pt",
        "janelas": None,
        "detalhe": "YOLO11n do COCO, sem treino, imagem inteira",
    },
    {
        "nome": "Veículos, janelas 3x2",
        "modo": "veiculos",
        "pesos": "modelos/yolo11n.pt",
        "janelas": (3, 2),
        "detalhe": "YOLO11n do COCO, sem treino, quadro recortado em 6 janelas",
    },
    {
        "nome": "Vagas, treinado no PKLot",
        "modo": "vagas",
        "pesos": "modelos/vagas.pt",
        "janelas": None,
        "detalhe": "YOLO11n ajustado em 971 imagens, 2 classes",
    },
]


def porcento(valor) -> str:
    return "-" if valor is None else f"{valor * 100:.1f}%"


def tabela(titulo: str, linhas: list[tuple[str, dict]]) -> str:
    saida = [
        f"### {titulo}",
        "",
        "| Configuração | Acurácia (lidas) | Acurácia (todas) | Falso livre |"
        " Falso ocupado | Sem leitura | ms/quadro |",
        "|---|---|---|---|---|---|---|",
    ]
    for nome, dados in linhas:
        g = dados["geral"]
        saida.append(
            f"| {nome} | {porcento(g['acuracia'])} | {porcento(g['acuracia_sobre_todas'])} |"
            f" {porcento(g['falso_livre'])} | {porcento(g['falso_ocupado'])} |"
            f" {porcento(g['taxa_sem_leitura'])} | {dados['ms_por_quadro']:.0f} |"
        )
    saida.append("")
    return "\n".join(saida)


def tabela_grupos(titulo: str, linhas: list[tuple[str, dict]], chaves: list[str]) -> str:
    saida = [f"### {titulo}", "", "| Configuração | " + " | ".join(chaves) + " |",
             "|---" * (len(chaves) + 1) + "|"]
    for nome, dados in linhas:
        celulas = []
        for chave in chaves:
            grupo = dados.get("grupos", {}).get(chave)
            celulas.append(porcento(grupo["acuracia_sobre_todas"]) if grupo else "-")
        saida.append(f"| {nome} | " + " | ".join(celulas) + " |")
    saida.append("")
    return "\n".join(saida)


def main() -> int:
    parser = argparse.ArgumentParser(description="roda a comparacao completa")
    parser.add_argument("--raiz", default="dados/PKLot")
    parser.add_argument("--amostras", type=int, default=60)
    parser.add_argument("--tamanho", type=int, default=1280)
    parser.add_argument("--limiar", type=float, default=0.30)
    parser.add_argument("--saida-json", default="resultados/relatorio.json")
    parser.add_argument("--saida-md", default="docs/RESULTADOS.md")
    args = parser.parse_args()

    bruto: dict[str, dict] = {}
    for configuracao in CONFIGURACOES:
        if not Path(configuracao["pesos"]).exists():
            print(f"pulando {configuracao['nome']}: {configuracao['pesos']} nao existe")
            continue
        for conjunto in ("validacao", "teste"):
            chave = f"{configuracao['nome']} | {conjunto}"
            print(f"\n=== {chave} ===", flush=True)
            bruto[chave] = {
                **avaliar.avaliar(
                    args.raiz, conjunto, args.amostras, configuracao["modo"],
                    configuracao["pesos"], args.tamanho, 0.25, args.limiar,
                    janelas=configuracao["janelas"],
                ),
                "configuracao": configuracao["nome"],
                "detalhe": configuracao["detalhe"],
            }

    Path(args.saida_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida_json).write_text(
        json.dumps(bruto, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    validacao = [(c["nome"], bruto[f"{c['nome']} | validacao"])
                 for c in CONFIGURACOES if f"{c['nome']} | validacao" in bruto]
    teste = [(c["nome"], bruto[f"{c['nome']} | teste"])
             for c in CONFIGURACOES if f"{c['nome']} | teste" in bruto]

    texto = [
        "# Resultados medidos",
        "",
        f"Gerado por `treino/relatorio.py` em {date.today().strftime('%d/%m/%Y')},"
        f" com {args.amostras} fotos por conjunto e entrada de {args.tamanho} px.",
        "",
        "A divisão é por câmera e por dia (ver `treino/divisao.py`). O conjunto de",
        "**teste é a UFPR05 inteira**, uma câmera que nenhum modelo viu em treino.",
        "",
        "Os dois erros aparecem separados porque não custam o mesmo: dizer *livre*",
        "numa vaga ocupada manda o motorista ao lugar errado, dizer *ocupada* numa",
        "vaga livre apenas esconde uma vaga que existia.",
        "",
        "**Acurácia (lidas)** ignora a vaga que o modelo não leu; **acurácia (todas)**",
        "conta a vaga não lida como erro. A distância entre as duas colunas é onde",
        "mora a história deste projeto.",
        "",
        tabela("Validação (PUCPR e UFPR04, dias ímpares)", validacao),
        tabela("Teste (UFPR05, câmera nunca vista)", teste),
        tabela_grupos("Acurácia por estacionamento, na validação", validacao,
                      ["PUCPR", "UFPR04"]),
        tabela_grupos("Acurácia por clima, no teste", teste,
                      ["Sunny", "Cloudy", "Rainy"]),
        "## Como reproduzir",
        "",
        "```",
        "python treino/exportar_yolo.py --por-dia 25",
        "python treino/treinar.py --epocas 30 --tamanho 1280 --lote 4",
        f"python treino/relatorio.py --amostras {args.amostras}",
        "```",
        "",
    ]
    Path(args.saida_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida_md).write_text("\n".join(texto), encoding="utf-8")
    print(f"\n{args.saida_md}\n{args.saida_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
