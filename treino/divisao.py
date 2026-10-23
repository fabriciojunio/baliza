"""A divisão da base, que é a decisão mais importante do projeto.

O PKLot fotografa o mesmo pátio de cinco em cinco minutos. Duas fotos
seguidas do mesmo dia são quase a mesma foto. Se a divisão for aleatória, a
foto das 10h05 cai no treino e a das 10h10 no teste, o modelo "acerta" quase
tudo e o número não descreve nada.

Por isso a divisão é por câmera e por dia:

    treino     PUCPR e UFPR04, dias pares
    validacao  PUCPR e UFPR04, dias ímpares
    teste      UFPR05 inteiro, uma câmera que o modelo nunca viu

Nenhuma foto do mesmo dia e da mesma câmera aparece em dois conjuntos.
"""

from __future__ import annotations

from datetime import date

from baliza.pklot import Foto, listar_fotos

TREINO = ("PUCPR", "UFPR04")
TESTE = ("UFPR05",)


def data_da_foto(foto: Foto) -> date:
    # A distribuição original contém UFPR05/Sunny/2013-14-16, mas as
    # fotografias nessa pasta se chamam 2013-04-16_*. Não inventar um dia para
    # qualquer nome inválido: a exceção conhecida exige o nome da foto.
    if foto.dia == "2013-14-16" and foto.nome.startswith("2013-04-16_"):
        return date(2013, 4, 16)
    return date.fromisoformat(foto.dia)


def dia_par(foto: Foto) -> bool:
    return data_da_foto(foto).day % 2 == 0


def conjunto(
    raiz: str,
    nome: str,
    limite_por_dia: int | None = 4,
) -> list[Foto]:
    if nome == "treino":
        fotos = listar_fotos(raiz, estacionamentos=TREINO, limite_por_dia=limite_por_dia)
        return [f for f in fotos if dia_par(f)]
    if nome == "validacao":
        fotos = listar_fotos(raiz, estacionamentos=TREINO, limite_por_dia=limite_por_dia)
        return [f for f in fotos if not dia_par(f)]
    if nome == "teste":
        return listar_fotos(raiz, estacionamentos=TESTE, limite_por_dia=limite_por_dia)
    raise ValueError(f"conjunto desconhecido: {nome}")


# O modelo que vai para a demonstração precisa conhecer as três câmeras, e
# para ele a divisão é outra: todas entram, separadas só por dia. O conjunto
# acima continua existindo porque é ele que responde a pergunta científica,
# "quanto o modelo perde numa câmera que nunca viu".
TODOS = TREINO + TESTE


def conjunto_completo(raiz: str, nome: str, limite_por_dia: int | None = 4) -> list[Foto]:
    fotos = listar_fotos(raiz, estacionamentos=TODOS, limite_por_dia=limite_por_dia)
    if nome == "treino":
        return [f for f in fotos if dia_par(f)]
    if nome in ("validacao", "teste"):
        impares = [f for f in fotos if not dia_par(f)]
        # A regra depende do dia, nunca do índice da foto: alternar fotos do
        # mesmo dia colocava quadros quase idênticos na validação e no teste.
        # Dias 1, 5, 9... validam; 3, 7, 11... ficam para o teste. A atribuição
        # é estável quando muda a ordem, o clima ou o limite de fotos por dia.
        resto = 1 if nome == "validacao" else 3
        return [f for f in impares if data_da_foto(f).day % 4 == resto]
    raise ValueError(f"conjunto desconhecido: {nome}")


def resumo(raiz: str, limite_por_dia: int | None = 4) -> dict[str, dict]:
    saida = {}
    for nome in ("treino", "validacao", "teste"):
        fotos = conjunto(raiz, nome, limite_por_dia)
        climas: dict[str, int] = {}
        estacionamentos: dict[str, int] = {}
        for foto in fotos:
            climas[foto.clima] = climas.get(foto.clima, 0) + 1
            estacionamentos[foto.estacionamento] = (
                estacionamentos.get(foto.estacionamento, 0) + 1
            )
        saida[nome] = {
            "fotos": len(fotos),
            "dias": len({(f.estacionamento, f.dia) for f in fotos}),
            "climas": climas,
            "estacionamentos": estacionamentos,
        }
    return saida


if __name__ == "__main__":
    import json
    import sys

    raiz = sys.argv[1] if len(sys.argv) > 1 else "dados/PKLot"
    limite = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    print(json.dumps(resumo(raiz, limite), indent=2, ensure_ascii=False))
