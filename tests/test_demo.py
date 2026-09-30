"""O pacote de demonstração precisa ser honesto e precisa estar inteiro.

São duas perguntas diferentes, e as duas já foram respondidas errado aqui:

- o dia de cada vídeo não pode ser dia de treino, senão a demonstração mede o
  que o modelo decorou e não o que ele acerta;
- o mapa de cada câmera tem que apontar para um arquivo de pesos que existe em
  disco, senão o programa cai no detector geral em plena aula.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
DEMO = RAIZ / "demo"
sys.path.insert(0, str(RAIZ / "treino"))

APELIDOS = ("ufpr04", "ufpr05", "pucpr")


def test_todo_mapa_aponta_para_pesos_que_existem():
    from baliza.mapa import Mapa

    for apelido in APELIDOS:
        caminho = DEMO / "mapas" / f"{apelido}.json"
        if not caminho.exists():
            pytest.skip("pacote de demonstracao nao montado")
        mapa = Mapa.carregar(caminho)
        if mapa.detector != "vagas":
            continue
        assert mapa.pesos, f"{apelido}: detector treinado sem pesos gravados"
        assert (RAIZ / mapa.pesos).exists(), f"{apelido}: {mapa.pesos} nao esta em disco"


def test_todo_patio_tem_video_mapa_fotos_e_anotado():
    relatorio = DEMO / "relatorio.json"
    if not relatorio.exists():
        pytest.skip("pacote de demonstracao nao montado")
    dados = json.loads(relatorio.read_text(encoding="utf-8"))
    for apelido in APELIDOS:
        assert (DEMO / "patios" / f"{apelido}.mp4").exists(), apelido
        assert (DEMO / "mapas" / f"{apelido}.json").exists(), apelido
        assert list((DEMO / "fotos" / apelido).glob("*.jpg")), apelido
        assert (DEMO / "anotado" / f"{apelido}.mp4").exists(), apelido
        assert apelido in dados["patios"], apelido


def test_todo_historico_gravado_diz_com_que_detector_foi_feito():
    """O painel escolhe a curva por esse campo, e sem ele escolhe pelo nome."""
    relatorio = DEMO / "relatorio.json"
    if not relatorio.exists():
        pytest.skip("pacote de demonstracao nao montado")
    anotado = json.loads(relatorio.read_text(encoding="utf-8")).get("anotado", {})
    assert anotado, "nenhum video anotado no relatorio"
    for chave, dados in anotado.items():
        assert dados.get("detector") in ("vagas", "veiculos"), chave
        if dados["detector"] == "vagas":
            assert dados.get("pesos"), f"{chave}: detector treinado sem pesos"


def test_nenhum_dia_da_demonstracao_e_dia_de_treino():
    """O erro que motivou este arquivo.

    Dois dos três vídeos saíam de dia par, que é treino em `divisao.py`. A
    acurácia da demonstração media memória, e ninguém tinha como notar olhando
    o número, porque memória dá número melhor.
    """
    relatorio = DEMO / "relatorio.json"
    if not relatorio.exists():
        pytest.skip("pacote de demonstracao nao montado")
    patios = json.loads(relatorio.read_text(encoding="utf-8"))["patios"]
    for apelido, dados in patios.items():
        assert "dia_no_treino" in dados, f"{apelido}: relatorio antigo, remonte a demo"
        assert dados["dia_no_treino"] is False, (
            f"{apelido}: o video sai de {dados['dia']}, que e dia de treino"
        )


@pytest.mark.lento
def test_o_dia_escolhido_hoje_continua_fora_do_treino():
    """Refaz a escolha a partir da base, e não do relatório já gravado."""
    import divisao
    import montar_demo

    base = RAIZ / "dados" / "PKLot"
    if not base.exists():
        pytest.skip("base PKLot nao esta em disco")
    for apelido, (estacionamento, clima) in montar_demo.DIAS.items():
        fotos = montar_demo.escolher_dia(str(base), estacionamento, clima)
        assert fotos, f"{apelido}: nenhuma foto de dia impar"
        assert not divisao.dia_par(fotos[0]), (
            f"{apelido}: escolheu {fotos[0].dia}, que e dia de treino"
        )
