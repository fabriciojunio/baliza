import json

import pytest

from baliza.mapa import Mapa, MapaInvalido, dividir_em_setores
from baliza.tipos import Vaga

from conftest import vaga_retangular


def test_mapa_sem_vaga_e_rejeitado():
    with pytest.raises(MapaInvalido):
        Mapa("x", [])


def test_vaga_repetida_e_rejeitada():
    with pytest.raises(MapaInvalido, match="repetida"):
        Mapa("x", [vaga_retangular("1", 0, 0), vaga_retangular("1", 20, 0)])


def test_contorno_de_dois_pontos_e_rejeitado():
    with pytest.raises(MapaInvalido, match="tres pontos"):
        Mapa("x", [Vaga("1", [(0, 0), (1, 1)])])


def test_ida_e_volta_pelo_disco(tmp_path, mapa_de_quatro):
    caminho = tmp_path / "mapa.json"
    mapa_de_quatro.salvar(caminho)
    lido = Mapa.carregar(caminho)
    assert len(lido) == 4
    assert lido.camera == "teste"
    assert lido.setores == ["A", "B"]
    assert lido.por_id["3"].setor == "B"


def test_arquivo_que_nao_existe(tmp_path):
    with pytest.raises(FileNotFoundError):
        Mapa.carregar(tmp_path / "nao_existe.json")


def test_json_quebrado(tmp_path):
    caminho = tmp_path / "ruim.json"
    caminho.write_text("{isso nao e json", encoding="utf-8")
    with pytest.raises(MapaInvalido, match="JSON"):
        Mapa.carregar(caminho)


def test_json_sem_a_chave_vagas(tmp_path):
    caminho = tmp_path / "ruim.json"
    caminho.write_text(json.dumps({"camera": "x"}), encoding="utf-8")
    with pytest.raises(MapaInvalido, match="malformado"):
        Mapa.carregar(caminho)


def test_divisao_em_setores_separa_pela_posicao():
    vagas = [vaga_retangular(str(i), i * 20, 0) for i in range(6)]
    mapa = dividir_em_setores(Mapa("x", vagas), colunas=3)
    setores = [v.setor for v in sorted(mapa.vagas, key=lambda v: v.centro[0])]
    assert setores == ["A", "A", "B", "B", "C", "C"]


def test_divisao_com_uma_coluna_deixa_tudo_no_mesmo_setor(mapa_de_quatro):
    assert dividir_em_setores(mapa_de_quatro, 1).setores == ["A"]


def test_divisao_com_zero_colunas_e_erro(mapa_de_quatro):
    with pytest.raises(ValueError):
        dividir_em_setores(mapa_de_quatro, 0)


def test_iterar_o_mapa_devolve_as_vagas(mapa_de_quatro):
    assert [v.id for v in mapa_de_quatro] == ["1", "2", "3", "4"]
