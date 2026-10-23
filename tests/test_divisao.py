"""A mesma câmera/dia não pode participar de duas partições de avaliação."""

import importlib.util
from pathlib import Path

import pytest

from baliza.pklot import Foto


def carregar_divisao():
    caminho = Path(__file__).resolve().parents[1] / "treino" / "divisao.py"
    spec = importlib.util.spec_from_file_location("divisao_auditoria", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def fotos_de_exemplo():
    return [
        Foto(Path(f"{camera}/{dia}/{clima}/{i}.jpg"), Path(f"{i}.xml"), camera, clima,
             f"2013-01-{dia:02d}")
        for camera in ("PUCPR", "UFPR04", "UFPR05")
        for dia in (2, 3, 5, 7, 9)
        for clima in ("Sunny", "Rainy")
        for i in range(4)
    ]


def test_divisao_completa_nao_mistura_dias_entre_conjuntos(monkeypatch):
    divisao = carregar_divisao()
    fotos = fotos_de_exemplo()
    monkeypatch.setattr(divisao, "listar_fotos", lambda *a, **kw: fotos)
    partes = {nome: divisao.conjunto_completo("base", nome)
              for nome in ("treino", "validacao", "teste")}
    grupos = {nome: {(f.estacionamento, f.dia) for f in parte}
              for nome, parte in partes.items()}
    assert grupos["treino"].isdisjoint(grupos["validacao"])
    assert grupos["treino"].isdisjoint(grupos["teste"])
    assert grupos["validacao"].isdisjoint(grupos["teste"])
    assert set().union(*(set(parte) for parte in partes.values())) == set(fotos)
    for nome in ("validacao", "teste"):
        assert {f.estacionamento for f in partes[nome]} == set(divisao.TODOS)


def test_particao_do_dia_independe_da_ordem_e_quantidade_de_fotos(monkeypatch):
    divisao = carregar_divisao()
    fotos = fotos_de_exemplo()
    monkeypatch.setattr(divisao, "listar_fotos", lambda *a, **kw: fotos)
    antes = {(f.estacionamento, f.dia) for f in divisao.conjunto_completo("base", "teste")}
    fotos[:] = list(reversed(fotos[::4]))
    depois = {(f.estacionamento, f.dia) for f in divisao.conjunto_completo("base", "teste")}
    assert antes == depois


def test_erro_de_nome_conhecido_do_pklot_preserva_dia_real():
    divisao = carregar_divisao()
    foto = Foto(Path("2013-04-16_06_30_00.jpg"), Path("foto.xml"),
                "UFPR05", "Sunny", "2013-14-16")
    assert divisao.dia_par(foto)
    assert divisao.data_da_foto(foto).isoformat() == "2013-04-16"


@pytest.mark.parametrize("dia,nome", [
    ("data-invalida", "foto.jpg"), ("2013-14-16", "2013-04-17_06_30_00.jpg"),
])
def test_data_invalida_nao_e_alocada_silenciosamente_no_treino(dia, nome):
    divisao = carregar_divisao()
    with pytest.raises(ValueError):
        divisao.dia_par(Foto(Path(nome), Path("foto.xml"), "UFPR05", "Sunny", dia))
