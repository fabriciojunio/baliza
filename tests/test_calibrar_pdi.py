"""A avaliação offline deve usar as mesmas unidades declaradas no JSON."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from baliza.pdi import Calibracao, Medidas


def carregar_script(nome):
    caminho = Path(__file__).resolve().parents[1] / "treino" / f"{nome}.py"
    spec = importlib.util.spec_from_file_location(nome, caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_global_pontua_em_offline_igual_ao_runtime():
    script = carregar_script("comparar_pdi_e_yolo")
    cal = Calibracao(medias=np.arange(5.0), desvios=np.arange(1.0, 6.0),
                     pesos=np.array([1.0, -1.0, 1.0, 0.0, 1.0]))
    bloco = np.arange(20.0).reshape(4, 5)
    esperado = [cal.pontuar(Medidas(*linha)) for linha in bloco]
    assert np.allclose(script.pontuar_blocos(cal, [bloco])[0], esperado)


@pytest.mark.parametrize("normalizacao", ["camera", "imagem"])
def test_normalizacao_offline_repete_desvio_amostral_da_calibracao(normalizacao):
    script = carregar_script("comparar_pdi_e_yolo")
    cal = Calibracao(normalizacao=normalizacao)
    blocos = [np.arange(10.0).reshape(2, 5), np.arange(15.0, 30.0).reshape(3, 5)]
    pontos = script.pontuar_blocos(cal, blocos)
    for bloco, pontuacao in zip(blocos, pontos):
        base = np.vstack(blocos) if normalizacao == "camera" else bloco
        esperado = (bloco[:, 0] - base[:, 0].mean()) / base[:, 0].std(ddof=1)
        assert np.allclose(pontuacao, esperado)


def test_sem_vagas_validas_nao_produz_comparacao():
    script = carregar_script("comparar_pdi_e_yolo")
    with pytest.raises(ValueError, match="nenhuma imagem"):
        script.pontuar_blocos(Calibracao(), [])


def test_busca_por_f1_publica_acuracia_real(monkeypatch):
    script = carregar_script("calibrar_pdi")
    monkeypatch.setattr(script, "_melhor_limiar", lambda *a: (0.0, 0.123))
    treino = np.arange(25.0).reshape(5, 5)
    y = np.array([0, 0, 1, 1, 1])
    cal = script.calibrar(treino, y)
    pred = [cal.ocupada(Medidas(*linha)) for linha in treino]
    assert cal.acuracia_de_treino == pytest.approx((np.array(pred) == y).mean())
    assert cal.acuracia_de_treino != 0.123
