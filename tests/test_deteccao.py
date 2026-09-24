import numpy as np
import pytest

from baliza.deteccao import (
    CLASSES_VEICULO_COCO,
    DetectorEmJanelas,
    DetectorVagas,
    DetectorVeiculos,
    abrir_detector,
    nms,
)
from baliza.tipos import Deteccao


def test_abrir_detector_escolhe_a_classe_certa():
    assert isinstance(abrir_detector("veiculos", "modelos/yolo11n.pt"), DetectorVeiculos)
    assert isinstance(abrir_detector("vagas", "qualquer.pt"), DetectorVagas)
    assert isinstance(
        abrir_detector("veiculos", "modelos/yolo11n.pt", janelas=(2, 2)), DetectorEmJanelas
    )


def test_modo_vagas_sem_pesos_e_erro():
    with pytest.raises(ValueError, match="pesos treinados"):
        abrir_detector("vagas")


def test_modo_desconhecido():
    with pytest.raises(ValueError, match="desconhecido"):
        abrir_detector("outro")


def test_classes_de_veiculo_do_coco():
    assert CLASSES_VEICULO_COCO[2] == "carro"
    assert set(CLASSES_VEICULO_COCO) == {2, 3, 5, 7}


def test_nms_junta_as_duas_caixas_do_mesmo_carro():
    deteccoes = [
        Deteccao((0, 0, 10, 10), "carro", 0.9),
        Deteccao((1, 1, 11, 11), "carro", 0.7),
    ]
    assert len(nms(deteccoes)) == 1
    assert nms(deteccoes)[0].confianca == 0.9


def test_nms_preserva_carros_separados():
    deteccoes = [
        Deteccao((0, 0, 10, 10), "carro", 0.9),
        Deteccao((50, 50, 60, 60), "carro", 0.8),
    ]
    assert len(nms(deteccoes)) == 2


def test_nms_nao_mistura_classes():
    deteccoes = [
        Deteccao((0, 0, 10, 10), "carro", 0.9),
        Deteccao((0, 0, 10, 10), "moto", 0.8),
    ]
    assert len(nms(deteccoes)) == 2


def test_grade_de_janelas_cobre_o_quadro_inteiro():
    detector = DetectorEmJanelas(colunas=3, linhas=2, sobreposicao=0.0)
    janelas = detector.janelas(1200, 600)
    assert len(janelas) == 6
    assert janelas[0] == (0, 0, 400, 300)
    assert janelas[-1] == (800, 300, 1200, 600)


def test_sobreposicao_alarga_as_janelas():
    detector = DetectorEmJanelas(colunas=2, linhas=1, sobreposicao=0.2)
    primeira, segunda = detector.janelas(1000, 400)
    assert primeira[2] > 500  # invade a metade da direita
    assert segunda[0] < 500   # e vice-versa


def test_grade_invalida():
    with pytest.raises(ValueError):
        DetectorEmJanelas(colunas=0)
    with pytest.raises(ValueError):
        DetectorEmJanelas(sobreposicao=0.95)


@pytest.mark.lento
def test_modelo_de_verdade_acha_carro_no_patio():
    """Roda o YOLO mesmo, numa foto do pacote de demonstracao."""
    import cv2
    from pathlib import Path

    foto = next(Path("demo/fotos").glob("ufpr04_*.jpg"), None)
    if foto is None:
        pytest.skip("pacote de demonstracao nao montado")
    detector = DetectorVeiculos("modelos/yolo11n.pt", tamanho=1280)
    deteccoes = detector.detectar(cv2.imread(str(foto)))
    assert len(deteccoes) > 5
    assert all(d.classe in CLASSES_VEICULO_COCO.values() for d in deteccoes)
