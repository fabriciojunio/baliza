import pytest

from baliza.geometria import (
    area,
    area_interseccao,
    caixa_para_poligono,
    centro,
    envolvente,
    fracao_coberta,
    iou,
    recortar,
)

QUADRADO = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_area_de_quadrado():
    assert area(QUADRADO) == pytest.approx(100)


def test_area_independe_do_sentido():
    assert area(list(reversed(QUADRADO))) == pytest.approx(100)


def test_area_de_poligono_degenerado_e_zero():
    assert area([(0, 0), (1, 1)]) == 0
    assert area([]) == 0


def test_area_de_triangulo():
    assert area([(0, 0), (4, 0), (0, 3)]) == pytest.approx(6)


def test_recorte_totalmente_dentro_devolve_o_sujeito():
    recortado = recortar([(2, 2), (4, 2), (4, 4), (2, 4)], QUADRADO)
    assert area(recortado) == pytest.approx(4)


def test_recorte_sem_sobreposicao_e_vazio():
    assert area(recortar(QUADRADO, [(50, 50), (60, 50), (60, 60), (50, 60)])) == 0


def test_interseccao_pela_metade():
    assert area_interseccao(QUADRADO, (5, 0, 15, 10)) == pytest.approx(50)


def test_fracao_coberta_vai_de_zero_a_um():
    assert fracao_coberta(QUADRADO, (0, 0, 10, 10)) == pytest.approx(1.0)
    assert fracao_coberta(QUADRADO, (5, 0, 15, 10)) == pytest.approx(0.5)
    assert fracao_coberta(QUADRADO, (20, 20, 30, 30)) == 0.0


def test_caminhao_que_transborda_ainda_cobre_a_vaga_inteira():
    """A razao escolhida não pune a caixa grande demais, e a IoU puniria."""
    caminhao = (-20, -20, 40, 40)
    assert fracao_coberta(QUADRADO, caminhao) == pytest.approx(1.0)
    assert iou(QUADRADO, caminhao) < 0.05


def test_vaga_girada_e_recortada_corretamente():
    losango = [(5, 0), (10, 5), (5, 10), (0, 5)]
    assert area(losango) == pytest.approx(50)
    assert fracao_coberta(losango, (0, 0, 5, 10)) == pytest.approx(0.5)


def test_fracao_de_poligono_sem_area_nao_estoura():
    assert fracao_coberta([(0, 0), (1, 0), (2, 0)], (0, 0, 10, 10)) == 0.0


def test_iou_de_caixas_identicas_e_um():
    assert iou(QUADRADO, (0, 0, 10, 10)) == pytest.approx(1.0)


def test_envolvente_e_centro():
    losango = [(5, 0), (10, 5), (5, 10), (0, 5)]
    assert envolvente(losango) == (0, 0, 10, 10)
    assert centro(losango) == pytest.approx((5, 5))


def test_caixa_para_poligono_tem_quatro_cantos():
    assert caixa_para_poligono((1, 2, 3, 4)) == [(1, 2), (3, 2), (3, 4), (1, 4)]
