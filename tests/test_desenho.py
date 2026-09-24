import numpy as np

from baliza.desenho import _sem_acento, anotar, legenda
from baliza.tipos import Estado, Leitura, Resultado


def resultado_misto():
    return Resultado(leituras=[
        Leitura("1", Estado.LIVRE),
        Leitura("2", Estado.OCUPADA),
        Leitura("3", Estado.SEM_LEITURA),
        Leitura("4", Estado.LIVRE),
    ], ms_inferencia=12.3)


def test_acento_sai_do_texto_que_vai_para_a_imagem():
    assert _sem_acento("caminhão às 5h") == "caminhao as 5h"


def test_anotar_nao_mexe_no_quadro_original(quadro, mapa_de_quatro):
    copia = quadro.copy()
    anotar(quadro, mapa_de_quatro, resultado_misto())
    assert np.array_equal(quadro, copia)


def test_anotar_preserva_o_tamanho(quadro, mapa_de_quatro):
    saida = anotar(quadro, mapa_de_quatro, resultado_misto())
    assert saida.shape == quadro.shape


def test_anotar_realmente_pinta_alguma_coisa(quadro, mapa_de_quatro):
    saida = anotar(quadro, mapa_de_quatro, resultado_misto())
    assert not np.array_equal(saida, quadro)


def test_vaga_que_nao_esta_no_mapa_e_ignorada(quadro, mapa_de_quatro):
    r = Resultado(leituras=[Leitura("99", Estado.LIVRE)])
    assert anotar(quadro, mapa_de_quatro, r).shape == quadro.shape


def test_legenda_escurece_o_rodape(quadro, mapa_de_quatro):
    saida = legenda(anotar(quadro, mapa_de_quatro, resultado_misto()))
    assert saida[-5, 5].tolist() == [25, 25, 25]


def test_desenho_sem_preenchimento_tambem_funciona(quadro, mapa_de_quatro):
    saida = anotar(quadro, mapa_de_quatro, resultado_misto(), preencher=False)
    assert saida.shape == quadro.shape
