import pytest

from baliza.suavizacao import Suavizador
from baliza.tipos import Estado, Leitura, Resultado


def res(estado):
    return Resultado(leituras=[Leitura("1", estado)])


def estados(suavizador, sequencia):
    return [suavizador.aplicar(res(e)).leituras[0].estado for e in sequencia]


def test_janela_invalida():
    with pytest.raises(ValueError):
        Suavizador(0)


def test_um_quadro_estranho_no_meio_nao_troca_o_estado():
    s = Suavizador(5)
    saida = estados(s, [Estado.OCUPADA] * 5 + [Estado.LIVRE] + [Estado.OCUPADA] * 2)
    assert saida[-3:] == [Estado.OCUPADA, Estado.OCUPADA, Estado.OCUPADA]


def test_mudanca_de_verdade_passa_quando_a_maioria_concorda():
    s = Suavizador(5)
    estados(s, [Estado.OCUPADA] * 5)
    saida = estados(s, [Estado.LIVRE] * 5)
    assert saida[-1] is Estado.LIVRE
    assert saida[0] is Estado.OCUPADA


def test_primeira_leitura_sai_direto():
    s = Suavizador(5)
    assert estados(s, [Estado.OCUPADA]) == [Estado.OCUPADA]


def test_janela_de_um_nao_suaviza_nada():
    s = Suavizador(1)
    assert estados(s, [Estado.OCUPADA, Estado.LIVRE]) == [Estado.OCUPADA, Estado.LIVRE]


def test_esquecer_zera_o_historico():
    s = Suavizador(3)
    estados(s, [Estado.OCUPADA] * 3)
    s.esquecer()
    assert estados(s, [Estado.LIVRE]) == [Estado.LIVRE]


def test_suavizacao_preserva_confianca_e_cobertura():
    s = Suavizador(3)
    entrada = Resultado(leituras=[Leitura("1", Estado.OCUPADA, 0.8, 0.6, "carro")])
    saida = s.aplicar(entrada).leituras[0]
    assert (saida.confianca, saida.cobertura, saida.classe_detectada) == (0.8, 0.6, "carro")


def test_vagas_diferentes_tem_historicos_independentes():
    s = Suavizador(3)
    entrada = Resultado(leituras=[
        Leitura("1", Estado.OCUPADA), Leitura("2", Estado.LIVRE),
    ])
    for _ in range(3):
        saida = s.aplicar(entrada)
    assert [l.estado for l in saida.leituras] == [Estado.OCUPADA, Estado.LIVRE]
