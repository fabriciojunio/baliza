import pytest

from baliza.ocupacao import decidir_por_vaga, decidir_por_veiculo
from baliza.tipos import Deteccao, Estado


def test_vaga_com_carro_em_cima_fica_ocupada(mapa_de_quatro, carro_na_vaga_1):
    resultado = decidir_por_veiculo(mapa_de_quatro, carro_na_vaga_1)
    por_id = {l.vaga_id: l for l in resultado.leituras}
    assert por_id["1"].estado is Estado.OCUPADA
    assert por_id["2"].estado is Estado.LIVRE


def test_patio_vazio_deixa_tudo_livre(mapa_de_quatro):
    resultado = decidir_por_veiculo(mapa_de_quatro, [])
    assert resultado.livres == 4
    assert resultado.ocupadas == 0


def test_carro_de_raspao_nao_ocupa_a_vaga_vizinha(mapa_de_quatro):
    # cobre 20% da vaga 2, abaixo do limiar de 30%
    deteccoes = [Deteccao((18, 0, 22, 10), "carro", 0.8)]
    resultado = decidir_por_veiculo(mapa_de_quatro, deteccoes, limiar=0.30)
    por_id = {l.vaga_id: l for l in resultado.leituras}
    assert por_id["2"].estado is Estado.LIVRE
    assert por_id["2"].cobertura == pytest.approx(0.2)


def test_limiar_mais_frouxo_muda_a_decisao(mapa_de_quatro):
    deteccoes = [Deteccao((18, 0, 22, 10), "carro", 0.8)]
    resultado = decidir_por_veiculo(mapa_de_quatro, deteccoes, limiar=0.15)
    por_id = {l.vaga_id: l for l in resultado.leituras}
    assert por_id["2"].estado is Estado.OCUPADA


def test_caminhao_ocupa_varias_vagas_de_uma_vez(mapa_de_quatro):
    deteccoes = [Deteccao((0, 0, 50, 10), "caminhao", 0.7)]
    resultado = decidir_por_veiculo(mapa_de_quatro, deteccoes)
    assert resultado.ocupadas == 3


def test_zona_de_duvida_evita_chamar_de_livre(mapa_de_quatro):
    deteccoes = [Deteccao((0, 0, 10, 10 * 0.25), "carro", 0.6)]  # cobre 25%
    resultado = decidir_por_veiculo(mapa_de_quatro, deteccoes, limiar=0.30, zona_duvida=0.5)
    por_id = {l.vaga_id: l for l in resultado.leituras}
    assert por_id["1"].estado is Estado.SEM_LEITURA


def test_leitura_guarda_a_classe_do_veiculo(mapa_de_quatro):
    deteccoes = [Deteccao((0, 0, 10, 10), "moto", 0.55)]
    resultado = decidir_por_veiculo(mapa_de_quatro, deteccoes)
    por_id = {l.vaga_id: l for l in resultado.leituras}
    assert por_id["1"].classe_detectada == "moto"
    assert por_id["2"].classe_detectada is None


def test_detector_de_vagas_usa_o_rotulo_do_modelo(mapa_de_quatro):
    deteccoes = [
        Deteccao((0, 0, 10, 10), "vaga-ocupada", 0.9),
        Deteccao((20, 0, 30, 10), "vaga-livre", 0.8),
    ]
    resultado = decidir_por_vaga(mapa_de_quatro, deteccoes)
    por_id = {l.vaga_id: l for l in resultado.leituras}
    assert por_id["1"].estado is Estado.OCUPADA
    assert por_id["2"].estado is Estado.LIVRE


def test_vaga_que_o_modelo_nao_viu_fica_sem_leitura(mapa_de_quatro):
    resultado = decidir_por_vaga(mapa_de_quatro, [])
    assert resultado.sem_leitura == 4
    assert resultado.livres == 0


def test_entre_duas_deteccoes_na_mesma_vaga_vence_a_mais_confiante(mapa_de_quatro):
    deteccoes = [
        Deteccao((0, 0, 10, 10), "vaga-livre", 0.4),
        Deteccao((0, 0, 10, 10), "vaga-ocupada", 0.95),
    ]
    resultado = decidir_por_vaga(mapa_de_quatro, deteccoes)
    por_id = {l.vaga_id: l for l in resultado.leituras}
    assert por_id["1"].estado is Estado.OCUPADA
    assert por_id["1"].confianca == pytest.approx(0.95)


def test_deteccao_longe_de_qualquer_vaga_e_ignorada(mapa_de_quatro):
    deteccoes = [Deteccao((500, 500, 520, 520), "vaga-ocupada", 0.9)]
    resultado = decidir_por_vaga(mapa_de_quatro, deteccoes)
    assert resultado.sem_leitura == 4


def test_ms_de_inferencia_atravessa_o_resultado(mapa_de_quatro):
    resultado = decidir_por_veiculo(mapa_de_quatro, [], ms_inferencia=42.0)
    assert resultado.ms_inferencia == 42.0
