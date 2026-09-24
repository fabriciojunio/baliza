from baliza.tipos import Estado, Leitura, Resultado


def resultado_exemplo():
    return Resultado(leituras=[
        Leitura("1", Estado.LIVRE),
        Leitura("2", Estado.OCUPADA),
        Leitura("3", Estado.OCUPADA),
        Leitura("4", Estado.SEM_LEITURA),
    ])


def test_contagens():
    r = resultado_exemplo()
    assert (r.livres, r.ocupadas, r.sem_leitura, r.total) == (1, 2, 1, 4)


def test_por_setor_nao_conta_sem_leitura_como_livre(mapa_de_quatro):
    r = resultado_exemplo()
    assert r.por_setor(mapa_de_quatro.por_id) == {"A": (1, 2), "B": (0, 2)}


def test_vaga_fora_do_mapa_cai_no_setor_interrogacao(mapa_de_quatro):
    r = Resultado(leituras=[Leitura("99", Estado.LIVRE)])
    assert r.por_setor(mapa_de_quatro.por_id) == {"?": (1, 1)}


def test_estado_vira_texto_legivel():
    assert str(Estado.OCUPADA) == "ocupada"
    assert f"{Estado.SEM_LEITURA}" == "sem_leitura"


def test_leitura_sabe_se_esta_ocupada():
    assert Leitura("1", Estado.OCUPADA).ocupada
    assert not Leitura("1", Estado.LIVRE).ocupada


def test_vaga_calcula_centro_e_envolvente(mapa_de_quatro):
    vaga = mapa_de_quatro.por_id["1"]
    assert vaga.centro == (5, 5)
    assert vaga.envolvente == (0, 0, 10, 10)
