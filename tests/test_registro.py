from baliza.registro import Registro
from baliza.tipos import Estado, Leitura, Resultado


def resultado(ocupadas):
    leituras = [Leitura(str(i), Estado.OCUPADA) for i in range(ocupadas)]
    leituras += [Leitura(f"l{i}", Estado.LIVRE) for i in range(3)]
    return Resultado(leituras=leituras)


def test_grava_e_le_o_estado_atual():
    with Registro(":memory:") as r:
        assert r.gravar("patio", resultado(2), "2026-09-23T10:00:00") == 5
        atual = r.estado_atual("patio")
        assert atual["0"] == "ocupada"
        assert atual["l0"] == "livre"


def test_estado_atual_de_camera_sem_nada_e_vazio():
    with Registro(":memory:") as r:
        assert r.estado_atual("nao_existe") == {}


def test_serie_de_ocupacao_sai_em_ordem():
    with Registro(":memory:") as r:
        r.gravar("patio", resultado(1), "2026-09-23T10:00:00")
        r.gravar("patio", resultado(3), "2026-09-23T10:05:00")
        serie = r.serie_ocupacao("patio")
        assert [ocupadas for _, ocupadas, _ in serie] == [1, 3]
        assert serie[0][2] == 4


def test_so_a_ultima_leitura_conta_como_atual():
    with Registro(":memory:") as r:
        r.gravar("patio", resultado(1), "2026-09-23T10:00:00")
        r.gravar("patio", resultado(3), "2026-09-23T10:05:00")
        assert sum(1 for e in r.estado_atual("patio").values() if e == "ocupada") == 3


def test_cameras_listadas_sem_repetir():
    with Registro(":memory:") as r:
        r.gravar("a", resultado(1))
        r.gravar("b", resultado(1))
        r.gravar("a", resultado(1))
        assert r.cameras() == ["a", "b"]


def test_exportacao_em_csv(tmp_path):
    caminho = tmp_path / "saida.csv"
    with Registro(":memory:") as r:
        r.gravar("patio", resultado(2), "2026-09-23T10:00:00")
        assert r.exportar_csv(caminho, "patio") == 5
    linhas = caminho.read_text(encoding="utf-8").strip().splitlines()
    assert linhas[0].startswith("camera,vaga,estado")
    assert len(linhas) == 6


def test_banco_em_arquivo_persiste(tmp_path):
    caminho = tmp_path / "sub" / "baliza.db"
    with Registro(caminho) as r:
        r.gravar("patio", resultado(1), "2026-09-23T10:00:00")
    with Registro(caminho) as r:
        assert r.ultimo_instante("patio") == "2026-09-23T10:00:00"


def test_dois_quadros_no_mesmo_segundo_nao_se_misturam():
    """O carimbo tem milissegundos justamente por causa disso."""
    from baliza.registro import agora

    primeiro, segundo = agora(), agora()
    assert len(primeiro) > len("2026-09-23T10:00:00-03:00")
    with Registro(":memory:") as r:
        r.gravar("patio", resultado(1), primeiro)
        r.gravar("patio", resultado(3), segundo if segundo != primeiro else primeiro + "x")
        assert len(r.serie_ocupacao("patio")) == 2
