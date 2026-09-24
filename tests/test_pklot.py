import pytest

from baliza import pklot
from baliza.tipos import Estado

XML_BOM = """<?xml version="1.0"?>
<parking id="ufpr04">
  <space id="1" occupied="1">
    <rotatedRect><center x="50" y="50" /><size w="20" h="10" /><angle d="0" /></rotatedRect>
    <contour>
      <point x="40" y="45" /><point x="60" y="45" />
      <point x="60" y="55" /><point x="40" y="55" />
    </contour>
  </space>
  <space id="2" occupied="0">
    <rotatedRect><center x="90" y="50" /><size w="20" h="10" /><angle d="0" /></rotatedRect>
    <contour>
      <point x="80" y="45" /><point x="100" y="45" />
      <point x="100" y="55" /><point x="80" y="55" />
    </contour>
  </space>
</parking>
"""

XML_SEM_ROTULO = """<?xml version="1.0"?>
<parking id="x">
  <space id="1">
    <contour><point x="0" y="0" /><point x="1" y="0" /><point x="1" y="1" /></contour>
  </space>
</parking>
"""


@pytest.fixture
def anotacao(tmp_path):
    caminho = tmp_path / "foto.xml"
    caminho.write_text(XML_BOM, encoding="utf-8")
    return caminho


def test_le_duas_vagas_com_estado(anotacao):
    vagas = pklot.ler_anotacao(anotacao)
    assert len(vagas) == 2
    assert vagas[0][2] is Estado.OCUPADA
    assert vagas[1][2] is Estado.LIVRE


def test_verdade_vira_dicionario(anotacao):
    assert pklot.verdade(anotacao) == {"1": Estado.OCUPADA, "2": Estado.LIVRE}


def test_mapa_sai_da_anotacao(anotacao):
    mapa = pklot.mapa_de_anotacao(anotacao, "UFPR04", 1280, 720)
    assert len(mapa) == 2
    assert mapa.camera == "UFPR04"
    assert mapa.por_id["1"].centro == (50, 50)


def test_vaga_sem_o_atributo_occupied_e_descartada(tmp_path):
    caminho = tmp_path / "sem.xml"
    caminho.write_text(XML_SEM_ROTULO, encoding="utf-8")
    assert pklot.ler_anotacao(caminho) == []
    with pytest.raises(pklot.AnotacaoInvalida, match="nenhuma vaga"):
        pklot.mapa_de_anotacao(caminho, "x")


def test_xml_quebrado_avisa_qual_arquivo(tmp_path):
    caminho = tmp_path / "quebrado.xml"
    caminho.write_text("<parking><space", encoding="utf-8")
    with pytest.raises(pklot.AnotacaoInvalida, match="quebrado.xml"):
        pklot.ler_anotacao(caminho)


def test_base_que_nao_existe():
    with pytest.raises(FileNotFoundError, match="UFPR04"):
        pklot.listar_fotos("c:/isso/nao/existe")


def test_listagem_filtra_e_amostra(tmp_path):
    dia = tmp_path / "UFPR04" / "Sunny" / "2012-12-07"
    dia.mkdir(parents=True)
    for i in range(10):
        (dia / f"foto{i:02d}.jpg").write_bytes(b"x")
        (dia / f"foto{i:02d}.xml").write_text(XML_BOM, encoding="utf-8")
    (dia / "sem_par.jpg").write_bytes(b"x")

    todas = pklot.listar_fotos(tmp_path)
    assert len(todas) == 10

    amostradas = pklot.listar_fotos(tmp_path, limite_por_dia=3)
    assert len(amostradas) == 3

    assert pklot.listar_fotos(tmp_path, climas=("Rainy",)) == []
    assert pklot.dias_disponiveis(tmp_path, "UFPR04") == ["2012-12-07"]
