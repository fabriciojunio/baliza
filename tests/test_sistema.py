from baliza.registro import Registro
from baliza.sistema import Baliza
from baliza.tipos import Deteccao, Estado

from conftest import DetectorFalso


def test_processa_um_quadro(quadro, mapa_de_quatro):
    detector = DetectorFalso([Deteccao((0, 0, 10, 10), "carro", 0.9)])
    sistema = Baliza(mapa_de_quatro, detector, janela_suavizacao=1)
    resultado = sistema.processar(quadro)
    assert resultado.ocupadas == 1
    assert resultado.livres == 3
    assert resultado.ms_inferencia == 7.5


def test_suavizacao_segura_o_quadro_estranho(quadro, mapa_de_quatro):
    detector = DetectorFalso([Deteccao((0, 0, 10, 10), "carro", 0.9)])
    sistema = Baliza(mapa_de_quatro, detector, janela_suavizacao=5)
    for _ in range(5):
        sistema.processar(quadro)
    detector.deteccoes = []
    assert sistema.processar(quadro).ocupadas == 1


def test_rodar_a_fonte_inteira_grava_no_banco(tmp_path, mapa_de_quatro):
    import cv2
    import numpy as np

    from baliza.captura import FontePasta

    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i}.jpg"), np.zeros((50, 100, 3), dtype=np.uint8))

    registro = Registro(":memory:")
    detector = DetectorFalso([Deteccao((0, 0, 10, 10), "carro", 0.9)])
    sistema = Baliza(mapa_de_quatro, detector, registro=registro)
    resultados = sistema.rodar(FontePasta(tmp_path))

    assert len(resultados) == 3
    assert detector.chamadas == 3
    assert registro.estado_atual("teste")["1"] == "ocupada"
    registro.fechar()


def test_gravar_video_anotado(tmp_path, mapa_de_quatro):
    import cv2
    import numpy as np

    from baliza.captura import FontePasta

    for i in range(2):
        cv2.imwrite(str(tmp_path / f"{i}.jpg"), np.zeros((50, 100, 3), dtype=np.uint8))
    saida = tmp_path / "anotado.mp4"
    sistema = Baliza(mapa_de_quatro, DetectorFalso())
    sistema.rodar(FontePasta(tmp_path), gravar=saida)
    assert saida.exists() and saida.stat().st_size > 0


def test_callback_recebe_o_quadro_anotado(tmp_path, mapa_de_quatro):
    import cv2
    import numpy as np

    from baliza.captura import FontePasta

    cv2.imwrite(str(tmp_path / "a.jpg"), np.zeros((50, 100, 3), dtype=np.uint8))
    vistos = []
    sistema = Baliza(mapa_de_quatro, DetectorFalso())
    sistema.rodar(FontePasta(tmp_path), a_cada_quadro=lambda i, q, r: vistos.append(r))
    assert len(vistos) == 1
    assert vistos[0].total == 4
