import cv2
import numpy as np
import pytest

from baliza.captura import (
    FonteImagem,
    FonteIndisponivel,
    FontePasta,
    FonteVideo,
    abrir_fonte,
)


@pytest.fixture
def imagem(tmp_path):
    caminho = tmp_path / "foto.jpg"
    cv2.imwrite(str(caminho), np.full((40, 60, 3), 90, dtype=np.uint8))
    return caminho


@pytest.fixture
def video(tmp_path):
    caminho = tmp_path / "video.mp4"
    escritor = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), 10, (60, 40))
    for i in range(20):
        escritor.write(np.full((40, 60, 3), i * 5, dtype=np.uint8))
    escritor.release()
    return caminho


def test_imagem_rende_um_quadro(imagem):
    with FonteImagem(imagem) as fonte:
        assert len(list(fonte.quadros())) == 1


def test_imagem_repetida_rende_varios(imagem):
    with FonteImagem(imagem, repetir=4) as fonte:
        assert len(list(fonte.quadros())) == 4


def test_imagem_inexistente(tmp_path):
    with pytest.raises(FonteIndisponivel):
        FonteImagem(tmp_path / "nada.jpg")


def test_pasta_le_em_ordem(tmp_path):
    for nome in ("c.jpg", "a.jpg", "b.jpg"):
        cv2.imwrite(str(tmp_path / nome), np.zeros((10, 10, 3), dtype=np.uint8))
    fonte = FontePasta(tmp_path)
    assert [p.name for p in fonte.arquivos] == ["a.jpg", "b.jpg", "c.jpg"]
    assert len(list(fonte.quadros())) == 3


def test_pasta_com_limite(tmp_path):
    for i in range(5):
        cv2.imwrite(str(tmp_path / f"{i}.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))
    assert FontePasta(tmp_path, limite=2).total_quadros == 2


def test_pasta_vazia(tmp_path):
    with pytest.raises(FonteIndisponivel):
        FontePasta(tmp_path)


def test_video_le_todos_os_quadros(video):
    with FonteVideo(video) as fonte:
        assert len(list(fonte.quadros())) == 20


def test_intervalo_reduz_a_leitura(video):
    with FonteVideo(video, intervalo_s=0.5) as fonte:
        assert fonte.passo == 5
        assert len(list(fonte.quadros())) == 4


def test_abrir_fonte_reconhece_imagem_pasta_e_video(imagem, video, tmp_path):
    assert isinstance(abrir_fonte(str(imagem)), FonteImagem)
    assert isinstance(abrir_fonte(str(video)), FonteVideo)
    assert isinstance(abrir_fonte(str(imagem.parent)), FontePasta)


def test_abrir_fonte_de_caminho_que_nao_existe():
    with pytest.raises(FonteIndisponivel):
        abrir_fonte("c:/nao/existe/mesmo.mp4")


def test_limite_corta_o_video(video):
    with FonteVideo(video, limite=6) as fonte:
        assert len(list(fonte.quadros())) == 6


def test_limite_vale_depois_do_intervalo(video):
    with FonteVideo(video, intervalo_s=0.2, limite=3) as fonte:
        assert len(list(fonte.quadros())) == 3


def test_abrir_fonte_repassa_o_limite_para_o_video(video):
    with abrir_fonte(str(video), limite=4) as fonte:
        assert len(list(fonte.quadros())) == 4
