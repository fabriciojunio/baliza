"""Uma nova divisão não pode coexistir com arquivos da exportação antiga."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from baliza.pklot import Foto

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "treino"))
import exportar_yolo  # noqa: E402


def test_exportacao_recusa_destino_antigo_sem_apagar(tmp_path):
    destino = tmp_path / "yolo"
    velho = destino / "images/val/quadro.jpg"
    velho.parent.mkdir(parents=True)
    velho.write_bytes(b"arquivo da divisao anterior")
    with pytest.raises(ValueError, match="destino nao vazio"):
        exportar_yolo.exportar("base", destino, 4, completo=True)
    assert velho.read_bytes() == b"arquivo da divisao anterior"
    assert list(destino.iterdir()) == [destino / "images"]


def test_exportacao_em_pasta_nova_escreve_imagens_e_rotulos(tmp_path, monkeypatch):
    fotos = {}
    for nome, dia in (("treino", 2), ("validacao", 5), ("teste", 3)):
        imagem = tmp_path / f"2013-01-{dia:02d}_10_00_00.jpg"
        assert cv2.imwrite(str(imagem), np.zeros((20, 30, 3), dtype=np.uint8))
        anotacao = imagem.with_suffix(".xml")
        anotacao.write_text('<parking><space id="1" occupied="1"><contour>'
                            '<point x="1" y="1"/><point x="10" y="1"/>'
                            '<point x="10" y="10"/><point x="1" y="10"/>'
                            '</contour></space></parking>', encoding="utf-8")
        fotos[nome] = [Foto(imagem, anotacao, "UFPR04", "Sunny", f"2013-01-{dia:02d}")]
    monkeypatch.setattr(exportar_yolo.divisao, "conjunto_completo",
                        lambda raiz, nome, **kw: fotos[nome])
    destino = tmp_path / "nova_divisao"
    assert exportar_yolo.exportar("base", destino, 4, completo=True) == dict(treino=1, validacao=1, teste=1)
    for pasta in ("train", "val", "test"):
        assert len(list((destino / "images" / pasta).glob("*.jpg"))) == 1
        rotulos = list((destino / "labels" / pasta).glob("*.txt"))
        assert len(rotulos) == 1
        assert rotulos[0].read_text(encoding="utf-8").startswith("1 ")
    assert (destino / "pklot.yaml").exists()


def test_exportacao_recusa_limite_nao_positivo(tmp_path):
    with pytest.raises(ValueError, match="maior que zero"):
        exportar_yolo.exportar("base", tmp_path / "novo", 0)
