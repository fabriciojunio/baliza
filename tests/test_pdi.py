"""Testes do detector clássico.

Verificam comportamento com casos sintéticos cuja resposta é conhecida: uma
imagem lisa tem de parecer vaga vazia, e uma imagem com estrutura tem de
parecer vaga ocupada. Nenhum teste depende do PKLot estar em disco.
"""

from __future__ import annotations

import numpy as np
import pytest

from baliza.pdi import (
    NOMES_DAS_MEDIDAS,
    Calibracao,
    DetectorClassico,
    Medidas,
    limiar_de_otsu,
    medir,
    recortar_vaga,
)
from baliza.tipos import Vaga


def _lisa(valor: int = 120) -> np.ndarray:
    return np.full((64, 64, 3), valor, dtype=np.uint8)


def _estruturada(semente: int = 0) -> np.ndarray:
    rng = np.random.default_rng(semente)
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    # retângulo claro com borda escura: a assinatura de um carro visto de cima
    img[18:46, 10:54] = 215
    img[18:20, 10:54] = 20
    img[44:46, 10:54] = 20
    img[18:46, 10:12] = 20
    img[18:46, 52:54] = 20
    img = img.astype(np.int16) + rng.integers(-8, 9, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_medidas_tem_as_cinco():
    m = medir(_lisa())
    assert len(m.como_vetor()) == 5
    assert set(m.como_dict()) == set(NOMES_DAS_MEDIDAS)


def test_superficie_lisa_tem_pouca_estrutura():
    m = medir(_lisa())
    assert m.densidade_de_borda < 0.05
    assert m.desvio_do_cinza < 5.0


def test_estrutura_aumenta_todas_as_medidas():
    """A direção de cada medida é o que o detector assume; se inverter, ele erra."""
    lisa = medir(_lisa())
    carro = medir(_estruturada())
    assert carro.densidade_de_borda > lisa.densidade_de_borda
    assert carro.desvio_do_cinza > lisa.desvio_do_cinza
    assert carro.gradiente_medio > lisa.gradiente_medio
    assert carro.entropia > lisa.entropia


def test_recorte_retifica_para_tamanho_fixo():
    """Vaga longe e vaga perto têm de virar o mesmo tamanho antes de medir."""
    quadro = np.zeros((480, 640, 3), dtype=np.uint8)
    quadro[100:140, 200:260] = 200
    pequena = [(200, 100), (260, 100), (260, 140), (200, 140)]
    grande = [(100, 50), (400, 50), (400, 300), (100, 300)]
    a = recortar_vaga(quadro, pequena)
    b = recortar_vaga(quadro, grande)
    assert a.shape == b.shape == (64, 64, 3)


def test_recorte_ordena_os_cantos():
    """Canto fora de ordem não pode espelhar nem torcer o recorte."""
    quadro = np.zeros((200, 200, 3), dtype=np.uint8)
    quadro[20:60, 20:100] = 255
    ordenado = [(20, 20), (100, 20), (100, 60), (20, 60)]
    bagunçado = [(100, 60), (20, 20), (20, 60), (100, 20)]
    a = recortar_vaga(quadro, ordenado)
    b = recortar_vaga(quadro, bagunçado)
    assert np.allclose(a.astype(int), b.astype(int), atol=2)


def test_recorte_tolera_poligono_fora_do_quadro():
    quadro = np.zeros((100, 100, 3), dtype=np.uint8)
    r = recortar_vaga(quadro, [(-50, -50), (20, -50), (20, 20), (-50, 20)])
    assert r.shape == (64, 64, 3)


def test_otsu_separa_dois_grupos():
    valores = np.array([1.0, 1.1, 0.9, 1.0, 9.0, 9.2, 8.8, 9.1])
    t = limiar_de_otsu(valores)
    assert 1.1 < t < 9.0


def test_otsu_com_valor_unico_nao_quebra():
    assert limiar_de_otsu(np.array([5.0])) == pytest.approx(5.0)
    assert np.isfinite(limiar_de_otsu(np.array([3.0, 3.0, 3.0])))


def test_calibracao_ida_e_volta():
    c = Calibracao(
        medias=np.arange(5, dtype=float),
        desvios=np.ones(5),
        pesos=np.array([1.0, -1.0, 0.0, 0.0, 1.0]),
        limiar=0.5,
        camera_de_treino="A+B",
        acuracia_de_treino=0.9,
    )
    d = Calibracao.de_dict(c.como_dict())
    assert np.allclose(d.pesos, c.pesos)
    assert d.limiar == c.limiar
    assert d.camera_de_treino == "A+B"


def test_calibracao_classifica_pela_pontuacao():
    c = Calibracao(
        medias=np.zeros(5),
        desvios=np.ones(5),
        pesos=np.array([1.0, 0, 0, 0, 0]),
        limiar=0.5,
    )
    assert c.ocupada(Medidas(1.0, 0, 0, 0, 0))
    assert not c.ocupada(Medidas(0.1, 0, 0, 0, 0))


def _patio(n_ocupadas: int, n_livres: int):
    """Monta um quadro com vagas lado a lado, metade lisa e metade com carro."""
    total = n_ocupadas + n_livres
    quadro = np.full((160, 140 * total, 3), 120, dtype=np.uint8)
    vagas = []
    for i in range(total):
        x = 140 * i + 20
        contorno = [(x, 40), (x + 100, 40), (x + 100, 130), (x, 130)]
        ocupada = i < n_ocupadas
        if ocupada:
            quadro[55:115, x + 10 : x + 90] = 215
            quadro[55:59, x + 10 : x + 90] = 20
            quadro[111:115, x + 10 : x + 90] = 20
            quadro[55:115, x + 10 : x + 14] = 20
            quadro[55:115, x + 86 : x + 90] = 20
        vagas.append(Vaga(id=f"v{i}", contorno=contorno))
    return quadro, vagas, [i < n_ocupadas for i in range(total)]


def test_detector_separa_vaga_lisa_de_vaga_com_carro():
    """Ponta a ponta: num mesmo quadro, Otsu tem de achar os dois grupos."""
    quadro, vagas, esperado = _patio(n_ocupadas=4, n_livres=4)
    d = DetectorClassico(modo="otsu")
    ocupadas, medidas, limiar = d.estados(quadro, vagas)
    assert ocupadas == esperado, f"medidas: {[m.como_dict() for m in medidas]}"
    assert np.isfinite(limiar)
    assert d.ms_ultima_inferencia > 0


def test_otsu_erra_quando_o_patio_esta_todo_cheio():
    """Limite declarado: sem dois grupos no quadro, o modo Otsu inventa um.

    Com todas as vagas ocupadas e um pouco de variacao entre elas, o Otsu
    corta no maior salto e chama de livre a metade menos estruturada. E a
    razao de o modo calibrado existir, e o numero esta no relatorio: em 80
    imagens do teste o patio estava quase todo cheio ou quase todo vazio.
    """
    quadro, vagas, _ = _patio(n_ocupadas=6, n_livres=0)
    rng = np.random.default_rng(3)
    ruido = rng.integers(-6, 7, quadro.shape)
    quadro = np.clip(quadro.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
    ocupadas, _, _ = DetectorClassico(modo="otsu").estados(quadro, vagas)
    assert not all(ocupadas), (
        "o modo Otsu deveria errar aqui, e e por isso que ele nao e o padrao"
    )


def test_detectar_devolve_uma_deteccao_por_vaga_ocupada():
    quadro, vagas, _ = _patio(n_ocupadas=1, n_livres=5)
    saida = DetectorClassico(modo="otsu").detectar(quadro, vagas)
    assert len(saida) == 1
    assert saida[0].classe == "veiculo_por_pdi"
    assert 0.0 <= saida[0].confianca <= 1.0


def test_detector_nao_carrega_peso_nenhum():
    """A razão de existir do detector: nenhum arquivo de modelo envolvido."""
    d = DetectorClassico()
    assert not hasattr(d, "pesos")
    assert d.nome == "classico"


@pytest.mark.parametrize("normalizacao", ["camera", "imagem"])
def test_runtime_recusa_calibracao_com_momentos_externos(normalizacao):
    cal = Calibracao(normalizacao=normalizacao)
    with pytest.raises(ValueError, match="incompativel com execucao online"):
        DetectorClassico(cal)


def test_calibracao_sem_protocolo_nao_e_adivinhada():
    dados = Calibracao().como_dict()
    del dados["normalizacao"]
    with pytest.raises(ValueError, match="sem normalizacao declarada"):
        Calibracao.de_dict(dados)


@pytest.mark.parametrize("campo,valor", [
    ("medias", [0.0]),
    ("pesos", [float("nan")] * 5),
    ("desvios", [-1.0] * 5),
    ("limiar", float("inf")),
    ("limiar", None),
    ("medidas", list(reversed(NOMES_DAS_MEDIDAS))),
])
def test_calibracao_recusa_estatisticas_incompativeis(campo, valor):
    dados = Calibracao().como_dict()
    dados[campo] = valor
    with pytest.raises(ValueError):
        Calibracao.de_dict(dados)
