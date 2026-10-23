import pytest

from baliza.cli import construir_parser, main


def test_parser_tem_os_tres_comandos():
    parser = construir_parser()
    for comando in ("rodar", "mapa-do-pklot", "demonstracao"):
        assert parser.parse_args([comando] + (["x", "--mapa", "m.json"] if comando == "rodar"
                                              else (["raiz", "--saida", "s.json"]
                                                    if comando == "mapa-do-pklot" else []))
                                 ).comando == comando


def test_rodar_exige_mapa():
    with pytest.raises(SystemExit):
        construir_parser().parse_args(["rodar", "video.mp4"])


def test_valores_padrao_de_rodar():
    args = construir_parser().parse_args(["rodar", "v.mp4", "--mapa", "m.json"])
    assert args.tamanho == 1280
    assert args.limiar == 0.30
    assert args.janela == 5
    # Sem --detector o valor fica em aberto: quem decide e a recomendacao do mapa.
    assert args.detector is None


def test_grade_de_janelas_e_convertida():
    args = construir_parser().parse_args(
        ["rodar", "v.mp4", "--mapa", "m.json", "--janelas", "3x2"])
    assert args.janelas == (3, 2)


def test_grade_de_janelas_invalida():
    with pytest.raises(SystemExit):
        construir_parser().parse_args(
            ["rodar", "v.mp4", "--mapa", "m.json", "--janelas", "tres"])


def test_detector_invalido_e_rejeitado():
    with pytest.raises(SystemExit):
        construir_parser().parse_args(
            ["rodar", "v.mp4", "--mapa", "m.json", "--detector", "chute"]
        )


def test_mapa_do_pklot_com_base_inexistente(tmp_path, capsys):
    codigo = main(["mapa-do-pklot", str(tmp_path / "nada"), "--saida", str(tmp_path / "m.json")])
    assert codigo == 2


def test_rodar_com_alvo_inexistente(tmp_path, capsys, mapa_de_quatro):
    caminho = tmp_path / "m.json"
    mapa_de_quatro.salvar(caminho)
    codigo = main(["rodar", str(tmp_path / "nao_existe.mp4"), "--mapa", str(caminho),
                   "--pesos", "modelos/yolo11n.pt"])
    assert codigo == 2
    assert "erro" in capsys.readouterr().err


def test_sem_argumento_cai_na_demonstracao(monkeypatch):
    """Quem abre o programa por engano tem que ver alguma coisa acontecer."""
    chamado = {}
    import baliza.cli as cli

    monkeypatch.setattr(cli, "_comando_demonstracao", lambda args: chamado.setdefault("ok", 1) or 0)
    parser = cli.construir_parser()
    args = parser.parse_args(["demonstracao"])
    assert args.comando == "demonstracao"


def test_argumentos_de_verdade_nao_sao_descartados(monkeypatch, tmp_path, mapa_de_quatro):
    """Regressao: main(None) chegou a trocar sys.argv pela demonstracao."""
    import sys

    import baliza.cli as cli

    caminho = tmp_path / "m.json"
    mapa_de_quatro.salvar(caminho)
    monkeypatch.setattr(
        sys, "argv",
        ["baliza", "rodar", str(tmp_path / "nao_existe.mp4"), "--mapa", str(caminho),
         "--pesos", "modelos/yolo11n.pt"],
    )
    # Se os argumentos fossem descartados, cairia na demonstração e devolveria 0.
    assert cli.main() == 2


def test_lista_vazia_cai_na_demonstracao(monkeypatch):
    import baliza.cli as cli

    visto = {}
    monkeypatch.setattr(cli, "_comando_demonstracao",
                        lambda args: visto.setdefault("escolha", args.escolha) or 0)
    assert cli.main([]) == 0
    assert "escolha" in visto



class MapaFalso:
    def __init__(self, detector=None, camera="X"):
        self.detector = detector
        self.camera = camera


def test_detector_explicito_ganha_do_mapa():
    from baliza.cli import _resolver_detector

    modo, _, motivo = _resolver_detector("veiculos", None, MapaFalso(detector="vagas"))
    assert modo == "veiculos"
    assert "linha de comando" in motivo


def test_sem_escolha_vale_a_recomendacao_do_mapa(tmp_path):
    from baliza.cli import _resolver_detector

    pesos = tmp_path / "vagas.pt"
    pesos.write_bytes(b"x")
    modo, caminho, motivo = _resolver_detector(None, str(pesos), MapaFalso(detector="vagas"))
    assert modo == "vagas"
    assert caminho == str(pesos)
    assert "recomendado" in motivo


def test_mapa_sem_recomendacao_cai_no_detector_geral():
    from baliza.cli import _resolver_detector

    modo, _, motivo = _resolver_detector(None, None, MapaFalso())
    assert modo == "veiculos"
    assert motivo == "padrao"


def test_pesos_treinados_ausentes_caem_no_geral_avisando(tmp_path):
    """Numa maquina sem os pesos, a demonstracao tem que rodar assim mesmo."""
    from baliza.cli import _resolver_detector

    modo, _, motivo = _resolver_detector(None, str(tmp_path / "nao_existe.pt"),
                                         MapaFalso(detector="vagas"))
    assert modo == "veiculos"
    assert "nao esta em disco" in motivo


def test_classico_processa_imagem_com_calibracao_explicita(tmp_path, mapa_de_quatro, quadro, capsys):
    import json

    import cv2

    from baliza.pdi import Calibracao

    mapa = tmp_path / "mapa.json"
    mapa_de_quatro.salvar(mapa)
    imagem = tmp_path / "patio.png"
    assert cv2.imwrite(str(imagem), quadro)
    calibracao = tmp_path / "calibracao.json"
    calibracao.write_text(json.dumps(Calibracao(limiar=2.0).como_dict()), encoding="utf-8")
    assert main(["rodar", str(imagem), "--mapa", str(mapa), "--detector", "classico",
                 "--pesos", str(calibracao), "--janela", "1"]) == 0
    saida = capsys.readouterr().out
    assert "1 quadros processados" in saida
    assert "4 livres, 0 ocupadas" in saida


def test_classico_padrao_resolve_json_de_calibracao():
    from pathlib import Path

    from baliza.cli import _resolver_detector

    modo, caminho, _ = _resolver_detector("classico", None, MapaFalso())
    assert modo == "classico"
    assert Path(caminho).name == "pdi_calibracao.json"


def test_classico_recomendado_no_mapa_preserva_calibracao(tmp_path, mapa_de_quatro):
    from baliza.cli import _resolver_detector
    from baliza.mapa import Mapa

    mapa_de_quatro.detector = "classico"
    mapa_de_quatro.pesos = str(tmp_path / "minha_calibracao.json")
    caminho_mapa = tmp_path / "mapa.json"
    mapa_de_quatro.salvar(caminho_mapa)
    lido = Mapa.carregar(caminho_mapa)
    modo, caminho, motivo = _resolver_detector(None, None, lido)
    assert modo == "classico"
    assert caminho == mapa_de_quatro.pesos
    assert "recomendado" in motivo


@pytest.mark.parametrize("tipo", ["ausente", "camera", "json_quebrado"])
def test_classico_falha_com_erro_claro_para_calibracao_invalida(
    tipo, tmp_path, mapa_de_quatro, capsys
):
    import json

    from baliza.pdi import Calibracao

    mapa = tmp_path / "mapa.json"
    mapa_de_quatro.salvar(mapa)
    cal = tmp_path / "calibracao.json"
    if tipo == "camera":
        cal.write_text(json.dumps(Calibracao(normalizacao="camera").como_dict()), encoding="utf-8")
    elif tipo == "json_quebrado":
        cal.write_text("{", encoding="utf-8")
    codigo = main(["rodar", "imagem.png", "--mapa", str(mapa),
                   "--detector", "classico", "--pesos", str(cal)])
    assert codigo == 2
    assert "erro:" in capsys.readouterr().err
