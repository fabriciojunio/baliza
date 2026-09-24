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
    assert args.detector == "veiculos"


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
    # Se os argumentos fossem descartados, cairia na demonstracao e devolveria 0.
    assert cli.main() == 2


def test_lista_vazia_cai_na_demonstracao(monkeypatch):
    import baliza.cli as cli

    visto = {}
    monkeypatch.setattr(cli, "_comando_demonstracao",
                        lambda args: visto.setdefault("escolha", args.escolha) or 0)
    assert cli.main([]) == 0
    assert "escolha" in visto
