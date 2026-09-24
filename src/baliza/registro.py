"""Histórico de ocupação em SQLite.

Grava-se a linha de estado, nunca o quadro. Imagem de pátio pode conter
placa, que identifica pessoa por caminho indireto, e o sistema não precisa
dela para nada.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from .tipos import Estado, Resultado

ESQUEMA = """
CREATE TABLE IF NOT EXISTS leitura (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    camera    TEXT    NOT NULL,
    vaga_id   TEXT    NOT NULL,
    estado    TEXT    NOT NULL,
    confianca REAL    NOT NULL DEFAULT 0,
    cobertura REAL    NOT NULL DEFAULT 0,
    instante  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_leitura_camera_instante ON leitura (camera, instante);
CREATE INDEX IF NOT EXISTS idx_leitura_vaga ON leitura (camera, vaga_id, instante);
"""


def agora() -> str:
    # Milissegundos, e nao segundos: o sistema chega a ler mais de um quadro
    # por segundo, e dois quadros com o mesmo carimbo viram uma leitura so na
    # hora de perguntar qual e o estado atual.
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


class Registro:
    def __init__(self, caminho: str | Path = "baliza.db"):
        self.caminho = str(caminho)
        if self.caminho != ":memory:":
            Path(self.caminho).parent.mkdir(parents=True, exist_ok=True)
        self.conexao = sqlite3.connect(self.caminho, check_same_thread=False)
        self.conexao.row_factory = sqlite3.Row
        self.conexao.executescript(ESQUEMA)
        self.conexao.commit()

    def __enter__(self) -> "Registro":
        return self

    def __exit__(self, *_) -> None:
        self.fechar()

    def fechar(self) -> None:
        self.conexao.close()

    def gravar(self, camera: str, resultado: Resultado, instante: str | None = None) -> int:
        instante = instante or agora()
        linhas = [
            (camera, l.vaga_id, str(l.estado), l.confianca, l.cobertura, instante)
            for l in resultado.leituras
        ]
        with closing(self.conexao.cursor()) as cursor:
            cursor.executemany(
                "INSERT INTO leitura (camera, vaga_id, estado, confianca, cobertura, instante)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                linhas,
            )
        self.conexao.commit()
        return len(linhas)

    def ultimo_instante(self, camera: str) -> str | None:
        linha = self.conexao.execute(
            "SELECT MAX(instante) AS ultimo FROM leitura WHERE camera = ?", (camera,)
        ).fetchone()
        return linha["ultimo"] if linha else None

    def estado_atual(self, camera: str) -> dict[str, str]:
        instante = self.ultimo_instante(camera)
        if instante is None:
            return {}
        linhas = self.conexao.execute(
            "SELECT vaga_id, estado FROM leitura WHERE camera = ? AND instante = ?",
            (camera, instante),
        ).fetchall()
        return {linha["vaga_id"]: linha["estado"] for linha in linhas}

    def serie_ocupacao(self, camera: str) -> list[tuple[str, int, int]]:
        """(instante, ocupadas, total) para desenhar a curva do dia."""
        linhas = self.conexao.execute(
            "SELECT instante,"
            "       SUM(CASE WHEN estado = ? THEN 1 ELSE 0 END) AS ocupadas,"
            "       COUNT(*) AS total"
            "  FROM leitura WHERE camera = ?"
            " GROUP BY instante ORDER BY instante",
            (str(Estado.OCUPADA), camera),
        ).fetchall()
        return [(l["instante"], int(l["ocupadas"]), int(l["total"])) for l in linhas]

    def cameras(self) -> list[str]:
        linhas = self.conexao.execute(
            "SELECT DISTINCT camera FROM leitura ORDER BY camera"
        ).fetchall()
        return [linha["camera"] for linha in linhas]

    def exportar_csv(self, caminho: str | Path, camera: str | None = None) -> int:
        import csv

        consulta = "SELECT camera, vaga_id, estado, confianca, cobertura, instante FROM leitura"
        parametros: tuple = ()
        if camera:
            consulta += " WHERE camera = ?"
            parametros = (camera,)
        consulta += " ORDER BY instante, vaga_id"

        linhas = self.conexao.execute(consulta, parametros).fetchall()
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        with caminho.open("w", newline="", encoding="utf-8") as arquivo:
            escritor = csv.writer(arquivo)
            escritor.writerow(
                ["camera", "vaga", "estado", "confianca", "cobertura", "instante"]
            )
            for linha in linhas:
                escritor.writerow([linha[c] for c in linha.keys()])
        return len(linhas)
