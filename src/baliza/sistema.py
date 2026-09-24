"""Junta as peças: quadro entra, estado das vagas sai."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .captura import Fonte
from .deteccao import DetectorBase, DetectorVagas
from .desenho import anotar, legenda
from .mapa import Mapa
from .ocupacao import LIMIAR_COBERTURA, decidir_por_vaga, decidir_por_veiculo
from .registro import Registro
from .suavizacao import Suavizador
from .tipos import Resultado


class Baliza:
    def __init__(
        self,
        mapa: Mapa,
        detector: DetectorBase,
        limiar: float = LIMIAR_COBERTURA,
        janela_suavizacao: int = 5,
        registro: Registro | None = None,
    ):
        self.mapa = mapa
        self.detector = detector
        self.limiar = limiar
        self.suavizador = Suavizador(janela_suavizacao) if janela_suavizacao > 1 else None
        self.registro = registro

    @property
    def por_vaga(self) -> bool:
        return isinstance(self.detector, DetectorVagas)

    def processar(self, quadro: np.ndarray, suavizar: bool = True) -> Resultado:
        deteccoes = self.detector.detectar(quadro)
        ms = self.detector.ms_ultima_inferencia
        if self.por_vaga:
            resultado = decidir_por_vaga(self.mapa, deteccoes, ms_inferencia=ms)
        else:
            resultado = decidir_por_veiculo(
                self.mapa, deteccoes, limiar=self.limiar, ms_inferencia=ms
            )
        if suavizar and self.suavizador is not None:
            resultado = self.suavizador.aplicar(resultado)
        return resultado

    def rodar(
        self,
        fonte: Fonte,
        mostrar: bool = False,
        gravar: str | Path | None = None,
        a_cada_quadro: Callable[[int, np.ndarray, Resultado], None] | None = None,
    ) -> list[Resultado]:
        """Processa a fonte inteira. Devolve um resultado por quadro lido."""
        import cv2

        resultados: list[Resultado] = []
        escritor = None
        janela = f"Baliza - {self.mapa.camera}"

        try:
            for indice, quadro in fonte.quadros():
                resultado = self.processar(quadro)
                resultados.append(resultado)

                if self.registro is not None:
                    self.registro.gravar(self.mapa.camera, resultado)

                precisa_desenhar = mostrar or gravar is not None or a_cada_quadro is not None
                anotado = legenda(anotar(quadro, self.mapa, resultado)) if precisa_desenhar else quadro

                if gravar is not None:
                    if escritor is None:
                        altura, largura = anotado.shape[:2]
                        fps = fonte.fps if fonte.fps and fonte.fps > 1 else 5.0
                        escritor = cv2.VideoWriter(
                            str(gravar),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (largura, altura),
                        )
                    escritor.write(anotado)

                if a_cada_quadro is not None:
                    a_cada_quadro(indice, anotado, resultado)

                if mostrar:
                    cv2.imshow(janela, anotado)
                    tecla = cv2.waitKey(1) & 0xFF
                    if tecla in (ord("q"), 27):
                        break
                    if tecla == ord(" "):
                        while (cv2.waitKey(30) & 0xFF) != ord(" "):
                            pass
        finally:
            if escritor is not None:
                escritor.release()
            if mostrar:
                cv2.destroyAllWindows()
            fonte.fechar()

        return resultados
