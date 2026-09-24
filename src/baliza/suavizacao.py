"""Voto da maioria das últimas leituras de cada vaga.

Em vídeo, uma leitura isolada oscila: alguém passa na frente da vaga, um
farol estoura o quadro, um pássaro cruza. Estacionamento muda devagar, então
trocar o estado publicado só depois de a maioria das últimas leituras
concordar custa alguns segundos de atraso e elimina o pisca-pisca.
"""

from __future__ import annotations

from collections import deque

from .tipos import Estado, Leitura, Resultado


class Suavizador:
    def __init__(self, janela: int = 5):
        if janela < 1:
            raise ValueError("a janela precisa ter pelo menos 1 leitura")
        self.janela = janela
        self._historico: dict[str, deque[Estado]] = {}
        self._publicado: dict[str, Estado] = {}

    def aplicar(self, resultado: Resultado) -> Resultado:
        suavizadas = []
        for leitura in resultado.leituras:
            fila = self._historico.setdefault(leitura.vaga_id, deque(maxlen=self.janela))
            fila.append(leitura.estado)
            suavizadas.append(
                Leitura(
                    vaga_id=leitura.vaga_id,
                    estado=self._votar(leitura.vaga_id, fila),
                    confianca=leitura.confianca,
                    cobertura=leitura.cobertura,
                    classe_detectada=leitura.classe_detectada,
                )
            )
        return Resultado(leituras=suavizadas, ms_inferencia=resultado.ms_inferencia)

    def _votar(self, vaga_id: str, fila: deque[Estado]) -> Estado:
        contagem: dict[Estado, int] = {}
        for estado in fila:
            contagem[estado] = contagem.get(estado, 0) + 1
        vencedor, votos = max(contagem.items(), key=lambda par: par[1])

        # Empate ou maioria simples: só troca o que está no ar quando a
        # maioria absoluta da janela cheia concordar. Enquanto a janela não
        # encheu, o último estado publicado vale.
        anterior = self._publicado.get(vaga_id)
        if anterior is None:
            self._publicado[vaga_id] = vencedor
            return vencedor
        if vencedor != anterior and votos * 2 <= len(fila):
            return anterior
        self._publicado[vaga_id] = vencedor
        return vencedor

    def esquecer(self) -> None:
        self._historico.clear()
        self._publicado.clear()
