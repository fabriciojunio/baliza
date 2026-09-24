"""Os dois detectores.

`DetectorVeiculos` usa os pesos gerais do YOLO (treinados no COCO) e procura
carro, moto, ônibus e caminhão. Ele funciona em qualquer pátio sem nenhum
treino, e é o caminho que o sistema usa por padrão.

`DetectorVagas` usa os pesos que treinamos no PKLot e procura a vaga em si,
já classificada em livre ou ocupada.

A diferença prática aparece na vaga vazia: o primeiro detector não vê nada
ali e conclui "livre" por ausência, o segundo vê a vaga vazia e afirma
"livre". Qual dos dois erra menos é medição, não opinião, e está no relatório.
"""

from __future__ import annotations

import time
from pathlib import Path

from .tipos import Deteccao

# Índices do COCO que interessam num estacionamento.
CLASSES_VEICULO_COCO = {2: "carro", 3: "moto", 5: "onibus", 7: "caminhao"}

PESOS_PADRAO = "yolo11n.pt"


def _escolher_dispositivo(pedido: str | None) -> str:
    if pedido and pedido != "auto":
        return pedido
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class DetectorBase:
    """Casca em volta do Ultralytics, para o resto do sistema não conhecê-lo."""

    def __init__(
        self,
        pesos: str | Path = PESOS_PADRAO,
        confianca: float = 0.25,
        tamanho: int = 640,
        dispositivo: str | None = "auto",
    ):
        self.pesos = str(pesos)
        self.confianca = confianca
        self.tamanho = tamanho
        self.dispositivo = _escolher_dispositivo(dispositivo)
        self._modelo = None
        self.ms_ultima_inferencia = 0.0

    @property
    def modelo(self):
        if self._modelo is None:
            try:
                from ultralytics import YOLO
            except ImportError as erro:  # pragma: sem cobertura
                raise ImportError(
                    "ultralytics nao esta instalado. Rode: pip install ultralytics"
                ) from erro
            self._modelo = YOLO(self.pesos)
        return self._modelo

    @property
    def nomes(self) -> dict[int, str]:
        return self.modelo.names

    def _rodar(self, quadro):
        inicio = time.perf_counter()
        resultado = self.modelo.predict(
            quadro,
            conf=self.confianca,
            imgsz=self.tamanho,
            device=self.dispositivo,
            verbose=False,
        )[0]
        self.ms_ultima_inferencia = (time.perf_counter() - inicio) * 1000
        return resultado

    def detectar(self, quadro) -> list[Deteccao]:  # pragma: sem cobertura
        raise NotImplementedError


class DetectorVeiculos(DetectorBase):
    """Procura veículo com os pesos gerais. Não precisa de treino nenhum."""

    def detectar(self, quadro) -> list[Deteccao]:
        resultado = self._rodar(quadro)
        deteccoes = []
        for caixa in resultado.boxes:
            indice = int(caixa.cls)
            nome = CLASSES_VEICULO_COCO.get(indice)
            if nome is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in caixa.xyxy[0])
            deteccoes.append(Deteccao((x1, y1, x2, y2), nome, float(caixa.conf)))
        return deteccoes


class DetectorVagas(DetectorBase):
    """Procura a vaga, já rotulada, com os pesos treinados no PKLot."""

    def detectar(self, quadro) -> list[Deteccao]:
        resultado = self._rodar(quadro)
        nomes = resultado.names
        deteccoes = []
        for caixa in resultado.boxes:
            x1, y1, x2, y2 = (float(v) for v in caixa.xyxy[0])
            classe = nomes[int(caixa.cls)]
            deteccoes.append(Deteccao((x1, y1, x2, y2), classe, float(caixa.conf)))
        return deteccoes


def nms(deteccoes: list[Deteccao], limiar_iou: float = 0.55) -> list[Deteccao]:
    """Supressão de sobreposição entre janelas.

    O mesmo carro aparece em duas janelas vizinhas quando cai na faixa de
    sobreposição, e sem isso ele viraria dois carros.
    """
    from .geometria import caixa_para_poligono, iou as iou_poligono

    ordenadas = sorted(deteccoes, key=lambda d: d.confianca, reverse=True)
    mantidas: list[Deteccao] = []
    for candidata in ordenadas:
        repetida = False
        for guardada in mantidas:
            if guardada.classe != candidata.classe:
                continue
            if iou_poligono(caixa_para_poligono(guardada.caixa), candidata.caixa) > limiar_iou:
                repetida = True
                break
        if not repetida:
            mantidas.append(candidata)
    return mantidas


class DetectorEmJanelas(DetectorVeiculos):
    """Roda o detector em pedaços do quadro, e não no quadro inteiro.

    Numa câmera distante, como a da PUCPR no décimo andar, o carro ocupa
    poucos pixels e o detector simplesmente não o vê. Recortar o quadro em
    janelas com sobreposição e rodar cada uma na resolução cheia do modelo
    aumenta o carro em proporção, que é o que o detector precisa.

    Custa uma inferência por janela; em compensação as janelas vão em lote
    para a GPU, então o custo real fica bem abaixo do proporcional.
    """

    def __init__(
        self,
        pesos: str | Path = PESOS_PADRAO,
        confianca: float = 0.25,
        tamanho: int = 640,
        dispositivo: str | None = "auto",
        colunas: int = 3,
        linhas: int = 2,
        sobreposicao: float = 0.2,
    ):
        super().__init__(pesos, confianca, tamanho, dispositivo)
        if colunas < 1 or linhas < 1:
            raise ValueError("colunas e linhas precisam ser pelo menos 1")
        if not 0 <= sobreposicao < 0.9:
            raise ValueError("sobreposicao precisa ficar entre 0 e 0.9")
        self.colunas = colunas
        self.linhas = linhas
        self.sobreposicao = sobreposicao

    def janelas(self, largura: int, altura: int) -> list[tuple[int, int, int, int]]:
        passo_x = largura / self.colunas
        passo_y = altura / self.linhas
        folga_x = passo_x * self.sobreposicao
        folga_y = passo_y * self.sobreposicao
        recortes = []
        for linha in range(self.linhas):
            for coluna in range(self.colunas):
                x1 = max(0, int(coluna * passo_x - folga_x))
                y1 = max(0, int(linha * passo_y - folga_y))
                x2 = min(largura, int((coluna + 1) * passo_x + folga_x))
                y2 = min(altura, int((linha + 1) * passo_y + folga_y))
                recortes.append((x1, y1, x2, y2))
        return recortes

    def detectar(self, quadro) -> list[Deteccao]:
        altura, largura = quadro.shape[:2]
        recortes = self.janelas(largura, altura)
        pedacos = [quadro[y1:y2, x1:x2] for x1, y1, x2, y2 in recortes]

        inicio = time.perf_counter()
        resultados = self.modelo.predict(
            pedacos,
            conf=self.confianca,
            imgsz=self.tamanho,
            device=self.dispositivo,
            verbose=False,
        )
        self.ms_ultima_inferencia = (time.perf_counter() - inicio) * 1000

        deteccoes = []
        for (x1, y1, _, _), resultado in zip(recortes, resultados):
            for caixa in resultado.boxes:
                nome = CLASSES_VEICULO_COCO.get(int(caixa.cls))
                if nome is None:
                    continue
                a, b, c, d = (float(v) for v in caixa.xyxy[0])
                deteccoes.append(
                    Deteccao((a + x1, b + y1, c + x1, d + y1), nome, float(caixa.conf))
                )
        return nms(deteccoes)


def abrir_detector(
    modo: str = "veiculos",
    pesos: str | Path | None = None,
    confianca: float = 0.25,
    tamanho: int = 640,
    dispositivo: str | None = "auto",
    janelas: tuple[int, int] | None = None,
) -> DetectorBase:
    if modo == "veiculos":
        if janelas:
            return DetectorEmJanelas(
                pesos or PESOS_PADRAO, confianca, tamanho, dispositivo,
                colunas=janelas[0], linhas=janelas[1],
            )
        return DetectorVeiculos(pesos or PESOS_PADRAO, confianca, tamanho, dispositivo)
    if modo == "vagas":
        if pesos is None:
            raise ValueError("o modo vagas exige os pesos treinados (--pesos)")
        return DetectorVagas(pesos, confianca, tamanho, dispositivo)
    raise ValueError(f"modo de deteccao desconhecido: {modo}")
