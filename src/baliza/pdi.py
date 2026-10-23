"""Detector por processamento digital de imagens, sem rede neural.

Existe porque o professor pediu que o problema fosse resolvido com PDI, e que
biblioteca pronta de detecção entrasse só em último caso. Este módulo não usa
YOLO nem pesos de rede neural. Extrai medidas por operações clássicas de
imagem e escolhe pesos e limiar de uma combinação linear com rótulos de treino.

**A ideia.** Asfalto vazio é liso: tem pouca borda, pouca variação de
intensidade e pouca textura. Carro em cima é o contrário: tem contorno,
vidro, para-brisa, sombra dura, reflexo. Então a ocupação vira uma pergunta
sobre **quanta estrutura existe dentro do polígono da vaga**.

**O passo a passo, e o motivo de cada etapa:**

1. recorte do polígono por transformação de perspectiva, para que vaga ao
   fundo e vaga na frente virem retângulos do mesmo tamanho. Sem isso, a vaga
   distante tem menos pixels e sempre pareceria mais lisa;
2. escala de cinza;
3. **CLAHE** (equalização adaptativa limitada por contraste) para tirar a
   diferença entre dia claro e dia nublado. É o passo que faz o mesmo limiar
   valer de manhã e à tarde;
4. desfoque gaussiano leve, porque borda de ruído de compressão JPEG conta
   como borda e o PKLot é todo JPEG;
5. **Canny** com limiares derivados da mediana da imagem (limiar automático),
   não fixos;
6. fechamento morfológico, para unir contorno quebrado;
7. medidas: densidade de borda, variância do laplaciano, desvio do cinza,
   entropia do histograma e média do gradiente de Sobel.

**A decisão.** O limiar não é chutado: ele é calibrado em
`treino/calibrar_pdi.py` nas câmeras de treino. O runtime aceita somente
normalização global, com momentos de treino; a avaliação histórica por câmera
é transdutiva e não mede esse runtime. Há também o modo **Otsu por quadro**, que separa as
vagas do próprio quadro em dois grupos sem constante nenhuma e se adapta
sozinho à iluminação do momento.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometria import Poligono
from .tipos import Deteccao

LARGURA_PADRAO = 64
ALTURA_PADRAO = 64


@dataclass(frozen=True)
class Medidas:
    """As cinco medidas extraídas do recorte de uma vaga."""

    densidade_de_borda: float
    variancia_do_laplaciano: float
    desvio_do_cinza: float
    entropia: float
    gradiente_medio: float

    def como_vetor(self) -> np.ndarray:
        return np.array(
            [
                self.densidade_de_borda,
                self.variancia_do_laplaciano,
                self.desvio_do_cinza,
                self.entropia,
                self.gradiente_medio,
            ],
            dtype=np.float64,
        )

    def como_dict(self) -> dict[str, float]:
        return {
            "densidade_de_borda": self.densidade_de_borda,
            "variancia_do_laplaciano": self.variancia_do_laplaciano,
            "desvio_do_cinza": self.desvio_do_cinza,
            "entropia": self.entropia,
            "gradiente_medio": self.gradiente_medio,
        }


NOMES_DAS_MEDIDAS = (
    "densidade_de_borda",
    "variancia_do_laplaciano",
    "desvio_do_cinza",
    "entropia",
    "gradiente_medio",
)


def recortar_vaga(
    quadro: np.ndarray,
    contorno: Poligono,
    largura: int = LARGURA_PADRAO,
    altura: int = ALTURA_PADRAO,
) -> np.ndarray:
    """Retifica o polígono da vaga para um retângulo de tamanho fixo.

    A vaga no fundo do pátio ocupa poucos pixels e a da frente ocupa muitos.
    Medir densidade de borda sem corrigir isso faria a vaga distante parecer
    sempre mais lisa, ou seja, sempre livre. A transformação de perspectiva
    coloca todas no mesmo tamanho antes de qualquer medida.
    """
    pts = np.array(contorno, dtype=np.float32)
    if pts.shape[0] != 4:
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        recorte = quadro[max(y, 0) : y + h, max(x, 0) : x + w]
        if recorte.size == 0:
            return np.zeros((altura, largura, 3), dtype=np.uint8)
        return cv2.resize(recorte, (largura, altura))

    # ordena os quatro cantos: cima-esquerda, cima-direita, baixo-direita,
    # baixo-esquerda. Sem ordenar, a transformação espelha ou torce o recorte.
    soma = pts.sum(axis=1)
    dif = np.diff(pts, axis=1).ravel()
    origem = np.array(
        [
            pts[np.argmin(soma)],
            pts[np.argmin(dif)],
            pts[np.argmax(soma)],
            pts[np.argmax(dif)],
        ],
        dtype=np.float32,
    )
    destino = np.array(
        [[0, 0], [largura - 1, 0], [largura - 1, altura - 1], [0, altura - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(origem, destino)
    return cv2.warpPerspective(quadro, m, (largura, altura))


def _clahe(cinza: np.ndarray, limite: float = 2.0, grade: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=limite, tileGridSize=(grade, grade)).apply(cinza)


def _canny_automatico(img: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Limiares derivados da mediana, em vez de constantes fixas.

    Limiar fixo de Canny é a causa mais comum de pipeline que funciona numa
    foto e falha na seguinte: muda a iluminação, muda a faixa de gradiente.
    """
    mediana = float(np.median(img))
    baixo = int(max(0, (1.0 - sigma) * mediana))
    alto = int(min(255, (1.0 + sigma) * mediana))
    if alto <= baixo:
        baixo, alto = 50, 150
    return cv2.Canny(img, baixo, alto)


def medir(recorte: np.ndarray) -> Medidas:
    """Extrai as cinco medidas de um recorte já retificado."""
    if recorte.ndim == 3:
        cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    else:
        cinza = recorte
    cinza = _clahe(cinza)
    suave = cv2.GaussianBlur(cinza, (3, 3), 0)

    bordas = _canny_automatico(suave)
    nucleo = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bordas = cv2.morphologyEx(bordas, cv2.MORPH_CLOSE, nucleo)
    densidade = float(np.count_nonzero(bordas)) / bordas.size

    laplaciano = cv2.Laplacian(suave, cv2.CV_64F, ksize=3)
    var_lap = float(laplaciano.var())

    desvio = float(cinza.std())

    hist = cv2.calcHist([cinza], [0], None, [32], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    entropia = float(-(p * np.log2(p)).sum())

    gx = cv2.Sobel(suave, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(suave, cv2.CV_64F, 0, 1, ksize=3)
    gradiente = float(np.mean(np.hypot(gx, gy)))

    return Medidas(densidade, var_lap, desvio, entropia, gradiente)


def medir_vaga(quadro: np.ndarray, contorno: Poligono) -> Medidas:
    return medir(recortar_vaga(quadro, contorno))


# ------------------------------------------------------------------ decisao


@dataclass
class Calibracao:
    """Pesos e limiar escolhidos nas câmeras de treino.

    A regra é uma combinação linear das medidas padronizadas, com limiar. Não
    é uma rede neural nem um classificador treinado por biblioteca: são cinco
    constantes escolhidas por varredura, publicadas e auditáveis.
    """

    medias: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    )
    desvios: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    )
    pesos: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    )
    limiar: float = 0.0
    camera_de_treino: str = ""
    acuracia_de_treino: float = 0.0
    normalizacao: str = "global"

    def __post_init__(self):
        for nome in ("medias", "desvios", "pesos"):
            vetor = np.asarray(getattr(self, nome), dtype=float)
            if vetor.shape != (len(NOMES_DAS_MEDIDAS),) or not np.isfinite(vetor).all():
                raise ValueError(f"calibracao invalida: {nome} deve ter cinco valores finitos")
            setattr(self, nome, vetor)
        if (self.desvios < 0).any() or not np.isfinite(self.limiar):
            raise ValueError("calibracao invalida: desvios negativos ou limiar nao finito")
        if self.normalizacao not in ("global", "camera", "imagem"):
            raise ValueError(f"normalizacao desconhecida: {self.normalizacao}")

    def pontuar(self, m: Medidas) -> float:
        z = (m.como_vetor() - self.medias) / np.where(
            self.desvios == 0, 1.0, self.desvios
        )
        return float(self.pesos @ z)

    def ocupada(self, m: Medidas) -> bool:
        return self.pontuar(m) >= self.limiar

    def como_dict(self) -> dict:
        return {
            "medias": self.medias.tolist(),
            "desvios": self.desvios.tolist(),
            "pesos": self.pesos.tolist(),
            "limiar": self.limiar,
            "camera_de_treino": self.camera_de_treino,
            "acuracia_de_treino": self.acuracia_de_treino,
            "medidas": list(NOMES_DAS_MEDIDAS),
            "normalizacao": self.normalizacao,
        }

    @staticmethod
    def de_dict(d: dict) -> "Calibracao":
        if not isinstance(d, dict) or "normalizacao" not in d:
            raise ValueError(
                "calibracao sem normalizacao declarada; gere novamente com "
                "treino/calibrar_pdi.py --normalizacao global"
            )
        if d.get("medidas") != list(NOMES_DAS_MEDIDAS):
            raise ValueError("calibracao incompativel: ordem das cinco medidas incorreta")
        if not all(nome in d for nome in ("medias", "desvios", "pesos", "limiar")):
            raise ValueError("calibracao incompleta: faltam medias, desvios, pesos ou limiar")
        try:
            return Calibracao(
                medias=np.array(d["medias"], dtype=float),
                desvios=np.array(d["desvios"], dtype=float),
                pesos=np.array(d["pesos"], dtype=float),
                limiar=float(d["limiar"]),
                camera_de_treino=d.get("camera_de_treino", ""),
                acuracia_de_treino=float(d.get("acuracia_de_treino", 0.0)),
                normalizacao=d["normalizacao"],
            )
        except (TypeError, ValueError) as erro:
            raise ValueError(f"calibracao invalida: {erro}") from erro


def limiar_de_otsu(valores: np.ndarray) -> float:
    """Separa os valores em dois grupos maximizando a variância entre eles.

    É o modo sem constante nenhuma: num pátio com vagas livres e ocupadas no
    mesmo quadro, Otsu acha sozinho a fronteira e acompanha a iluminação do
    momento. Degrada quando o pátio está quase todo cheio ou quase todo vazio,
    porque aí não existem dois grupos, e isso está medido no relatório.
    """
    v = np.asarray(valores, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float(v.mean()) if v.size else 0.0
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return lo

    def _maior_lacuna() -> float:
        """Divide no maior salto entre valores consecutivos.

        Com poucas vagas no quadro, Otsu fica degenerado: ele opera sobre um
        histograma de 256 níveis e, com oito amostras, pode devolver um corte
        que deixa um dos grupos vazio. O maior salto resolve o mesmo problema
        sem depender do tamanho da amostra.
        """
        ordenado = np.sort(v)
        saltos = np.diff(ordenado)
        i = int(np.argmax(saltos))
        return float((ordenado[i] + ordenado[i + 1]) / 2.0)

    if v.size < 8:
        return _maior_lacuna()

    norm = np.clip(((v - lo) / (hi - lo) * 255.0).round(), 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # O corte do Otsu vive no espaco quantizado de 0 a 255. Converte-lo de
    # volta por regra de tres introduz erro de arredondamento que pode jogar um
    # valor para o grupo errado. O limiar devolvido e o ponto medio entre os
    # dois grupos NA ESCALA ORIGINAL, que nao tem esse problema.
    baixo = v[norm <= t]
    alto = v[norm > t]
    if baixo.size == 0 or alto.size == 0:
        return _maior_lacuna()
    return float((baixo.max() + alto.min()) / 2.0)


class DetectorClassico:
    """Decide a ocupação de cada vaga sem rede neural.

    Dois modos:

    - `calibrado`: usa a combinação linear e o limiar vindos das câmeras de
      treino, com normalização global. Pode executar em câmera nova, mas sua
      acurácia precisa ser medida;
    - `otsu`: separa as vagas do próprio quadro em dois grupos. Não precisa de
      constante, mas assume que o pátio tem vagas dos dois tipos no quadro.
    """

    nome = "classico"
    dispositivo = "cpu"

    def __init__(
        self,
        calibracao: Calibracao | None = None,
        modo: str = "calibrado",
        medida_para_otsu: str = "densidade_de_borda",
    ):
        self.calibracao = calibracao or Calibracao()
        if modo not in ("calibrado", "otsu"):
            raise ValueError(f"modo classico desconhecido: {modo}")
        if modo == "calibrado" and self.calibracao.normalizacao != "global":
            raise ValueError(
                f"calibracao com normalizacao {self.calibracao.normalizacao} "
                "incompativel com execucao online: faltam as estatisticas "
                "externas usadas na avaliacao; gere com --normalizacao global"
            )
        if medida_para_otsu not in NOMES_DAS_MEDIDAS:
            raise ValueError(f"medida para Otsu desconhecida: {medida_para_otsu}")
        self.modo = modo
        self.medida_para_otsu = medida_para_otsu
        self.ms_ultima_inferencia = 0.0

    def estados(
        self, quadro: np.ndarray, vagas
    ) -> tuple[list[bool], list[Medidas], float]:
        """Devolve (ocupada?, medidas, limiar usado) para cada vaga."""
        t0 = time.perf_counter()
        medidas = [medir_vaga(quadro, v.contorno) for v in vagas]
        if self.modo == "otsu":
            idx = NOMES_DAS_MEDIDAS.index(self.medida_para_otsu)
            vals = np.array([m.como_vetor()[idx] for m in medidas])
            t = limiar_de_otsu(vals)
            ocupadas = [bool(v >= t) for v in vals]
        else:
            t = self.calibracao.limiar
            ocupadas = [self.calibracao.ocupada(m) for m in medidas]
        self.ms_ultima_inferencia = (time.perf_counter() - t0) * 1000.0
        return ocupadas, medidas, float(t)

    def detectar(self, quadro: np.ndarray, vagas) -> list[Deteccao]:
        """Adapta a saída ao formato que o resto do sistema já consome.

        Cada vaga considerada ocupada vira uma detecção com a própria
        envolvente da vaga, para que `ocupacao.decidir_por_veiculo` funcione
        sem mudança.
        """
        ocupadas, medidas, _ = self.estados(quadro, vagas)
        saida: list[Deteccao] = []
        for vaga, ocup, m in zip(vagas, ocupadas, medidas):
            if not ocup:
                continue
            x1, y1, x2, y2 = vaga.envolvente
            conf = 1.0 / (1.0 + np.exp(-self.calibracao.pontuar(m))) if (
                self.modo == "calibrado"
            ) else 0.75
            saida.append(
                Deteccao(
                    caixa=(float(x1), float(y1), float(x2), float(y2)),
                    classe="veiculo_por_pdi",
                    confianca=float(conf),
                )
            )
        return saida
