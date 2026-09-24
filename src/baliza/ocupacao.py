"""Da lista de detecções para o estado de cada vaga."""

from __future__ import annotations

from .geometria import Caixa, fracao_coberta, iou
from .mapa import Mapa
from .tipos import Deteccao, Estado, Leitura, Resultado

# Cobertura mínima da vaga para chamá-la de ocupada. O valor foi escolhido
# varrendo o conjunto de validação (ver treino/calibrar.py), não no chute.
LIMIAR_COBERTURA = 0.30

# Abaixo do limiar mas acima desta fração, a leitura fica em dúvida em vez de
# virar "livre". Serve para não mandar o motorista a uma vaga meio encoberta.
ZONA_DUVIDA = 0.0


def _longe(caixa: Caixa, envolvente: Caixa) -> bool:
    """Descarte rápido antes de recortar polígono."""
    return (
        caixa[2] < envolvente[0]
        or caixa[0] > envolvente[2]
        or caixa[3] < envolvente[1]
        or caixa[1] > envolvente[3]
    )


def decidir_por_veiculo(
    mapa: Mapa,
    deteccoes: list[Deteccao],
    limiar: float = LIMIAR_COBERTURA,
    zona_duvida: float = ZONA_DUVIDA,
    ms_inferencia: float = 0.0,
) -> Resultado:
    """Vaga com veículo em cima está ocupada; vaga sem veículo está livre.

    O critério é quanto da área da vaga um veículo cobre, e não a IoU: um
    caminhão que transborda a vaga continua ocupando a vaga, e a IoU cairia
    justamente nesse caso.
    """
    leituras = []
    for vaga in mapa:
        envolvente = vaga.envolvente
        melhor_cobertura = 0.0
        melhor_confianca = 0.0
        melhor_classe = None
        for deteccao in deteccoes:
            if _longe(deteccao.caixa, envolvente):
                continue
            cobertura = fracao_coberta(vaga.contorno, deteccao.caixa)
            if cobertura > melhor_cobertura:
                melhor_cobertura = cobertura
                melhor_confianca = deteccao.confianca
                melhor_classe = deteccao.classe

        if melhor_cobertura >= limiar:
            estado = Estado.OCUPADA
        elif zona_duvida > 0 and melhor_cobertura >= limiar * (1 - zona_duvida):
            estado = Estado.SEM_LEITURA
        else:
            estado = Estado.LIVRE
            melhor_classe = None

        leituras.append(
            Leitura(
                vaga_id=vaga.id,
                estado=estado,
                confianca=melhor_confianca if estado is Estado.OCUPADA else 0.0,
                cobertura=melhor_cobertura,
                classe_detectada=melhor_classe,
            )
        )
    return Resultado(leituras=leituras, ms_inferencia=ms_inferencia)


def decidir_por_vaga(
    mapa: Mapa,
    deteccoes: list[Deteccao],
    iou_minima: float = 0.30,
    ms_inferencia: float = 0.0,
) -> Resultado:
    """Usa o modelo treinado, que já devolve a vaga rotulada.

    Cada detecção é casada com a vaga do mapa de maior IoU. Vaga sem
    detecção casada fica sem leitura: aqui o silêncio é falha do modelo, e
    não evidência de vaga vazia.
    """
    melhor_por_vaga: dict[str, tuple[float, Deteccao]] = {}
    for deteccao in deteccoes:
        melhor_id = None
        melhor_iou = iou_minima
        for vaga in mapa:
            if _longe(deteccao.caixa, vaga.envolvente):
                continue
            valor = iou(vaga.contorno, deteccao.caixa)
            if valor >= melhor_iou:
                melhor_iou = valor
                melhor_id = vaga.id
        if melhor_id is None:
            continue
        atual = melhor_por_vaga.get(melhor_id)
        if atual is None or deteccao.confianca > atual[1].confianca:
            melhor_por_vaga[melhor_id] = (melhor_iou, deteccao)

    leituras = []
    for vaga in mapa:
        casada = melhor_por_vaga.get(vaga.id)
        if casada is None:
            leituras.append(Leitura(vaga.id, Estado.SEM_LEITURA))
            continue
        valor_iou, deteccao = casada
        estado = Estado.OCUPADA if "ocupada" in deteccao.classe else Estado.LIVRE
        leituras.append(
            Leitura(
                vaga_id=vaga.id,
                estado=estado,
                confianca=deteccao.confianca,
                cobertura=valor_iou,
                classe_detectada=deteccao.classe,
            )
        )
    return Resultado(leituras=leituras, ms_inferencia=ms_inferencia)
