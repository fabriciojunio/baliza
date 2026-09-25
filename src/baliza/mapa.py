"""O mapa de vagas: onde fica cada vaga no quadro daquela câmera.

É desenhado uma vez por câmera e guardado em JSON. Sem ele o sistema conta
carros; com ele o sistema sabe dizer qual vaga está livre e em que setor.
"""

from __future__ import annotations

import json
from pathlib import Path

from .tipos import Vaga


class MapaInvalido(ValueError):
    pass


class Mapa:
    def __init__(self, camera: str, vagas: list[Vaga], largura: int = 0, altura: int = 0,
                 detector: str | None = None, pesos: str | None = None):
        if not vagas:
            raise MapaInvalido("mapa sem vaga nenhuma")
        vistos = set()
        for vaga in vagas:
            if vaga.id in vistos:
                raise MapaInvalido(f"vaga repetida no mapa: {vaga.id}")
            if len(vaga.contorno) < 3:
                raise MapaInvalido(f"vaga {vaga.id} tem menos de tres pontos")
            vistos.add(vaga.id)
        if detector not in (None, "veiculos", "vagas"):
            raise MapaInvalido(f"detector recomendado desconhecido: {detector}")
        self.camera = camera
        self.vagas = vagas
        self.largura = largura
        self.altura = altura
        # Qual detector funciona nesta câmera, medido e não chutado. O pátio
        # distante precisa do modelo treinado; o pátio que o modelo nunca viu
        # precisa do detector geral. Sem isso a demonstração abre no detector
        # errado e mostra vaga ocupada pintada de verde.
        self.detector = detector
        # E qual arquivo de pesos. Um modelo so serve as cameras que ele viu em
        # treino, entao a camera aponta para o modelo que a conhece.
        self.pesos = pesos

    def __len__(self) -> int:
        return len(self.vagas)

    def __iter__(self):
        return iter(self.vagas)

    @property
    def por_id(self) -> dict[str, Vaga]:
        return {vaga.id: vaga for vaga in self.vagas}

    @property
    def setores(self) -> list[str]:
        return sorted({vaga.setor for vaga in self.vagas})

    def para_dicionario(self) -> dict:
        return {
            "camera": self.camera,
            "largura": self.largura,
            "altura": self.altura,
            "detector_recomendado": self.detector,
            "pesos_recomendados": self.pesos,
            "vagas": [
                {
                    "id": vaga.id,
                    "setor": vaga.setor,
                    "contorno": [[round(x, 1), round(y, 1)] for x, y in vaga.contorno],
                }
                for vaga in self.vagas
            ],
        }

    def salvar(self, caminho: str | Path) -> None:
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(
            json.dumps(self.para_dicionario(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def de_dicionario(cls, dados: dict) -> "Mapa":
        try:
            vagas = [
                Vaga(
                    id=str(item["id"]),
                    contorno=[(float(p[0]), float(p[1])) for p in item["contorno"]],
                    setor=str(item.get("setor", "A")),
                )
                for item in dados["vagas"]
            ]
        except (KeyError, TypeError, IndexError) as erro:
            raise MapaInvalido(f"mapa malformado: {erro}") from erro
        return cls(
            camera=str(dados.get("camera", "camera")),
            vagas=vagas,
            largura=int(dados.get("largura", 0)),
            altura=int(dados.get("altura", 0)),
            detector=dados.get("detector_recomendado"),
            pesos=dados.get("pesos_recomendados"),
        )

    @classmethod
    def carregar(cls, caminho: str | Path) -> "Mapa":
        caminho = Path(caminho)
        if not caminho.exists():
            raise FileNotFoundError(f"mapa nao encontrado: {caminho}")
        try:
            dados = json.loads(caminho.read_text(encoding="utf-8"))
        except json.JSONDecodeError as erro:
            raise MapaInvalido(f"mapa nao e JSON valido: {erro}") from erro
        return cls.de_dicionario(dados)


def dividir_em_setores(mapa: Mapa, colunas: int = 2) -> Mapa:
    """Divide as vagas em setores por posição horizontal no quadro.

    O PKLot não traz setor, e o painel fica mais útil dizendo "3 livres no
    setor B" do que só um total. A divisão por faixa vertical do quadro é
    grosseira de propósito: é o que dá para inferir sem planta do pátio.
    """
    if colunas < 1:
        raise ValueError("colunas precisa ser pelo menos 1")
    xs = sorted(vaga.centro[0] for vaga in mapa.vagas)
    if not xs:
        return mapa
    limites = [xs[int(len(xs) * (i + 1) / colunas) - 1] for i in range(colunas)]
    nomes = [chr(ord("A") + i) for i in range(colunas)]

    novas = []
    for vaga in mapa.vagas:
        x = vaga.centro[0]
        setor = nomes[-1]
        for nome, limite in zip(nomes, limites):
            if x <= limite:
                setor = nome
                break
        novas.append(Vaga(id=vaga.id, contorno=vaga.contorno, setor=setor))
    return Mapa(mapa.camera, novas, mapa.largura, mapa.altura, mapa.detector, mapa.pesos)
