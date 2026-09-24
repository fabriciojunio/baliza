"""Painel do Baliza.

    streamlit run painel/app.py

Mostra o pátio com as vagas pintadas, o total de livres por setor e a curva
de ocupação do dia. Ele só chama o núcleo: nenhuma regra de decisão mora
aqui dentro.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from baliza.captura import abrir_fonte  # noqa: E402
from baliza.desenho import anotar, legenda  # noqa: E402
from baliza.deteccao import abrir_detector  # noqa: E402
from baliza.mapa import Mapa  # noqa: E402
from baliza.registro import Registro  # noqa: E402
from baliza.sistema import Baliza  # noqa: E402

DEMO = RAIZ / "demo"
PESOS_TREINADOS = RAIZ / "modelos" / "vagas.pt"

st.set_page_config(page_title="Baliza", page_icon="P", layout="wide")


@st.cache_resource(show_spinner=False)
def carregar_sistema(caminho_mapa: str, modo: str, pesos: str | None,
                     tamanho: int, limiar: float, janelas: str) -> Baliza:
    grade = None
    if janelas != "quadro inteiro":
        colunas, linhas = janelas.split("x")
        grade = (int(colunas), int(linhas))
    mapa = Mapa.carregar(caminho_mapa)
    detector = abrir_detector(modo, pesos, tamanho=tamanho, janelas=grade)
    return Baliza(mapa, detector, limiar=limiar)


def cartoes(resultado, mapa) -> None:
    colunas = st.columns(3 + len(mapa.setores))
    colunas[0].metric("Vagas livres", resultado.livres)
    colunas[1].metric("Ocupadas", resultado.ocupadas)
    colunas[2].metric("Sem leitura", resultado.sem_leitura)
    for coluna, (setor, (livres, total)) in zip(
        colunas[3:], resultado.por_setor(mapa.por_id).items()
    ):
        coluna.metric(f"Setor {setor}", f"{livres}/{total}")


def curva(banco: Path, camera: str) -> None:
    if not banco.exists():
        return
    with Registro(banco) as registro:
        serie = registro.serie_ocupacao(camera)
    if not serie:
        return
    tabela = pd.DataFrame(serie, columns=["instante", "ocupadas", "total"])
    tabela["ocupacao (%)"] = (tabela["ocupadas"] / tabela["total"] * 100).round(1)
    # O eixo e a ordem da leitura, e nao o relogio: processar um video de
    # arquivo leva um minuto, e carimbar o grafico com a hora do processamento
    # daria a entender que o patio encheu em um minuto. O carimbo real fica no
    # CSV, que e onde ele serve para alguma coisa.
    tabela.insert(0, "leitura", range(1, len(tabela) + 1))
    st.subheader("Ocupação ao longo da sequência")
    st.line_chart(tabela.set_index("leitura")[["ocupacao (%)"]], height=240)
    st.caption(
        f"{len(tabela)} leituras, pico de {tabela['ocupadas'].max()} vagas ocupadas de"
        f" {int(tabela['total'].iloc[0])}. Nos vídeos de demonstração cada leitura é uma"
        f" foto tirada de 5 em 5 minutos, então a curva cobre o dia inteiro."
    )
    st.download_button(
        "Baixar histórico em CSV",
        tabela.to_csv(index=False).encode("utf-8"),
        file_name=f"ocupacao_{camera}.csv",
        mime="text/csv",
    )


st.title("Baliza")
st.caption("Ocupação de vagas de estacionamento a partir de câmera fixa")

mapas = sorted(DEMO.glob("mapas/*.json"))
if not mapas:
    st.error("Nenhum mapa de vagas em demo/mapas. Rode `python treino/montar_demo.py`.")
    st.stop()

with st.sidebar:
    st.header("Configuração")
    escolhido = st.selectbox("Câmera", mapas, format_func=lambda p: p.stem.upper())

    tem_treinado = PESOS_TREINADOS.exists()
    opcoes = ["veículos (YOLO11 do COCO, sem treino)"]
    if tem_treinado:
        opcoes.insert(0, "vagas (YOLO11 treinado no PKLot)")
    escolha_modelo = st.radio("Detector", opcoes)
    modo = "vagas" if escolha_modelo.startswith("vagas") else "veiculos"
    pesos = str(PESOS_TREINADOS) if modo == "vagas" else str(RAIZ / "modelos" / "yolo11n.pt")

    tamanho = st.select_slider("Resolução de entrada", [640, 960, 1280, 1600], value=1280)
    janelas = st.selectbox(
        "Janelas deslizantes", ["quadro inteiro", "2x2", "3x2"],
        help="Recorta o quadro antes de detectar. Ajuda quando a câmera é distante.",
    )
    limiar = 0.30
    if modo == "veiculos":
        limiar = st.slider("Cobertura mínima da vaga", 0.05, 0.80, 0.30, 0.05)

    st.divider()
    origem = st.radio("Fonte", ["Vídeo de demonstração", "Enviar imagem", "Enviar vídeo"])
    quadros_max = st.number_input("Máximo de quadros", 1, 500, 40)

mapa = Mapa.carregar(escolhido)
st.write(
    f"**{mapa.camera}** · {len(mapa)} vagas · setores {', '.join(mapa.setores)}"
)

sistema = carregar_sistema(str(escolhido), modo, pesos, tamanho, limiar, janelas)
sistema.mapa = mapa

alvo = None
if origem == "Vídeo de demonstração":
    video = DEMO / "patios" / f"{escolhido.stem}.mp4"
    alvo = str(video) if video.exists() else None
    if alvo is None:
        st.warning("Vídeo de demonstração não encontrado.")
else:
    enviado = st.file_uploader(
        "Arquivo", type=["jpg", "jpeg", "png", "mp4", "avi", "mkv"]
    )
    if enviado is not None:
        sufixo = Path(enviado.name).suffix
        temporario = tempfile.NamedTemporaryFile(delete=False, suffix=sufixo)
        temporario.write(enviado.read())
        temporario.close()
        alvo = temporario.name

if alvo and st.button("Processar", type="primary"):
    painel_imagem = st.empty()
    painel_cartoes = st.empty()
    barra = st.progress(0.0)
    banco = DEMO / "historico" / f"{escolhido.stem}_painel.db"
    banco.parent.mkdir(parents=True, exist_ok=True)
    if banco.exists():
        banco.unlink()

    registro = Registro(banco)
    sistema.registro = registro
    fonte = abrir_fonte(alvo)
    total = min(int(quadros_max), fonte.total_quadros or int(quadros_max))
    lidos = 0

    for indice, quadro in fonte.quadros():
        resultado = sistema.processar(quadro)
        registro.gravar(mapa.camera, resultado)
        anotado = legenda(anotar(quadro, mapa, resultado))
        painel_imagem.image(cv2.cvtColor(anotado, cv2.COLOR_BGR2RGB), use_container_width=True)
        with painel_cartoes.container():
            cartoes(resultado, mapa)
        lidos += 1
        barra.progress(min(1.0, lidos / total))
        if lidos >= quadros_max:
            break

    fonte.fechar()
    registro.fechar()
    barra.empty()
    st.success(f"{lidos} quadros processados.")
    curva(banco, mapa.camera)
else:
    historico = DEMO / "historico" / f"{escolhido.stem}.db"
    if historico.exists():
        st.info("Mostrando o histórico da última execução gravada.")
        curva(historico, mapa.camera)
