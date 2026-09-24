# Baliza

Diz quais vagas de um estacionamento estão livres, a partir do vídeo de uma
câmera fixa. Sem sensor no piso, sem cabo novo, sem obra: a câmera que já
está lá por causa da segurança enquadra dezenas de vagas ao mesmo tempo.

Trabalho da disciplina de Visão Computacional, feito por Fabrício Júnio
Almeida Dias, Camila Pereira Raimundo, Luan Miranda Padilha e Kauã Limão
Nunes.

```
baliza demonstracao                                   # roda o pacote que vem junto
baliza rodar demo/patios/ufpr04.mp4 --mapa demo/mapas/ufpr04.json --mostrar
baliza rodar 0 --mapa meu_patio.json --mostrar        # camera ao vivo
streamlit run painel/app.py                           # painel no navegador
```

Uma janela abre mostrando o pátio com cada vaga contornada, verde para livre
e vermelho para ocupada, e o placar no alto. Espaço pausa, `q` encerra.

## Os dois detectores, e por que existem dois

O sistema decide a ocupação de duas maneiras diferentes, e a escolha muda o
que o programa precisa saber de antemão.

**Detector de veículos** (`--detector veiculos`, é o padrão). Usa os pesos
gerais do YOLO11, treinados no COCO, e procura carro, moto, ônibus e
caminhão. Depois cruza cada caixa encontrada com o contorno das vagas: vaga
com veículo em cima está ocupada. Funciona em qualquer pátio sem treino
nenhum, e é o caminho para quem só quer apontar a câmera e ver.

**Detector de vagas** (`--detector vagas --pesos modelos/vagas.pt`). Usa os
pesos que treinamos no PKLot e procura a vaga em si, já classificada. Custa
um treino, e em troca enxerga o que o primeiro não enxerga: pátio grande
fotografado de longe, onde o carro tem vinte pixels e o detector geral
simplesmente não o vê.

A diferença entre os dois está medida em [docs/RESULTADOS.md](docs/RESULTADOS.md),
com número por estacionamento e por clima. O resumo:

| | Câmera do treino | Câmera nunca vista |
|---|---|---|
| Veículos, quadro inteiro | 70,2% | 98,1% |
| Veículos, janelas 3x2 | 87,7% | 99,2% |
| Vagas, treinado no PKLot | 100,0% | 25,6% |

Acurácia por vaga, contando como erro a vaga que o modelo não conseguiu ler.

O modelo treinado é imbatível no pátio em que treinou e inútil em outro: ele
não aprendeu a reconhecer vaga, aprendeu a geometria daqueles dois pátios. Em
oito fotos da câmera nunca vista ele devolveu 112 detecções para 319 vagas,
com IoU mediana de 0,05 contra o contorno verdadeiro. Por isso o padrão do
sistema é o detector de veículos, e não o treinado.

## O mapa de vagas

O sistema precisa saber onde fica cada vaga no quadro. Isso é um arquivo
JSON com o contorno de cada uma, desenhado uma vez por câmera:

```json
{
  "camera": "UFPR04",
  "vagas": [
    {"id": "1", "setor": "A", "contorno": [[720, 549], [805, 665], [715, 698], [644, 582]]}
  ]
}
```

Para as câmeras do PKLot o mapa sai pronto da própria base, que já traz o
contorno de cada vaga no XML:

```
baliza mapa-do-pklot dados/PKLot --estacionamento UFPR04 --saida demo/mapas/ufpr04.json
```

## Instalar

Python 3.12. A GPU não é obrigatória para rodar, só para treinar.

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e .[painel,dev]
```

O `pip install -e .` traz o PyTorch na versão de CPU. Para treinar, instale
antes a versão com CUDA que combina com a sua placa:

```
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

## Reproduzir os resultados

A base PKLot é pública e tem 4,9 GB. Um comando baixa, outro extrai:

```
curl -L -o dados/PKLot.tar.gz https://www.inf.ufpr.br/vri/databases/PKLot.tar.gz
tar -xzf dados/PKLot.tar.gz --wildcards "PKLot/PKLot/*" -C dados
```

Depois:

```
python treino/divisao.py dados/PKLot            # confere a divisao treino/validacao/teste
python treino/montar_demo.py --parte videos     # monta o pacote de demonstracao
python treino/calibrar.py --amostras 120        # escolhe o limiar de cobertura
python treino/exportar_yolo.py --por-dia 25     # PKLot -> formato YOLO
python treino/treinar.py --epocas 30            # ajuste fino do YOLO11n
python treino/avaliar.py --conjunto teste       # mede no estacionamento nunca visto
```

A divisão é **por câmera e por dia**, e não aleatória. A base fotografa o
mesmo pátio de cinco em cinco minutos: com divisão aleatória a foto das
10h05 cai no treino e a das 10h10 no teste, o número sobe e não quer dizer
nada. Treino e validação saem da PUCPR e da UFPR04, em dias alternados; o
teste é a UFPR05 inteira, uma câmera que o modelo nunca viu.

## Testes

```
pytest                      # a suite inteira, 110 testes
pytest -m "not lento"       # pula o que carrega o modelo de verdade
pytest --cov                # com cobertura
```

## Privacidade

O sistema grava a linha de estado da vaga, nunca o quadro. Foto de pátio
pode ter placa legível, e placa identifica pessoa por caminho indireto. O
vídeo anotado só existe quando alguém pede com `--gravar`.

## Base de dados

ALMEIDA, P. R. L.; OLIVEIRA, L. S.; BRITTO JR., A. S.; SILVA JR., E. J.;
KOERICH, A. L. **PKLot: a robust dataset for parking lot classification**.
Expert Systems with Applications, v. 42, n. 11, p. 4937-4949, 2015.
Licença CC BY 4.0.
