# Resultados medidos

Gerado por `treino/relatorio.py` em 24/09/2026, com 60 fotos por conjunto e entrada de 1280 px.

A divisão é por câmera e por dia (ver `treino/divisao.py`). O conjunto de
**teste é a UFPR05 inteira**, uma câmera que nenhum modelo viu em treino.

Os dois erros aparecem separados porque não custam o mesmo: dizer *livre*
numa vaga ocupada manda o motorista ao lugar errado, dizer *ocupada* numa
vaga livre apenas esconde uma vaga que existia.

**Acurácia (lidas)** ignora a vaga que o modelo não leu; **acurácia (todas)**
conta a vaga não lida como erro. A distância entre as duas colunas é onde
mora a história deste projeto.

### Validação (PUCPR e UFPR04, dias ímpares)

| Configuração | Acurácia (lidas) | Acurácia (todas) | Falso livre | Falso ocupado | Sem leitura | ms/quadro |
|---|---|---|---|---|---|---|
| Veículos, quadro inteiro | 70.2% | 70.2% | 60.8% | 0.6% | 0.0% | 72 |
| Veículos, janelas 3x2 | 87.7% | 87.7% | 24.7% | 0.6% | 0.0% | 200 |
| Vagas, treinado no PKLot | 100.0% | 100.0% | 0.1% | 0.0% | 0.0% | 26 |

### Teste (UFPR05, câmera nunca vista)

| Configuração | Acurácia (lidas) | Acurácia (todas) | Falso livre | Falso ocupado | Sem leitura | ms/quadro |
|---|---|---|---|---|---|---|
| Veículos, quadro inteiro | 98.1% | 98.1% | 2.9% | 0.4% | 0.0% | 27 |
| Veículos, janelas 3x2 | 99.2% | 99.2% | 0.9% | 0.4% | 0.0% | 202 |
| Vagas, treinado no PKLot | 99.5% | 25.6% | 0.0% | 1.6% | 74.3% | 25 |

### Acurácia por estacionamento, na validação

| Configuração | PUCPR | UFPR04 |
|---|---|---|
| Veículos, quadro inteiro | 62.8% | 98.2% |
| Veículos, janelas 3x2 | 85.0% | 98.1% |
| Vagas, treinado no PKLot | 99.9% | 100.0% |

### Acurácia por clima, no teste

| Configuração | Sunny | Cloudy | Rainy |
|---|---|---|---|
| Veículos, quadro inteiro | 97.5% | 99.0% | 98.4% |
| Veículos, janelas 3x2 | 99.2% | 99.6% | 98.8% |
| Vagas, treinado no PKLot | 27.9% | 23.3% | 21.3% |

## O segundo modelo, o que roda na demonstração

As tabelas acima são do `vagas-experimento.pt`, treinado só em PUCPR e UFPR04,
que é o modelo que responde à pergunta científica. Existe um segundo,
`vagas.pt`, treinado nas três câmeras em 18 épocas, e ele existe por um motivo
prático: numa câmera nunca vista o primeiro não serve, e a demonstração precisa
funcionar nas três.

Medidos nos dias de demonstração, todos ímpares e portanto fora do treino
(`resultados/validacao_final.json`):

| Câmera | `vagas-experimento.pt` | `vagas.pt` | Usa |
|---|---|---|---|
| UFPR04 | 99,9% | 99,9% | experimento |
| UFPR05 | 32,8% | 100,0% | vagas |
| PUCPR | 98,0% | 66,3% | experimento |

A PUCPR é o caso interessante: o modelo que viu **mais** câmeras vai **pior**
nela, porque treinou 18 épocas contra 30 do outro. Mais dado não compensa treino
mais curto, e por isso a escolha é por câmera e medida, não por intuição.

Duas limitações honestas destes números:

- Para o `vagas.pt`, os dias ímpares serviram de validação durante o treino, o
  que escolhe a época gravada. Não treinou peso neles, mas também não é um
  conjunto de teste virgem, e chamar de teste seria exagero.
- O XML do PKLot às vezes deixa de anotar uma vaga num quadro, e leitura sem
  gabarito não entra no denominador. São 0,99% dos pares na UFPR04 e 0,04% nas
  outras duas, registrados no campo `sem_gabarito`.

## Como reproduzir

```
python treino/exportar_yolo.py --por-dia 25
python treino/treinar.py --epocas 30 --tamanho 1280 --lote 4
python treino/relatorio.py --amostras 60
```
