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

## Como reproduzir

```
python treino/exportar_yolo.py --por-dia 25
python treino/treinar.py --epocas 30 --tamanho 1280 --lote 4
python treino/relatorio.py --amostras 60
```
