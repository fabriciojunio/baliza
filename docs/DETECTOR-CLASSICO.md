# O detector sem rede neural

O professor pediu que o problema fosse resolvido com processamento digital de
imagens, e que biblioteca pronta de detecção entrasse só em último caso. Este
documento descreve um terceiro detector que não usa YOLO nem rede neural,
mas escolhe pesos e limiar de uma combinação linear com rótulos de treino, e o quanto ele
perde para o que já estava no projeto.

Os resultados históricos são **86,5% contra 97,5% do YOLO** em uma comparação
offline transdutiva: a PDI usa estatísticas de toda a amostra da câmera de
teste sem seus rótulos. Não são resultados do runtime online. Os 155 vezes
referem-se ao tempo de extração das medidas, sem o custo de adaptação por
câmera nem a latência de esperar os quadros futuros.

Em 11/10/2026 foram corrigidos o caminho do JSON na CLI, o repasse das vagas
e a decisão direta por polígono. O runtime agora recusa a calibração histórica
por câmera, pois não possui os momentos externos usados na avaliação. Para
rodar com momentos estimados apenas no treino:

```
python treino/calibrar_pdi.py --normalizacao global --saida resultados/pdi-online
baliza rodar demo/patios/ufpr04.mp4 --mapa demo/mapas/ufpr04.json --detector classico --pesos resultados/pdi-online/pdi_calibracao.json --mostrar
```

## A ideia

Asfalto vazio é liso: tem pouca borda, pouca variação de intensidade e pouca
textura. Carro em cima é o contrário: tem contorno, vidro, para-brisa, sombra
dura e reflexo. Então a ocupação vira uma pergunta sobre **quanta estrutura
existe dentro do polígono da vaga**.

## O passo a passo, e o motivo de cada etapa

Nenhuma etapa está aí por enfeite. Cada uma resolve um problema que apareceu
quando a anterior não bastava.

1. **Recorte por transformação de perspectiva**, levando o polígono da vaga
   para um retângulo de 64 por 64. A vaga no fundo do pátio ocupa poucos
   pixels e a da frente ocupa muitos; medir densidade de borda sem corrigir
   isso faria a vaga distante parecer sempre mais lisa, ou seja, sempre livre.
2. **Escala de cinza.** A cor do carro não ajuda a decidir se há carro.
3. **CLAHE**, equalização adaptativa limitada por contraste, para reduzir
   diferenças locais de iluminação. Isso não garante que o mesmo limiar
   funcione de manhã e à tarde ou em outro pátio.
4. **Desfoque gaussiano leve.** O PKLot é todo JPEG, e borda de artefato de
   compressão conta como borda.
5. **Canny com limiares derivados da mediana** da própria imagem, não fixos.
   Limiar fixo de Canny é a causa mais comum de pipeline que funciona numa
   foto e falha na seguinte.
6. **Fechamento morfológico**, para unir contorno quebrado.
7. **Cinco medidas**: densidade de borda, variância do laplaciano, desvio do
   cinza, entropia do histograma e média do gradiente de Sobel.

## A decisão, e por que ela não é um chute

O limiar não é escolhido no olho. Ele é calibrado em
`treino/calibrar_pdi.py`, e o protocolo é o mesmo do resto do projeto:
**calibra com os rótulos de duas câmeras e mede na terceira.**
No modo histórico por câmera, o cálculo dos momentos acessa as medidas da
terceira câmera durante a avaliação; ela é inédita apenas quanto aos rótulos.
Medir na mesma câmera em que se calibrou responde uma pergunta fácil e
diferente da que importa.

A regra é uma combinação linear de três das cinco medidas, com um limiar. São
quatro constantes, publicadas em `resultados/pdi_calibracao.json`, que dá para
conferir à mão. A busca que as escolheu é exaustiva sobre um espaço pequeno e
declarado: todos os subconjuntos de até três medidas, com peso mais um ou
menos um.

Duas escolhas mudaram o resultado de forma grande e merecem registro, porque
as duas foram erro antes de virarem acerto.

**Padronizar por câmera, não globalmente.** A escala absoluta das medidas
depende da câmera: altura, distância focal, tamanho da vaga em pixels e
exposição. Com padronização global, o detector transferia mal: precisão 0,98 e
revocação 0,48, ou seja, quase nunca errava ao dizer "ocupada" e deixava metade
das vagas ocupadas passar como livres. Padronizar dentro de cada câmera
transforma a pergunta "esta vaga tem muita estrutura?" em "esta vaga tem mais
estrutura que as outras desta câmera?", que é a pergunta que generaliza. É
adaptação de domínio por casamento de momentos, e **não usa rótulo nenhum da
câmera nova**. Usa, porém, todos os quadros amostrados, inclusive os futuros.
Esses momentos não estão disponíveis assim que a câmera liga. Estimar momentos
em uma janela passada seria outro protocolo, ainda sem validação neste projeto.

Padronizar por **imagem** em vez de por câmera foi testado e ficou pior (0,58):
com poucas dezenas de vagas por quadro, o desvio é ruidoso demais.

**Escolher o limiar por F1, não por acurácia.** Acurácia escolhe um limiar
conservador demais quando as classes são desbalanceadas, e foi exatamente o que
aconteceu na primeira versão.

## O que foi medido

Protocolo histórico: rótulos de treino de PUCPR e UFPR04, medido na UFPR05
com normalização transdutiva por câmera. 28.981 vagas medidas no total, 7.158
delas no teste. Os números abaixo não foram recalculados nesta correção. O
campo histórico `acuracia_de_treino` armazenava o F1 escolhido pela busca;
o script corrigido passa a calcular a acurácia efetiva para esse campo.

### Poder de separação de cada medida, sozinha

| Medida | AUC |
|---|---|
| desvio do cinza | 0,902 |
| entropia do histograma | 0,898 |
| gradiente médio de Sobel | 0,862 |
| variância do laplaciano | 0,804 |
| densidade de borda | 0,798 |

A densidade de borda, que é a medida mais óbvia e a que a maioria dos tutoriais
usa sozinha, é a **pior** das cinco. O desvio do cinza, que é quase trivial de
calcular, é a melhor.

### Câmera com rótulos separados (UFPR05), 7.158 vagas

| | valor |
|---|---|
| acurácia | 0,821 |
| precisão | 0,990 |
| revocação | 0,739 |
| F1 | 0,846 |

Por clima: nublado 0,834, sol 0,851, **chuva 0,777**. A chuva é o pior caso, e
faz sentido: a água no asfalto cria reflexo e textura onde deveria haver
superfície lisa, que é exatamente o sinal que o detector usa.

O perfil do erro é assimétrico e isso importa para o uso: com precisão de
0,990, quando ele diz "ocupada" está certo quase sempre. O que ele erra é
deixar vaga ocupada passar por livre. Para quem procura vaga, esse é o erro
chato: manda o motorista a uma vaga que não existe.

### Modo Otsu, sem constante nenhuma

Há um segundo modo que não usa calibração: separa as vagas do próprio quadro em
dois grupos por Otsu. Deu **0,593**, e a razão está declarada: ele assume que o
quadro tem vagas dos dois tipos. Em 80 das imagens do teste o pátio estava
quase todo cheio ou quase todo vazio, e aí não existem dois grupos para separar.
Ficou no código como demonstração honesta de um caminho que não funciona bem.

## Contra o YOLO, nas mesmas imagens

A comparação só vale se os dois virem exatamente o mesmo conjunto. 60 imagens
da UFPR05, nenhuma vista por nenhum dos dois em treino. O YOLO roda com
`yolo11n.pt` do COCO a 1280 pixels, que é a configuração que
`treino/avaliar.py` usa.

| | acurácia | precisão | revocação | F1 | ms por imagem |
|---|---|---|---|---|---|
| clássico (PDI) | 0,865 | 0,994 | 0,787 | 0,878 | **34** |
| YOLO11n (COCO) | **0,975** | 0,996 | 0,963 | 0,979 | 5.266 |

Os dois concordam em 87,0% das vagas.

**O YOLO ganha por 10,9 pontos.** Não adianta disfarçar: para acertar a
ocupação, a rede treinada é melhor, e a diferença está quase toda na revocação
(0,963 contra 0,787). Ela enxerga carro escuro em sombra, carro parcialmente
encoberto e carro pequeno no fundo, situações em que a textura dentro do
polígono não é suficiente.

O que o clássico tem a favor, e não é pouco:

- **155 vezes mais rápido na CPU**, 34 ms contra 5,3 s por imagem. O número do
  YOLO é de CPU; com GPU ele cai muito, mas aí a conta deixa de ser só de
  software;
- **não carrega pesos de rede neural.** Não há download de modelo nem dependência de
  framework de aprendizado profundo para essas operações;
- **cada etapa é inspecionável.** Dá para abrir o recorte, ver a saída do Canny
  e entender por que uma vaga foi classificada daquele jeito. A rede responde
  com uma caixa e uma confiança;
- **precisão praticamente igual** (0,994 contra 0,996): quando ele afirma
  "ocupada", acerta tanto quanto a rede.

Onde isso muda a escolha: num equipamento de borda sem GPU, processando várias
câmeras, 34 ms por imagem contra 5 segundos é a diferença entre rodar e não
rodar. A perda de revocação vira um parâmetro de produto, não um defeito, se o
sistema for usado para contar ocupação agregada em vez de apontar uma vaga
específica.

## Reproduzir

```
python treino/calibrar_pdi.py --limite-por-camera 60 --normalizacao camera --saida resultados/pdi-offline
python treino/comparar_pdi_e_yolo.py --limite-por-clima 20 --tamanho 1280 --saida resultados/pdi-offline
```

Saídas na pasta indicada: `pdi_calibracao.json`, `pdi_resultado.json` e
`comparacao_pdi_yolo.json`. A comparação agora respeita a normalização
declarada no JSON e usa o mesmo desvio amostral (`ddof=1`) da calibração por
câmera/imagem. Isso exige recalcular os números ao reproduzir. O cache
`pdi_medidas.csv`, quando existente na pasta de saída, é reutilizado; altere
a pasta de saída para medir uma amostra nova.

## Limites declarados

- a comparação usa `yolo11n.pt`, o menor da família. Modelo maior aumentaria a
  distância;
- a medição é sobre as imagens do PKLot, não sobre vídeo contínuo. Em vídeo o
  clássico poderia melhorar com um modelo de fundo por mediana temporal, que
  não foi implementado;
- a padronização histórica por câmera usa a amostra de teste inteira.
  O runtime não a implementa nem promete primeira decisão confiável após
  alguns quadros; aceita somente normalização global com momentos de treino;
- chuva é o pior caso e está medido, não contornado.

A base e os XMLs de referência são descritos pelo [Laboratório VRI da
UFPR](https://web.inf.ufpr.br/vri/databases/parking-lot-database/). Essa fonte
documenta câmeras, climas, datas e rótulos; ela não valida os resultados do
detector deste projeto.
