# Roteiro da demonstração

Tudo abaixo roda sem internet. Antes de sair de casa, confira os quatro itens
da última seção.

## 1. Abrir o painel (o caminho mais seguro)

```
cd baliza
.venv\Scripts\activate
streamlit run painel/app.py
```

O navegador abre sozinho em `localhost:8501`.

Na barra lateral:

1. **Câmera**: comece pela `UFPR04`. É a câmera próxima, onde o sistema acerta
   quase tudo, e serve para mostrar o sistema funcionando antes de mostrar onde
   ele sofre.
2. **Fonte**: "Vídeo de demonstração".
3. **Máximo de quadros**: 40 é suficiente. Cada quadro do vídeo é uma foto real
   tirada de cinco em cinco minutos, então 40 quadros são mais de três horas do
   pátio.
4. Clique em **Processar**.

O quadro anotado aparece e vai mudando, com o contador de livres por setor
acima. Ao final sai a curva de ocupação do dia e o botão de baixar o CSV.

## 2. A parte que vale a nota: mostrar o problema

Troque a câmera para **PUCPR**. O painel abre no detector que foi *medido*
naquela câmera, que ali é o treinado, e mostra abaixo do seletor qual arquivo
de pesos carregou. Processe assim primeiro: o pátio sai quase todo certo.

Agora troque o **Detector** para "veículos (YOLO11 do COCO, sem treino)" e
processe de novo, na mesma câmera. O pátio é grande e fotografado do décimo
andar, e várias vagas ocupadas aparecem em verde: o detector geral não enxerga
carro de vinte pixels. É a comparação que resume o projeto, e a ordem importa,
porque ela mostra primeiro o que funciona e só depois o que quebra.

Se perguntarem qual modelo treinado está rodando: são dois, e cada câmera usa o
que mediu melhor nela. UFPR04 e PUCPR usam `vagas-experimento.pt`, treinado só
em duas câmeras; a UFPR05 usa `vagas.pt`, treinado nas três, porque é a câmera
que o primeiro nunca viu. O painel escreve o nome do arquivo na tela.

Se quiser mostrar o meio termo, deixe o detector de veículos e mude "Janelas
deslizantes" para `3x2`: o quadro é recortado em seis pedaços antes da
detecção, o carro distante cresce em proporção e parte das vagas erradas se
corrige, ao custo de mais tempo por quadro.

## 3. Se preferir a linha de comando

```
baliza rodar demo/patios/ufpr04.mp4 --mapa demo/mapas/ufpr04.json --mostrar
baliza rodar demo/patios/pucpr.mp4  --mapa demo/mapas/pucpr.json  --mostrar
baliza rodar demo/fotos/ufpr05      --mapa demo/mapas/ufpr05.json --mostrar
```

Não precisa escolher o detector: cada mapa guarda qual deles funciona naquela
câmera, e o programa diz qual escolheu e por quê ao iniciar. Para forçar o
outro e mostrar a diferença, acrescente `--detector veiculos` ou
`--detector vagas`.

Espaço pausa, `q` encerra. Para gravar em vez de mostrar, troque `--mostrar`
por `--gravar saida.mp4`.

**Cuidado com o mapa errado.** O mapa de vagas vale só para a câmera que o
gerou. Rodar as fotos da PUCPR contra o mapa da UFPR04 não dá erro, dá número
sem sentido. É por isso que as fotos estão separadas em `demo/fotos/ufpr04`,
`demo/fotos/ufpr05` e `demo/fotos/pucpr`.

## 4. Se perguntarem de que dia é cada vídeo

UFPR04 é 17/01/2013, UFPR05 é 13/03/2013 e PUCPR é 29/10/2012. Os três são
dias ímpares, e isso não é coincidência: dia par é treino, então demonstrar num
dia par mostraria o modelo acertando o que decorou. É a mesma pergunta que o
projeto inteiro responde, só que aplicada à própria demonstração.

## 5. Câmera ao vivo, se houver tempo

Precisa de um mapa de vagas daquela cena, que não existe pronto. Só vale a pena
se você já tiver desenhado o mapa antes. Na dúvida, fique no vídeo gravado: é
o mesmo sistema, com a vantagem de que o resultado é conferível contra o
rótulo verdadeiro da base.

## 6. Conferir antes de sair de casa

- [ ] `pytest` passa inteiro
- [ ] `streamlit run painel/app.py` abre e processa a UFPR04 até o fim
- [ ] `modelos/vagas.pt` **e** `modelos/vagas-experimento.pt` estão no lugar:
      são dois modelos, e cada câmera carrega o seu
- [ ] as pastas `demo/patios` e `demo/mapas` têm os três pátios
- [ ] `demo/anotado` tem os quatro vídeos já pintados, que são o plano B se a
      máquina da sala não der conta de processar ao vivo

Se o notebook da apresentação não tiver GPU, o sistema roda em CPU do mesmo
jeito, só mais devagar. Reduza "Máximo de quadros" para 10 e avise que o
tempo por quadro na tela é o da CPU.
