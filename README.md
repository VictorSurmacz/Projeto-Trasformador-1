# Projeto-Transformador-1
Por motivos do GITHUB não suportar o Dataset, segue link do Drive com todos os scripts e PDF com métricas e a produção de dados de todo o Artigo.  
https://drive.google.com/drive/folders/1TOy1EuGY4Za_bO1R3Q0mgZpqJumXjAhr?usp=sharing

## Pasta Scripts

### scripts/runs
Contém as detecções de cada modelo, YOLO11n, YOLO11s, YOLO11m e RT-DETR. Família YOLO com Slicing.  
Dentro de cada pasta de runs temos as detecções, ou seja, o predict de cada modelo.  
- `rtdetr` — treinos do RT-DETR-L  
- `visualize_iou` — imagens com boxes classificadas como GT, TP, FP e FN  
- `yolo11m` — treinos do YOLO11m  
- `yolo11s` — treinos do YOLO11s  
- `yolo11n` — treinos do YOLO11n  

### Scripts .py

**convert.py**  
Responsável pela preparação do dataset. Converte as anotações originais do formato DatasetNinja (JSON) para o formato YOLO (TXT), onde cada linha representa um objeto com classe e coordenadas normalizadas do bounding box. Também realiza a divisão das imagens em conjuntos de treino, validação e teste, e gera o arquivo `data.yaml` com os caminhos e nome da classe utilizada.

**slice_dataset.py**  
Aplica a técnica de image slicing no dataset já convertido. Cada imagem é subdividida em 4 fatias com sobreposição de 20%, gerando subimagens menores que preservam mais detalhes visuais dos animais. As anotações são reprojetadas para as coordenadas de cada fatia e bounding boxes com menos de 50% de área dentro da fatia são descartados. O script substitui as imagens originais pelas fatias nos conjuntos de treino e validação, mantendo o conjunto de teste inalterado.

**train_yolo11n.py**  
Script de treinamento dos modelos. Recebe como parâmetros o modelo base (`--weights`), o arquivo de configuração do dataset (`--data`), hiperparâmetros como número de épocas, resolução de entrada, batch size, workers, patience e seed. Realiza fine-tuning a partir de pesos pré-treinados disponibilizados pelo framework Ultralytics e salva os pesos resultantes na pasta `runs/`.

**evaluate_yolo11n.py**  
Responsável pela avaliação dos modelos treinados. Roda inferência sobre os conjuntos de validação e teste, calculando métricas de detecção (Precisão, Recall, F1, mAP50) e de contagem (MAE e RMSE). Utiliza um algoritmo de pareamento guloso (greedy matching) com limiar de IoU ≥ 0,50 para classificar cada detecção como verdadeiro positivo (TP), falso positivo (FP) ou falso negativo (FN). Suporta o modo `--sliced`, que divide cada imagem em fatias durante a inferência e aplica NMS entre fatias para eliminar detecções duplicadas nas regiões de sobreposição. Os resultados são salvos em um arquivo JSON.

**visualiza_iou.py**  
Gera imagens com visualização das detecções classificadas por categoria. Para cada imagem do conjunto avaliado, desenha as caixas delimitadoras com cores distintas: azul para anotações reais (Ground Truth), verde para verdadeiros positivos (TP), vermelho para falsos positivos (FP) e magenta para falsos negativos (FN). Exibe também no canto da imagem um resumo com a contagem de TP, FP e FN, além das configurações de IoU e confiança utilizadas.

### PDF
- **Relatorio de teste.pdf** — primeiro relatório com os resultados dos treinos iniciais  
- **Tabelas + Slicing.pdf** — comparativo entre os primeiros treinos com e sem image slicing  
- **Tabelas e contagem.pdf** — relatório final com todos os modelos avaliados: família YOLO com e sem slicing e RT-DETR-L com resolução normal
