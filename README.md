# Segmentação multimodal de dentes: radiografia panorâmica + escaneamento intraoral

Código do Trabalho de Conclusão de Curso de **Thiago Kei Shimasaki**.

O sistema combina a radiografia panorâmica (2D) com a malha do escaneamento intraoral (3D) para segmentar e numerar
os dentes (notação FDI) e estimar o eixo longo e os pontos de referência (landmarks) de cada dente.

```
panorâmica ──► Mask R-CNN ──► um token por dente ──────────────────────┐
                                                                       ├─► fusão residual ─► graph-cut ─► landmarks
malha 3D ──► decimação (8k células) ──► MeshSegNet só 3D (rascunho) ───┘
                                └─► registro panorâmica ↔ malha (casa cada dente da imagem com a sua região na malha)
```

## Estrutura

| Pasta | Arquivo | O que faz |
|---|---|---|
| `00_comum/` | `caminhos.py` | Onde ficam dados, pesos e resultados (variável `TCC_DADOS`) |
| | `malha.py` | Decimação, features do MeshSegNet, graph-cut, remoção de ilhas, volta à malha cheia |
| | `meshsegnet.py` | MeshSegNet público (Lian et al., MIT) |
| `01_dados/` | `preprocessar_panoramica.py` | Panorâmica bruta → recorte, filtro, CLAHE, 512×512 |
| | `escolher_panoramica.py` | Escolhe a panorâmica certa quando o caso tem várias candidatas |
| | `preparar_fusao.py` | Malha `.vtp` com FDI → malha decimada com rótulo por célula + manifesto |
| | `redividir_novos.py` | Divisão treino/val/teste (75/10/15) da ampliação |
| | `teeth3ds.py` | Teeth3DS+ / 3DTeethLand (público): baixar, compactar, preparar |
| `02_detector_2d/` | `train_maskrcnn_dentex.py` | Treino do Mask R-CNN no DENTEX (detector D1, Kaggle) |
| | `tcc-xray-v2.ipynb` | Pré-rótulos e retreino com flip que troca o lado (detector D2, Kaggle) |
| | `pipeline_2d.py` | Inferência do detector e extração dos tokens por dente (módulo) |
| | `montar_cvat_teste_multimodal.py` | Pacote do conjunto de teste para anotar o GT 2D no CVAT |
| | `extrair_features_2d.py` | Tokens 2D de todos os casos |
| `03_registro/` | `registro.py` | Registro panorâmica ↔ malha (módulo, usado no treino e na inferência) |
| | `registro_fusao.py` | Registra todas as arcadas e mede o acerto no teste |
| | `montar_limpo.py` | Monta o conjunto final com o rascunho 3D próprio |
| `04_segmentacao_fusao/` | `dados_fusao.py`, `modelo_fusao.py` | Dataset e modelo (MeshSegNet + fusão por atenção) |
| | `treinar_fusao.py` | Treino e avaliação |
| | `exportar_predicoes.py` | Rascunhos 3D e probabilidades por célula de um run |
| | `escolher_lambda.py` | λ do graph-cut escolhido na validação |
| | `rodar_g5.sh` | **O experimento final inteiro** |
| `05_landmarks/` | `tabela_eixos.py`, `recortes_dentes.py` | Eixos do planejador e recortes por dente (dados próprios) |
| | `recortes_3dteethland.py` | Recortes por dente do 3DTeethLand |
| | `modelo_landmarks.py`, `treinar_landmarks.py` | Modelo de landmarks por dente e treino |
| | `landmarks.py` | Inferência dos landmarks (módulo) |
| | `teste_3dteethland.py`, `metricas_3dteethland.py` | Avaliação no teste oficial do 3DTeethLand |
| | `rodar_landmarks.sh` | Treino das 3 sementes + ablações |
| `06_avaliacao/` | `avaliar_g5.py` | Métricas finais: McNemar FR × SC, por tipo de dente, mIoU, acurácia |
| `07_inferencia/` | `rodar.py`, `orientacao.py` | Malha crua + panorâmica → segmentação, FDI e landmarks (ver `MANUAL.md`) |

Os scripts importam uns aos outros entre pastas. Cada um coloca sozinho as pastas `0*_*` no `sys.path`, então
podem ser chamados de qualquer diretório. Todos os comandos abaixo partem da raiz do repositório.

## Instalação

Python ≥ 3.10, PyTorch ≥ 2.0 (com CUDA para treinar) e:
```
pip install -r requirements.txt
```
O Detectron2 é instalado à parte: <https://github.com/facebookresearch/detectron2>.

## Dados

Os exames são de pacientes e **não estão no repositório**. A estrutura esperada está em [`DADOS.md`](DADOS.md).
Todos os caminhos partem da variável de ambiente `TCC_DADOS` (padrão: a pasta pessoal):
```
export TCC_DADOS=/caminho/dos/dados
```

## Passo a passo

Os passos 1–3 preparam os dados e rodam em CPU; os treinos (passos 2, 4 e 5) precisam de GPU: o detector 2D foi
treinado no Kaggle (T4) e a segmentação e os landmarks numa máquina virtual com NVIDIA L4.
**Atalho:** com `fusao_dados/` e `features_2d_flipswap/` já prontos, comece no passo 4.

### 1. Panorâmicas

| | |
|---|---|
| **Entrada** | panorâmicas brutas, uma por caso (`<case_id>.png/.jpg`) |
| **Saída** | `panoramicas/<lote>/processados/<case_id>.png` (512×512) |

```
python 01_dados/preprocessar_panoramica.py --entrada <pasta das brutas> --saida $TCC_DADOS/panoramicas/<lote>
```
Recorta a radiografia (tira cabeçalho, laudo e tarjas com dados do paciente), descarta o que não parece panorâmica
(`rejeitados/` + `auditoria.csv`), aplica CLAHE e redimensiona para 512×512 sem distorcer. As reprovadas e as
duvidosas passam por **revisão manual**. Quando um caso tem várias candidatas, `escolher_panoramica.py` roda o
detector em cada uma e fica com a que mostra as duas arcadas com mais dentes.

### 2. Detector 2D e tokens da panorâmica

| | |
|---|---|
| **Entrada** | DENTEX (público) e as panorâmicas do passo 1 |
| **Saída** | `pesos/maskrcnn_d2_flipswap.pth`, GT 2D do teste (CVAT) e `features_2d_flipswap/<case_id>.npz` |

1. **Treino do detector (Kaggle, GPU).** `train_maskrcnn_dentex.py` treina o Mask R-CNN (32 classes FDI) no DENTEX
   (D1). O `tcc-xray-v2.ipynb` gera os pré-rótulos e retreina com o flip que troca o lado do dente (D2, o usado
   na fusão). No notebook, `DATASET_PANORAMICAS` é o nome do dataset do Kaggle com as panorâmicas do passo 1.
2. **GT 2D do conjunto de teste.** O pacote com as panorâmicas e os pré-rótulos é anotado à mão no CVAT; o export
   (COCO) vai para `cvat_teste_multimodal/export_gt/`.
   ```
   python 02_detector_2d/montar_cvat_teste_multimodal.py --tag v2
   ```
3. **Tokens 2D.** Para cada dente detectado: caixa, máscara, FDI, probabilidades e as features internas do Mask
   R-CNN (`feat_box` 1024, `feat_mask` 256), que viram os tokens da fusão.
   ```
   python 02_detector_2d/extrair_features_2d.py --extra $TCC_DADOS/panoramicas/ds_complementar/processados \
                                                        $TCC_DADOS/panoramicas/ds_ampliacao/processados
   ```

### 3. Malhas

| | |
|---|---|
| **Entrada** | `brutos/<lote>_{upper,lower}/*.vtp` (malha do escaneamento com o FDI de cada face) |
| **Saída** | `fusao_dados/malhas/<case_id>_<arch>.npz` e os manifestos (`manifesto*.csv`, com a divisão) |

```
python 01_dados/preparar_fusao.py --lote tcc            # conjunto multimodal (teste = os casos anotados no CVAT)
python 01_dados/preparar_fusao.py --lote complementar   # 121 casos, só treino
python 01_dados/preparar_fusao.py --lote ampliacao      # ampliação
python 01_dados/redividir_novos.py --aplicar            # ampliação: 75/10/15, semente 42
```
Cada malha é decimada para ~8.000 células, o FDI vira 17 classes (gengiva + 16 dentes) e cada célula ganha a
posição ao longo do arco. Divisão final da arcada inferior: **2.088 treino / 278 val / 418 teste** (o
`rodar_g5.sh` confere e aborta se não bater).

### 4. Segmentação: o experimento final (G5)

| | |
|---|---|
| **Entrada** | `fusao_dados/` e `features_2d_flipswap/` |
| **Saída** | `runs5/` (checkpoints e logs) e `g5_resultados/` (probabilidades, λ e métricas) |
| **Hardware** | GPU (~3,5 dias numa NVIDIA L4); retomável |

```
bash 04_segmentacao_fusao/rodar_g5.sh
```

| Fase | O que faz | Scripts |
|---|---|---|
| A | MeshSegNet **só 3D**, do zero: 3 sementes no treino todo (G5-S) + 4 dobras. Cada arcada recebe um **rascunho 3D** de um modelo que não a viu no treino | `treinar_fusao.py`, `exportar_predicoes.py rascunho` |
| B | Registro panorâmica ↔ malha com o rascunho e montagem do conjunto final | `registro_fusao.py --rotulos`, `montar_limpo.py` |
| C | A partir de cada G5-S, com o mesmo número de passos: **fusão residual com a panorâmica** (G5-FR) × **só 3D continuado** (G5-SC, o controle) | `treinar_fusao.py` |
| D | Avaliação no teste (só aqui) e probabilidades do val/teste | `treinar_fusao.py --avaliar`, `exportar_predicoes.py probs` |
| E | λ do graph-cut escolhido no val e métricas finais | `escolher_lambda.py`, `avaliar_g5.py tudo` |

As receitas de cada treino estão no `rodar_g5.sh` e na docstring do `treinar_fusao.py`.

### 5. Landmarks

| | |
|---|---|
| **Entrada** | `fusao_dados/`, JSON do planejamento (`brutos/<lote>_json/`) e o 3DTeethLand (público) |
| **Saída** | `runs_landmarks/` e a avaliação no teste oficial do 3DTeethLand |

```
python 05_landmarks/tabela_eixos.py                                 # eixo de cada dente no planejamento
python 05_landmarks/recortes_dentes.py                              # recortes por dente (dados próprios)
python 01_dados/teeth3ds.py baixar
python 01_dados/teeth3ds.py compactar
python 01_dados/teeth3ds.py preparar --conjunto landmarks_treino
python 05_landmarks/recortes_3dteethland.py                         # recortes por dente (3DTeethLand)
bash 05_landmarks/rodar_landmarks.sh                                # GPU: 3 sementes + ablações
python 01_dados/teeth3ds.py preparar                                # teste oficial
python 05_landmarks/teste_3dteethland.py --seg-inferior <best.pt>   # mAP/mAR oficiais
```
O modelo recebe 512 pontos de cada dente (com 2 mm de gengiva) no referencial do dente e prevê o eixo longo e os
pontos mesial, distal, facial, interno, externo e as cúspides. É treinado junto com o eixo do planejador e os pontos
do 3DTeethLand.

### 6. Métricas

Rodam no fim do `rodar_g5.sh`, ou separadas:
```
python 06_avaliacao/avaliar_g5.py mcnemar gc    # regra pré-registrada: FR × SC por semente (McNemar), com graph-cut
python 06_avaliacao/avaliar_g5.py por-tipo gc   # por tipo de dente
python 06_avaliacao/avaliar_g5.py miou          # mIoU
python 06_avaliacao/avaliar_g5.py acuracia      # acurácia por célula com graph-cut
```

### 7. Inferência num caso novo

```
python 07_inferencia/rodar.py --inferior arcada_inferior.stl --panoramica panoramica.png --saida resultado/
```
Orienta a malha, segmenta, numera e devolve os landmarks no referencial do arquivo. Detalhes em
[`07_inferencia/MANUAL.md`](07_inferencia/MANUAL.md).

## Reprodutibilidade

O treino em GPU não é determinístico: um novo treino dá números próximos, mas não idênticos, aos da monografia.

## Licença

O código deste repositório é distribuído sob a licença MIT (ver [`LICENSE`](LICENSE)).

## Referências

- MeshSegNet: Lian et al., IEEE TMI 2020 — <https://github.com/Tai-Hsien/MeshSegNet> (MIT, `00_comum/LICENSE_MeshSegNet`).
- Detectron2 (Apache 2.0); os arquivos de configuração em `02_detector_2d/cfg/` vêm dele.
- DENTEX (detector 2D) e Teeth3DS+ / 3DTeethLand (avaliação dos landmarks): dados públicos, com as licenças próprias.
- `05_landmarks/metricas_3dteethland.py`: cópia da métrica oficial do desafio 3DTeethLand (<https://github.com/crns-smartvision/3DTeethLand>, `evaluation/metrics.py` e `evaluation/score.py`).
