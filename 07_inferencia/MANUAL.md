# Inferência: segmentação, numeração FDI e landmarks da arcada inferior

**Material de pesquisa (TCC), não é para uso clínico.** Os pesos não estão no repositório: foram treinados com
dados de pacientes e só podem ser obtidos com o autor.

## O que faz

Recebe o **escaneamento 3D da arcada inferior** e a **radiografia panorâmica** do mesmo paciente e devolve:

- cada dente segmentado na malha, com o número FDI (31–38, 41–48);
- os landmarks de cada dente: o eixo longo (IE), os pontos mesial, distal, facial, interno e externo, e as cúspides.

Tudo sai no **referencial do arquivo de entrada**, nas mesmas unidades.

## Como rodar

Python 3.10 ou mais novo, torch ≥ 2.0 e o Detectron2 instalado.
```
pip install -r requirements.txt          # na raiz do repositório
python 07_inferencia/rodar.py --inferior inferior.stl --panoramica panoramica.png --saida resultado/ [--pesos <pasta>] [--dispositivo cpu]
```
A pasta de pesos (padrão `<TCC_DADOS>/pesos`) deve ter `g4s.pt`, `g4fr.pt`, `landmarks_s42.pt`, `landmarks_s43.pt`,
`landmarks_s44.pt` e `maskrcnn_d2_flipswap.pth`.

## Requisitos da entrada

- **3D:** só a arcada **inferior**, em STL, PLY, OBJ ou VTP, em **milímetros**. Use o escaneamento intraoral cru, com
  a gengiva e aberto na base, como o escâner exporta. Uma arcada mede cerca de 60–90 mm. Se o arquivo parecer estar
  em metros ou não for uma arcada, o programa avisa.
- **Panorâmica:** PNG ou JPG, a radiografia panorâmica completa do **mesmo paciente**, na orientação usual (o lado
  direito do paciente à esquerda da imagem). As barras pretas nas bordas são removidas automaticamente.

## As etapas (o que o `rodar.py` faz)

1. **Orientação** (`orientacao.py`). O modelo foi treinado com a arcada "em pé": coroas para +Y, incisivos para +Z.
   A vertical é o eixo de menor variância da malha; o sinal vem da borda aberta do escaneamento, que fica embaixo.
   A frente é a ponta fechada do U. Por cima disso entra uma correção fixa, calibrada em 218 arcadas de treino. Em
   seguida o próprio G4-S confere o sinal vertical: roda com a arcada nas duas posições e fica com a de maior confiança.
2. **Decimação** para 8.000 células (`00_comum/malha.py`). É a resolução que a rede vê.
3. **G4-S** (MeshSegNet só 3D, treinado do zero): faz um primeiro rascunho da segmentação. Ele só serve para orientar
   o arco e para casar a malha com a panorâmica.
4. **Panorâmica** (`02_detector_2d/pipeline_2d.py`): um Mask R-CNN detecta e numera os dentes da radiografia. Cada dente vira um
   "token" com as características da imagem.
5. **Registro panorâmica ↔ malha** (`03_registro/registro.py`): casa a ordem dos dentes do rascunho 3D com a dos dentes da
   panorâmica.
6. **G4-FR** (fusão residual): segmenta e numera a malha usando o 3D **e** os tokens da panorâmica; a panorâmica só
   corrige o que o 3D sozinho erraria.
7. **Graph-cut** (λ = 7, escolhido na validação) + remoção de ilhas: limpa as fronteiras entre os dentes.
8. **Landmarks** (`05_landmarks/landmarks.py`): recorta cada dente com 2 mm de gengiva em volta e passa por uma rede por dente
   (média de 3 modelos). Saem o eixo longo e os pontos.
9. **Volta ao arquivo original:** o rótulo de cada face da malha original sai dos 3 vizinhos mais próximos na malha
   decimada, com suavização da fronteira. Os pontos voltam pela inversa da orientação.

## O que sai (na pasta `--saida`)

| Arquivo | Conteúdo |
|---|---|
| `segmentacao.ply` | A malha **original** colorida por dente: cor por face e por vértice, e a propriedade `fdi` em cada face (0 = gengiva). Abre no MeshLab ou no Blender. |
| `dentes.csv` | Uma linha por dente: FDI, células (8k), faces na malha original, área (mm²), confiança média da rede, se o dente apareceu na panorâmica e se tem landmarks. |
| `landmarks.json` | Por dente: `eixo_IE` (vetor unitário, sentido oclusal → apical), `centro`, `pontos` (Mesial, Distal, FacialPoint, InnerPoint, OuterPoint) e `cuspides`. Tudo no referencial do arquivo de entrada. |
| `landmarks.ply` | Os mesmos pontos, coloridos por tipo; o centro do dente fica em preto. |
| `orientacao.json` | A matriz 4×4 que leva o arquivo ao referencial do modelo, o método usado e a confiança do G4-S em cada sinal. |
| `visualizar.html` | Visualização 3D no navegador: a malha decimada colorida, os pontos e os eixos (precisa de internet para carregar o three.js). |

## Limitações conhecidas

- **Pesos da geração G4.** O pacote de inferência usa o G4-S e o G4-FR; os modelos finais da monografia (G5) são
  treinados por `04_segmentacao_fusao/rodar_g5.sh` e podem ser usados no lugar deles.
- **Só a arcada inferior.**
- **Terceiros molares (38/48):** é a classe mais fraca; eles são perdidos ou inventados com mais frequência.
- **Dente na panorâmica que não está na boca:** um dente incluso que aparece na radiografia mas não está no
  escaneamento pode fazer o modelo "inventar" esse dente e deslocar a numeração dos vizinhos.
- **Orientação automática:** validada nas 139 arcadas de teste, giradas ao acaso: erro mediano 5,7° (p90 12°,
  máximo 35°), nenhuma de ponta-cabeça. O erro se concentra no giro em torno da vertical; giros de 16–35° custaram
  de 0,11 a 0,60 de DSC em 4 arcadas. Confira `orientacao.json` e `visualizar.html`.
- **Escaneamentos de outros escâneres** (base fechada, muita gengiva, modelos de gesso) não foram testados.
- **A panorâmica deve ser do mesmo paciente e da mesma época.** O modelo confia nela para numerar.
