# Estrutura dos dados

Os dados dos pacientes não são públicos. Esta é a estrutura que os scripts esperam dentro de `$TCC_DADOS`
(`00_comum/caminhos.py`). As pastas marcadas com *(gerada)* são criadas pelos próprios scripts.

```
$TCC_DADOS/
├── brutos/                          (ou $TCC_BRUTOS)
│   ├── <lote>_lower/  <lote>_upper/     {ciclo}_{case_id}_v{versão}_{arch}.vtp — malha do escaneamento com o
│   │                                     array de célula "FDI" (0 = gengiva) vindo do planejamento
│   └── <lote>_json/                     {ciclo}_{case_id}_v{versão}.json — planejamento (eixo e posição de cada dente)
│        lotes: tcc (conjunto multimodal), complementar (121 casos usados no treino do MeshSegNet
│        pré-treinado; só entram no treino) e ampliacao (ampliação, redividida 75/10/15)
│        JSON: {"dentes": {nome: {"planejamento": {fdi, arch, initial_location, initial_rotation,
│        pivot_crown_local, deciduous_fdi, ...}}}}
├── panoramicas/                     panorâmicas pré-processadas (512×512, 01_dados/preprocessar_panoramica.py)
│   ├── ds_limpo/<lote>_limpo/processados/<case_id>.png
│   ├── ds_complementar/processados/<case_id>.png
│   └── ds_ampliacao/processados/<case_id>.png
├── listas/
│   ├── teste_multimodal_ordem.csv       ordem, pasta, case_id, panoramica, primeira_publicacao, ..., status, status_3d
│   │                                     (ordem ≤ 150 = conjunto de teste; status "pulado_*" = caso excluído)
│   ├── casos_planejamento.csv                   case_id, id_ciclo, versao, status, data_evento, primeira_publicacao
│   └── complementar_status.csv  ampliacao_status.csv     ordem, case_id, status, status_3d
├── cvat_teste_multimodal/           conjunto de teste multimodal
│   ├── export_gt/gt_cvat_2026-10-27_renum_planejador.zip   GT 2D (COCO) anotado no CVAT
│   ├── manifest_v2.csv
│   └── 3d/  cvat_v2/images/         (gerada: 02_detector_2d/montar_cvat_teste_multimodal.py)
├── pesos/
│   ├── maskrcnn_d1.pth              detector 2D v1 (train_maskrcnn_dentex.py)
│   ├── maskrcnn_d2_flipswap.pth     detector 2D v2 (tcc-xray-v2.ipynb), o usado na fusão
│   ├── pseudo_labels_novo_v2.zip    pré-rótulos v2 (para conferência)
│   └── g4s.pt  g4fr.pt  landmarks_s4{2,3,4}.pt   (só para 07_inferencia)
├── teeth3ds/                        dados públicos (01_dados/teeth3ds.py)
│   ├── splits/   scans/   prep/ (gerada)
│   └── 3DTeethLand_landmarks_{train,test}.zip
│
├── features_2d_flipswap/  (gerada)  <case_id>.npz (tokens 2D por dente) + indice.csv
├── fusao_dados/           (gerada)  malhas/<case_id>_<arch>.npz + manifesto.csv, manifesto_complementar.csv,
│                                    manifesto_ampliacao.csv (case_id, arch, split, arquivo, ...) + landmarks/
├── rascunhos_g5/  fusao_dados_limpo_g5/  runs5/  g5_resultados/  runs_landmarks/   (gerada)
```

## Formato das malhas preparadas (`fusao_dados/malhas/*.npz`)

| Campo | Forma | Conteúdo |
|---|---|---|
| `verts`, `faces` | (V, 3), (N, 3) | malha decimada (~8.000 células) |
| `bary`, `normals` | (N, 3) | baricentro e normal de cada célula |
| `label` | (N,) | classe 0..16 (0 = gengiva; 1..16 = do posterior direito ao posterior esquerdo) |
| `arco` | (N,) | coordenada ao longo do arco, 0..1 (0 = lado direito do paciente) |
| `rascunho_label`, `rascunho_conf` | (N,) | rascunho 3D: zerado aqui; preenchido pelo `03_registro/montar_limpo.py` |
| `deciduo`, `orient_gt_ok` | | dente decíduo por célula; conferência da orientação do arco com o GT |

## Divisão

- **Teste:** 418 arcadas inferiores (139 do conjunto multimodal anotado + 279 da ampliação); **val:** 278;
  **treino:** 2.088. A divisão é por caso (a superior acompanha a inferior), com semente 42.
- Nenhum caso do val/teste entra em qualquer treino (o `preparar_fusao.py` aborta se houver vazamento).
