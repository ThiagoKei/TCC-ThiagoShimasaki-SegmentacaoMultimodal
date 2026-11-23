"""Onde ficam os dados, os pesos e os resultados.

Tudo fica sob uma pasta raiz, dada pela variável de ambiente TCC_DADOS (padrão: a pasta pessoal do usuário).
A estrutura esperada dentro dela está descrita em DADOS.md. Cada caminho também pode ser trocado pelos
argumentos de linha de comando dos scripts (--dados, --pasta-2d, --saida, ...).

    export TCC_DADOS=/caminho/para/dados          (Linux/macOS)
    set TCC_DADOS=D:\\dados_tcc                     (Windows)
"""
import os
from pathlib import Path

RAIZ = Path(os.environ.get("TCC_DADOS", Path.home())).expanduser()

# Entradas
BRUTOS = Path(os.environ.get("TCC_BRUTOS", RAIZ / "brutos"))     # <lote>_{upper,lower}/*.vtp e <lote>_json/*.json
PANORAMICAS = RAIZ / "panoramicas"                              # panorâmicas pré-processadas (512×512), por lote
LISTAS = RAIZ / "listas"                                        # listas de casos e status (CSV)
CVAT_TESTE = RAIZ / "cvat_teste_multimodal"                     # conjunto de teste multimodal + GT 2D do CVAT
TEETH3DS = RAIZ / "teeth3ds"                                    # Teeth3DS+ / 3DTeethLand (dados públicos)
PESOS = RAIZ / "pesos"                                          # pesos do detector 2D (Mask R-CNN)

# Intermediários
FUSAO_DADOS = RAIZ / "fusao_dados"                              # malhas decimadas (.npz) + manifestos
FEATURES_2D = RAIZ / "features_2d_flipswap"                     # tokens 2D por caso (.npz)
FEATURES_2D_ESPELHO = RAIZ / "features_2d_flipswap_espelho"

# Saídas
RUNS = RAIZ / "runs5"                                           # checkpoints e logs dos treinos da G5
RESULTADOS = RAIZ / "g5_resultados"                             # probabilidades, λ do graph-cut e métricas
