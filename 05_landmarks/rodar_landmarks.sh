#!/bin/bash
# Treino e avaliação do módulo de pontos de referência (landmarks) por dente.
#   so3d   = treino conjunto (eixo do planejador + pontos do 3DTeethLand), sem panorâmica  ← modelo proposto
#   com2d  = o mesmo + fusão tardia com o token 2D do mesmo FDI e o ângulo da panorâmica  (hipótese refutada)
#   3 sementes de cada; ablação do treino conjunto só na semente 42: so_eixo (sem 3DTeethLand) e so_pontos (sem planejador)
# Antes: tabela_eixos.py → recortes_dentes.py; teeth3ds.py preparar --conjunto landmarks_treino → recortes_3dteethland.py
# Uso:   TCC_DADOS=/caminho/dos/dados bash 05_landmarks/rodar_landmarks.sh
# Retomável: pula o que já tem avaliacao.txt.
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
D=${TCC_DADOS:-$HOME}
export TCC_DADOS=$D
DADOS=${DADOS:-$D/fusao_dados/landmarks}
EP=${EP:-60}
DISP=${DISP:-cuda}
PY=${PY:-$(command -v python || command -v python3)}
cd "$D"
mkdir -p runs_landmarks
run () {
  NOME=$1; shift
  if [ -f runs_landmarks/$NOME/avaliacao.txt ]; then echo "== $NOME já feito"; return; fi
  echo "== $NOME  ($(date -u +%H:%M) UTC)"
  $PY $REPO/05_landmarks/treinar_landmarks.py --nome $NOME --dados $DADOS --dispositivo $DISP --epocas $EP --threads 4 "$@" \
      > runs_landmarks/$NOME.log 2>&1
  $PY $REPO/05_landmarks/treinar_landmarks.py --nome $NOME --dados $DADOS --dispositivo $DISP --avaliar \
      > runs_landmarks/$NOME/avaliacao.txt 2>&1
  tail -n 4 runs_landmarks/$NOME/avaliacao.txt
}
for SEM in 42 43 44; do
  run so3d_s$SEM  --semente $SEM
  run com2d_s$SEM --semente $SEM --com-2d
done
run so_eixo_s42   --semente 42 --modo so_eixo
run so_pontos_s42 --semente 42 --modo so_pontos
echo "fim ($(date -u +%H:%M) UTC)"
