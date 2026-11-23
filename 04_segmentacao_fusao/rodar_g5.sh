#!/bin/bash
# Experimento final (G5) da segmentação, arcada INFERIOR, sem intervenção. Pré-registro: monografia, cap. 4.
#   A  G5-S.s42/.s43/.s44 (só 3D, do zero, treino todo) + G5-S.k1..k4 (4 dobras do treino, semente 42), mesmo
#      orçamento em passos → rascunhos 3D (treino: a dobra que não viu o caso; val/teste: o G5-S.s42)
#   B  registro panorâmica ↔ malha com o rascunho próprio (registro_fusao --rotulos) + montar_limpo
#   C  cascata, por semente s: G5-FR.s (fusão residual com a panorâmica) e G5-SC.s (só 3D continuado = controle de
#      mesmo orçamento), os dois partindo do G5-S.s, com o mesmo número de passos
#   D  avaliação no teste (só aqui, no fim) + probabilidades do val/teste
#   E  λ do graph-cut escolhido no val (escolher_lambda) e métricas finais (06_avaliacao)
# Uso:   TCC_DADOS=/caminho/dos/dados bash 04_segmentacao_fusao/rodar_g5.sh
# Retomável: rodar de novo pula o que já terminou (arquivo FEITO em cada run).
# Variáveis: SEMENTES ("42 43 44"), PASSOS_S (13000), PASSOS_C (6500), LR_C (1e-3), MAXJ (runs simultâneos na GPU, 3),
#            PRECISAO (fp32; a G5 usou fp32), EXE (python).
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
SEG=$REPO/04_segmentacao_fusao; REG=$REPO/03_registro; AVAL=$REPO/06_avaliacao
D=${TCC_DADOS:-$HOME}
export TCC_DADOS=$D
mkdir -p "$D/runs5"
cd "$D"
exec > >(tee -a runs5/G5.log) 2>&1
EXE=${EXE:-$(command -v python || command -v python3)}
SEMENTES=${SEMENTES:-"42 43 44"}
S0=$(echo $SEMENTES | cut -d' ' -f1)          # a semente que gera os rascunhos do val/teste e roda as dobras
PASSOS_S=${PASSOS_S:-13000}
PASSOS_C=${PASSOS_C:-6500}
LR_C=${LR_C:-1e-3}                # lr do tronco na fase C (FR e SC iguais)
MAXJ=${MAXJ:-3}
PAUSA=${PAUSA:-180}               # s entre lançar um run e medir a memória livre da GPU de novo
PRECISAO=${PRECISAO:-fp32}
DADOS=$D/fusao_dados; P2D=$D/features_2d_flipswap
TAG=limpo_g5; LIMPO=$D/fusao_dados_limpo_g5; OUT=$D/rascunhos_g5; RES=$D/g5_resultados
echo "== $(date '+%d/%m %H:%M') G5 | dados $D | sementes $SEMENTES | passos S $PASSOS_S, C $PASSOS_C (lr C $LR_C) | $PRECISAO"

BASE="--arch lower --lr 1e-2 --bn-livre --batch 8 --rot-max 10 --com-complementar --com-ampliacao --paciencia 100000 --pasta-2d $P2D --precisao $PRECISAO"
FA="--fusao atencao --prior --portao-celula --peso-atencao 0.1 --congelar 0"
FR="$FA --residual --peso-erro-3d 5 --inicio-fusao 0"

# ---- conferência da divisão (treino 2.088 + val 278 + teste 418 arcadas inferiores) ----
for m in manifesto.csv manifesto_complementar.csv manifesto_ampliacao.csv; do
  [ -f $DADOS/$m ] || { echo "ABORTA: falta $DADOS/$m (ver DADOS.md)"; exit 1; }
done
N_TR=$(cd $DADOS && $EXE -c "
import pandas as pd
m=pd.concat([pd.read_csv(f) for f in ('manifesto.csv','manifesto_complementar.csv','manifesto_ampliacao.csv')])
m=m[m.arch=='lower']; print(len(m[m.split=='treino']), len(m[m.split!='treino']))")
echo "arcadas inferiores: treino + val/teste = $N_TR (esperado 2088 696)"
ESPERADO=$(echo $N_TR | awk '{print $1+$2}')
[ "${ESPERADO:-0}" -gt 0 ] || { echo "ABORTA: não consegui contar as arcadas dos manifestos"; exit 1; }
[ "$N_TR" = "2088 696" ] || [ -n "${SEM_CHECAR_SPLIT:-}" ] || { echo "ABORTA: divisão inesperada ($N_TR); SEM_CHECAR_SPLIT=1 ignora"; exit 1; }

espera_gpu() {   # só começa um run novo com > 7 GB livres na GPU
  command -v nvidia-smi >/dev/null || return
  while true; do
    livre=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
    [ "$livre" -gt 7000 ] && return
    sleep 60
  done
}
treina() {   # treina NOME DADOS args...   (o teste NÃO roda aqui: só na fase D)
  nome=$1; dados=$2; shift 2
  if [ -f runs5/$nome/FEITO ]; then echo "$nome já feito"; return; fi
  espera_gpu
  echo "== $(date '+%H:%M') início $nome ($*)"
  $EXE $SEG/treinar_fusao.py $BASE --dados $dados --saida runs5/$nome "$@" --resume > runs5/$nome.log 2>&1
  rc=$?
  [ $rc -eq 0 ] && touch runs5/$nome/FEITO
  echo "== $(date '+%H:%M') fim $nome (código $rc) | melhor: $(grep ' \*$' runs5/$nome.log | tail -n 1)"
}
# Fila de até MAXJ runs (nunca `wait` sem argumentos: esperaria também o `tee` do log).
J=()
fila() {
  while true; do
    vivos=(); for p in "${J[@]}"; do kill -0 $p 2>/dev/null && vivos+=($p); done; J=("${vivos[@]}")
    [ ${#J[@]} -lt $MAXJ ] && break
    wait -n "${J[@]}"
  done
  treina "$@" & J+=($!)
  [ -f runs5/$1/FEITO ] || sleep $PAUSA
}
espera_fila() { for p in "${J[@]}"; do wait $p; done; J=(); }
exige() { for n in "$@"; do [ -f runs5/$n/FEITO ] || { echo "ABORTA: $n não terminou (ver runs5/$n.log); rode de novo para retomar"; exit 1; }; done; }

# ================= A =================
echo "== $(date '+%H:%M') FASE A"
S3D="--fusao nenhuma --pesos-iniciais nenhum"
fila G5-S.s$S0 $DADOS $S3D --semente $S0 --passos $PASSOS_S
for k in 1 2 3 4; do fila G5-S.k$k $DADOS $S3D --semente $S0 --sem-dobra $((k-1))/4 --passos $PASSOS_S; done
for s in $SEMENTES; do [ $s = $S0 ] || fila G5-S.s$s $DADOS $S3D --semente $s --passos $PASSOS_S; done
espera_fila
exige G5-S.k1 G5-S.k2 G5-S.k3 G5-S.k4 $(for s in $SEMENTES; do echo G5-S.s$s; done)
for n in G5-S.k1 G5-S.k2 G5-S.k3 G5-S.k4 $(for s in $SEMENTES; do echo G5-S.s$s; done); do   # trava: do zero
  $EXE -c "import torch,sys; c=torch.load('runs5/$n/best.pt',map_location='cpu',weights_only=False); \
sys.exit(0 if c.get('pesos_iniciais')=='nenhum' else 1)" || { echo "ABORTA: $n não partiu do zero"; exit 1; }
done
n=$(ls $OUT 2>/dev/null | wc -l)
if [ "$n" -ne "$ESPERADO" ]; then
  mkdir -p $OUT
  for k in 1 2 3 4; do
    $EXE $SEG/exportar_predicoes.py rascunho runs5/G5-S.k$k --so-dobra $((k-1))/4 --com-ampliacao --com-probs --saida $OUT --dados $DADOS --pasta-2d $P2D
  done
  $EXE $SEG/exportar_predicoes.py rascunho runs5/G5-S.s$S0 --com-ampliacao --com-probs --saida $OUT --dados $DADOS --pasta-2d $P2D
  n=$(ls $OUT | wc -l)
fi
echo "rascunhos: $n (esperado $ESPERADO)"
[ "$n" -eq "$ESPERADO" ] || { echo "ABORTA: número de rascunhos errado"; exit 1; }

# ================= B =================
echo "== $(date '+%H:%M') FASE B"
MANIF="manifesto.csv manifesto_complementar.csv manifesto_ampliacao.csv"
if [ ! -f $LIMPO/montagem_limpo.csv ]; then
  rm -rf $LIMPO $DADOS/registro_$TAG
  for m in $MANIF; do
    $EXE $REG/registro_fusao.py --rotulos $OUT --tag $TAG --dados $DADOS --pasta-2d $P2D --manifesto $m || { echo "ABORTA: registro ($m)"; exit 1; }
  done
  $EXE $REG/montar_limpo.py --dados $DADOS --rascunhos $OUT --registro $DADOS/registro_$TAG --saida $LIMPO --dobras 4 \
    --manifestos $MANIF --run-cheio G5-S.s$S0 || { echo "ABORTA: montagem"; rm -rf $LIMPO; exit 1; }
fi

# ================= C =================
echo "== $(date '+%H:%M') FASE C"
for s in $SEMENTES; do
  P0=$D/runs5/G5-S.s$s/best.pt
  [ -f $P0 ] || { echo "ABORTA: falta $P0"; exit 1; }
  fila G5-FR.s$s $LIMPO $FR --semente $s --pesos-iniciais $P0 --passos $PASSOS_C --lr $LR_C
  fila G5-SC.s$s $LIMPO --fusao nenhuma --semente $s --pesos-iniciais $P0 --passos $PASSOS_C --lr $LR_C
done
espera_fila
exige $(for s in $SEMENTES; do echo G5-FR.s$s G5-SC.s$s; done)

# ================= D =================
echo "== $(date '+%H:%M') FASE D (teste: só agora)"
mkdir -p $RES/probs
for s in $SEMENTES; do
  for r in G5-S.s$s:$DADOS G5-SC.s$s:$LIMPO G5-FR.s$s:$LIMPO; do
    n=${r%%:*}; d=${r##*:}
    [ -f runs5/$n.teste.txt ] || $EXE $SEG/treinar_fusao.py --arch lower --saida runs5/$n --avaliar --com-ampliacao --dados $d --pasta-2d $P2D > runs5/$n.teste.txt 2>&1
    echo "teste $n: $(grep '^teste' runs5/$n.teste.txt | tr '\n' ' ')"
    [ -d $RES/probs/$n ] || $EXE $SEG/exportar_predicoes.py probs runs5/$n --com-ampliacao --dados $d --pasta-2d $P2D --saida $RES/probs/$n || echo "FALHOU probs: $n"
  done
done
for k in 1 2 3 4; do   # λ de cada dobra: só o val (o teste não entra)
  [ -d $RES/probs/G5-S.k$k ] || $EXE $SEG/exportar_predicoes.py probs runs5/G5-S.k$k --com-ampliacao --dados $DADOS --pasta-2d $P2D --splits val \
    --saida $RES/probs/G5-S.k$k || echo "FALHOU probs: G5-S.k$k"
done

# ================= E =================
echo "== $(date '+%H:%M') FASE E (graph-cut e métricas)"
$EXE $SEG/escolher_lambda.py --probs $RES/probs --saida $RES/lambda --retomar
$EXE $AVAL/avaliar_g5.py tudo
echo "== $(date '+%d/%m %H:%M') G5 terminou → $RES"
