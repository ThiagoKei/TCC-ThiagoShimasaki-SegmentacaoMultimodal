"""G4 — monta os dados LIMPOS: uma cópia do fusao_dados em que a predição do MeshSegNet
pré-treinado (PT) não existe mais.

Em cada arcada:
  rascunho_label / rascunho_conf ← rascunho do G4-S (val/teste) ou da dobra G4-S.k que não viu o caso (treino)
  arco                   ← invertido onde o rascunho próprio discorda do PT sobre o lado direito (arco*.csv do PC)
  registro/              ← o registro refeito no PC com o rascunho próprio (registro_fusao.py --rotulos --tag limpo)
O campo `rascunho_label` recebe o rascunho do nosso modelo só 3D.

Uso (G5, como no 04_segmentacao_fusao/rodar_g5.sh):
  python 03_registro/montar_limpo.py --dados <TCC_DADOS>/fusao_dados --rascunhos <TCC_DADOS>/rascunhos_g5 \
      --registro <TCC_DADOS>/fusao_dados/registro_limpo_g5 --saida <TCC_DADOS>/fusao_dados_limpo_g5 --dobras 4 \
      --manifestos manifesto.csv manifesto_complementar.csv manifesto_ampliacao.csv --run-cheio G5-S.s42
Confere: toda arcada da arcada pedida tem rascunho, o tamanho bate com a malha, e nenhum rascunho de treino veio
de um run que viu o caso (o campo `run` do rascunho contém a dobra certa).
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
import caminhos  # noqa: E402
from dados_fusao import dobra_do_caso  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dados", type=Path, default=caminhos.FUSAO_DADOS)
ap.add_argument("--rascunhos", type=Path, required=True)
ap.add_argument("--registro", type=Path, required=True, help="pasta registro_limpo/ vinda do PC (com arco*.csv)")
ap.add_argument("--saida", type=Path, required=True)
ap.add_argument("--arch", default="lower")
ap.add_argument("--dobras", type=int, default=5)
ap.add_argument("--manifestos", nargs="+", default=["manifesto.csv", "manifesto_complementar.csv"],
                help="G5: + manifesto_ampliacao.csv")
ap.add_argument("--run-cheio", default="G4-S", help="run que gera os rascunhos do val/teste (G5: G5-S.s42)")
a = ap.parse_args()

if a.saida.exists():
    raise SystemExit(f"{a.saida} já existe; apague ou escolha outra --saida")
(a.saida / "malhas").mkdir(parents=True)

inv = pd.concat([pd.read_csv(f) for f in sorted(a.registro.glob("arco*.csv"))], ignore_index=True)
inv = {(int(r.case_id), r.arch): bool(r.arco_invertido) for r in inv.itertuples()}

resumo = []
for nome in a.manifestos:
    man = pd.read_csv(a.dados / nome)
    man = man[man.arch == a.arch].reset_index(drop=True)
    for r in man.itertuples():
        chave = f"{int(r.case_id)}_{r.arch}"
        f = a.rascunhos / f"{chave}.npz"
        if not f.exists():
            raise SystemExit(f"sem rascunho: {chave} ({r.split}, {nome})")
        rz = np.load(f)
        run = str(rz["run"])
        if r.split == "treino":
            k = dobra_do_caso(r.case_id, a.dobras)
            assert run.rstrip("/").endswith(f".k{k + 1}"), f"{chave}: rascunho de {run}, devia vir da dobra k{k + 1}"
        else:
            assert Path(run).name == a.run_cheio, f"{chave}: rascunho de {run}, devia vir do {a.run_cheio}"
        z = dict(np.load(a.dados / "malhas" / r.arquivo))
        assert len(rz["label3d"]) == len(z["label"]), chave
        z["rascunho_label"] = rz["label3d"].astype(np.int8)
        z["rascunho_conf"] = rz["conf"].astype(np.float16)
        virou = inv.get((int(r.case_id), r.arch), False)
        if virou:
            z["arco"] = (1.0 - z["arco"]).astype(np.float32)
        np.savez(a.saida / "malhas" / r.arquivo, **z)
        resumo.append(dict(case_id=r.case_id, arch=r.arch, split=r.split, manifesto=nome, run=Path(run).name,
                           arco_invertido=virou, acc_rascunho=float(np.mean(z["rascunho_label"] == z["label"]))))
    man.to_csv(a.saida / nome, index=False)

shutil.copytree(a.registro, a.saida / "registro")
R = pd.DataFrame(resumo)
R.to_csv(a.saida / "montagem_limpo.csv", index=False)
print(f"{len(R)} arcadas ({a.arch}) → {a.saida}")
print(R.groupby("split").agg(arcadas=("case_id", "size"), acc_rascunho=("acc_rascunho", "mean"),
                             arco_invertido=("arco_invertido", "sum")).round(4).to_string())
print(f"registro: {len(list((a.saida / 'registro').glob('*.npz')))} arcadas registradas")
