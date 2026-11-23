"""G5: redistribui a ampliação em treino / val / teste (75/10/15).

Antes, toda a ampliação (manifesto_ampliacao.csv) ia para o treino, e o val (52) e o teste (139) eram só do conjunto
original. Agora, na arcada INFERIOR (a da G5), sobre o total de 2.784 arcadas:
  teste ≈ 15 % = os 139 de sempre + casos novos;  val ≈ 10 % = os 52 de sempre + casos novos;  o resto no treino.
Regras: sorteio por CASO (semente 42), só casos com panorâmica (features_2d_flipswap/<case_id>.npz) e com arcada
inferior; a arcada superior do mesmo caso vai junto para o mesmo split (para a superior depois). Os splits do
manifesto.csv e do manifesto_complementar.csv não mudam. Grava backup do manifesto antigo.
Uso: python 01_dados/redividir_novos.py [--dados <TCC_DADOS>/fusao_dados] [--aplicar]   (sem --aplicar só mostra)
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
import caminhos  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", type=Path, default=caminhos.FUSAO_DADOS)
    ap.add_argument("--pasta-2d", type=Path, default=caminhos.FEATURES_2D)
    ap.add_argument("--frac-val", type=float, default=0.10)
    ap.add_argument("--frac-teste", type=float, default=0.15)
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--aplicar", action="store_true")
    a = ap.parse_args()
    base = pd.concat([pd.read_csv(a.dados / f) for f in ("manifesto.csv", "manifesto_complementar.csv")])
    novos = pd.read_csv(a.dados / "manifesto_ampliacao.csv")
    assert (novos.split == "treino").all(), "a ampliação já foi redividida (rode a partir do backup)"
    assert not set(novos.case_id) & set(base.case_id[base.split != "treino"]), "caso da ampliação no val/teste antigo"
    low_b, low_n = base[base.arch == "lower"], novos[novos.arch == "lower"]
    total = len(low_b) + len(low_n)
    alvo_val = round(a.frac_val * total) - int((low_b.split == "val").sum())
    alvo_te = round(a.frac_teste * total) - int((low_b.split == "teste").sum())
    elegiveis = sorted(c for c in low_n.case_id.unique() if (a.pasta_2d / f"{c}.npz").exists())
    rng = np.random.default_rng(a.semente)
    ordem = list(rng.permutation(elegiveis))
    teste, val = set(ordem[:alvo_te]), set(ordem[alvo_te:alvo_te + alvo_val])
    novos["split"] = np.where(novos.case_id.isin(teste), "teste", np.where(novos.case_id.isin(val), "val", "treino"))
    tudo = pd.concat([base, novos])
    print(f"inferior: {total} arcadas; elegíveis na ampliação (com panorâmica): {len(elegiveis)}; "
          f"novos → teste {alvo_te}, val {alvo_val}")
    for arch in ("lower", "upper"):
        c = tudo[tudo.arch == arch].split.value_counts()
        n = c.sum()
        print(f"{arch}: " + " | ".join(f"{s} {c.get(s, 0)} ({100 * c.get(s, 0) / n:.1f} %)" for s in ("treino", "val", "teste")))
    print("ampliação por split (inferior):", novos[novos.arch == "lower"].split.value_counts().to_dict())
    if a.aplicar:
        bkp = a.dados / "manifesto_ampliacao.antes_redivisao.csv"
        if not bkp.exists():
            shutil.copy(a.dados / "manifesto_ampliacao.csv", bkp)
        novos.to_csv(a.dados / "manifesto_ampliacao.csv", index=False)
        print(f"gravado (backup em {bkp.name})")


if __name__ == "__main__":
    main()
