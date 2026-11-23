"""Exporta as predições de um run treinado, célula a célula na malha do fusao_dados (patch=None preserva a ordem).

Dois modos:
  rascunho  a P_3D LIMPA (só 3D, sem a panorâmica) de um modelo G5-S: é o rascunho 3D que orienta o arco e
            define as manchas do registro (03_registro). Saída: label3d (int8, 0..16) e conf (float16) e, com
            --com-probs, as 17 probabilidades (para o graph-cut com o λ* do modelo).
  probs     a saída FINAL do run (com a panorâmica, se for fusão), com as 17 probabilidades: entrada do
            escolher_lambda.py e das métricas. Saída: prob (float16, células × 17), split e época.

Uso (G5, como no 04_segmentacao_fusao/rodar_g5.sh):
  python 04_segmentacao_fusao/exportar_predicoes.py rascunho <TCC_DADOS>/runs5/G5-S.s42 --com-ampliacao --com-probs \
      --saida <TCC_DADOS>/rascunhos_g5                                                (val + teste)
  python 04_segmentacao_fusao/exportar_predicoes.py rascunho <TCC_DADOS>/runs5/G5-S.k1 --so-dobra 0/4 --com-ampliacao \
      --com-probs --saida <TCC_DADOS>/rascunhos_g5                                    (treino, só a dobra de fora)
  python 04_segmentacao_fusao/exportar_predicoes.py probs <TCC_DADOS>/runs5/G5-FR.s42 --com-ampliacao \
      --dados <TCC_DADOS>/fusao_dados_limpo_g5 --saida <TCC_DADOS>/g5_resultados/probs/G5-FR.s42
Saída: <saida>/<case_id>_<arch>.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
import caminhos  # noqa: E402
from dados_fusao import DatasetFusao, juntar  # noqa: E402
from treinar_fusao import novo_modelo, para_device, passo  # noqa: E402


def carregar(run, dev):
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    modelo = novo_modelo(ck)
    modelo.load_state_dict(ck["model_state_dict"])
    return ck, modelo.to(dev).eval()


def rascunho(a, dev):
    ck, modelo = carregar(a.run, dev)
    assert ck.get("assimetrica") or ck.get("residual") or ck["fusao"] == "nenhuma", "precisa de uma P_3D limpa"
    if a.so_dobra:
        a.splits = ["treino"]
        assert ck.get("sem_dobra") == a.so_dobra, f"vazamento: o run treinou sem a dobra {ck.get('sem_dobra')}, não {a.so_dobra}"
    a.saida.mkdir(parents=True, exist_ok=True)
    n = 0
    with torch.no_grad():
        for split in a.splits:
            ds = DatasetFusao(a.dados, split, ck["arch"], patch=None, pasta_2d=a.pasta_2d, com_complementar=True,
                              com_ampliacao=a.com_ampliacao, so_dobra=a.so_dobra if split == "treino" else None)
            for i, b in enumerate(DataLoader(ds, batch_size=1, collate_fn=juntar, num_workers=a.workers)):
                r = ds.linhas.iloc[i]
                assert int(r.case_id) == int(b["case_id"][0])
                prob, _ = passo(modelo, para_device(b, dev), sem_2d=True)
                p = prob[0].cpu()
                extra = {}
                if a.com_probs:   # as 17 probabilidades, para passar o graph-cut (λ* do modelo) nos rascunhos
                    extra["prob"] = p.numpy().astype(np.float16)          # mesma chave do modo probs
                np.savez_compressed(a.saida / f"{int(r.case_id)}_{r.arch}.npz",
                                    label3d=p.argmax(-1).numpy().astype(np.int8),
                                    conf=p.max(-1).values.numpy().astype(np.float16), split=split, run=str(a.run),
                                    epoca=ck["epoca"], **extra)
                n += 1
    print(f"{n} arcadas ({ck['arch']}, splits {a.splits}) → {a.saida}  [run {a.run.name}, ép {ck['epoca']}]")


def probs(a, dev):
    ck, modelo = carregar(a.run, dev)
    a.saida.mkdir(parents=True, exist_ok=True)
    n = 0
    with torch.no_grad():
        for split in a.splits:
            ds = DatasetFusao(a.dados, split, ck["arch"], patch=None, pasta_2d=a.pasta_2d, com_ampliacao=a.com_ampliacao)
            for i, b in enumerate(DataLoader(ds, batch_size=1, collate_fn=juntar, num_workers=a.workers)):
                r = ds.linhas.iloc[i]
                assert int(r.case_id) == int(b["case_id"][0])
                prob, _ = passo(modelo, para_device(b, dev), sem_2d=False)
                np.savez_compressed(a.saida / f"{int(r.case_id)}_{r.arch}.npz",
                                    prob=prob[0].cpu().numpy().astype(np.float16), split=split, epoca=ck["epoca"])
                n += 1
    print(f"{a.run.name}: {n} arcadas ({ck['arch']}, {a.splits}, fusão {ck['fusao']}, ép. {ck['epoca']}) → {a.saida}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("modo", choices=["rascunho", "probs"])
    ap.add_argument("run", type=Path, help="pasta do run (com best.pt)")
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--dados", type=Path, default=caminhos.FUSAO_DADOS)
    ap.add_argument("--pasta-2d", type=Path, default=caminhos.FEATURES_2D)
    ap.add_argument("--splits", nargs="+", default=["val", "teste"])
    ap.add_argument("--com-ampliacao", action="store_true", help="G5: inclui a ampliação (manifesto_ampliacao.csv)")
    ap.add_argument("--so-dobra", default=None,
                    help="rascunho: 'k/K' prevê só a dobra k do TREINO (o run tem de ser o --sem-dobra k/K)")
    ap.add_argument("--com-probs", action="store_true", help="rascunho: grava também prob (N×17, float16)")
    ap.add_argument("--workers", type=int, default=2, help="0 no Windows")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (rascunho if a.modo == "rascunho" else probs)(a, dev)


if __name__ == "__main__":
    main()
