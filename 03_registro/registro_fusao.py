"""Registro sem rótulo nas arcadas da fusão: o prior f̂(s) da atenção guiada e a fusão tardia.

Casamento do 03_registro/registro.py (ICP 1D: alinhamento que preserva a ordem e pula dos dois lados,
+ β quando rascunho 3D e v2 discordam do número), mas com EXATAMENTE o que a rede recebe:
  3D  o `arco` de cada célula (o s da rede, 0 = lado direito) e o `rascunho_label` cru do .npz do preparar_fusao;
      uma mancha por classe prevista, s da mancha = mediana do arco das suas células
  2D  as detecções do v2 em features_2d_flipswap/<case_id>.npz (as que viram tokens); x = centróide da
      coroa (40% oclusal da máscara), normalizado pelo recorte, como o pos_norm dos tokens
Assim f̂(arco_i) sai na mesma unidade do x do token, e o prior da atenção é −λ(f̂(s_i) − x_j)²/2σ².

Saída por arcada (fusao_dados/registro/<case_id>_<arch>.npz):
  coef      f̂ (grau 3, np.polyval; afim se < 8 pares)
  x_tok     x da coroa de cada token, na MESMA ordem do dados_fusao.tokens_2d (pos_norm x crescente, T_MAX)
  largura   largura mediana do dente 2D (mesma unidade de x)
  pares     (classe do rascunho 3D, índice do token em x_tok) casados
No teste (141 casos), contra o GT, e só aqui:
  par certo     a mancha (classe do planejador na maioria das células) e a detecção (GT do CVAT com IoU >= 0,3)
                são o mesmo dente
  erro_prior    |f̂(s) − x| / largura, com s = mediana do arco das células de cada dente do GT e x = coroa do CVAT
  fusão tardia  as células de cada mancha casada recebem o número do dente 2D; DSC por dente na malha decimada,
                contra o rascunho 3D na mesma malha (sem pós-processamento nos dois)

Uso: python 03_registro/registro_fusao.py [--manifesto <csv>] [--rotulos <pasta dos rascunhos> --tag <nome>]
     (~10 min nas 971 arcadas do conjunto multimodal)
"""
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
import caminhos  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pycocotools import mask as mu  # noqa: E402

import registro as rg  # noqa: E402
from dados_fusao import FEATURES_2D  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
DADOS = caminhos.FUSAO_DADOS
GT = caminhos.CVAT_TESTE / "export_gt" / "gt_cvat_2026-10-27_renum_planejador.zip"   # GT 2D do CVAT (só para medir)
MANIFESTO = caminhos.CVAT_TESTE / "manifest_v2.csv"
SAIDA = DADOS / "registro"
ORDER16 = {"upper": [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28],
           "lower": [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]}


def posicao(f):
    """Decíduo → a posição permanente que ele ocupa (53 → 13), como no preparar_fusao.py."""
    return (f // 10 - 4) * 10 + f % 10 if 51 <= f <= 85 else f


def coco(z):
    """{imagem: [(fdi, RLE)]} do GT do CVAT. As máscaras são decodificadas só na hora (todas juntas dariam ~1 GB)."""
    d = json.loads(zipfile.ZipFile(z).read("annotations/instances_default.json"))
    im = {i["id"]: i for i in d["images"]}
    cat = {c["id"]: int(c["name"].split("-")[1]) for c in d["categories"]}
    r = defaultdict(list)
    for a in d["annotations"]:
        i = im[a["image_id"]]
        r[i["file_name"].split("/")[-1]].append(
            (cat[a["category_id"]], mu.merge(mu.frPyObjects(a["segmentation"], i["height"], i["width"]))))
    return r


def maioria(x):
    v, c = np.unique(x, return_counts=True)
    return int(v[c.argmax()])


def dsc_dentes(pred, gt):
    c = [k for k in np.unique(gt) if k > 0]
    return float(np.mean([2 * np.sum((pred == k) & (gt == k)) / (np.sum(pred == k) + np.sum(gt == k)) for k in c]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="só as N primeiras arcadas de teste (conferência rápida)")
    ap.add_argument("--manifesto", default="manifesto.csv",
                    help="manifesto_complementar.csv registra só os 121 casos do treino do MeshSegNet pré-treinado "
                         "(o resumo vai para registro_resumo_<manifesto>.csv; o registro/ é o mesmo)")
    ap.add_argument("--rotulos", type=Path, default=None,
                    help="pasta com <case_id>_<arch>.npz (label3d) de um modelo 3D NOSSO; substitui o "
                         "rascunho_label do pré-treinado nas manchas E na orientação do arco. Só as arcadas com rótulo.")
    ap.add_argument("--tag", default=None, help="com --rotulos: registro_<tag>/ e registro_resumo_<tag>.csv")
    ap.add_argument("--dados", type=Path, default=None, help="raiz do fusao_dados")
    ap.add_argument("--pasta-2d", type=Path, default=None, help="pasta das features 2D")
    a = ap.parse_args()
    global SAIDA, DADOS, FEATURES_2D
    if a.dados:
        DADOS, SAIDA = a.dados, a.dados / "registro"
    if a.pasta_2d:
        FEATURES_2D = a.pasta_2d
    rot, tag, manif = a.rotulos, a.tag, a.manifesto   # `a` é reaproveitado no laço (pares)
    man = pd.read_csv(DADOS / a.manifesto)
    if rot:
        if not a.tag:
            raise SystemExit("--rotulos exige --tag (para não sobrescrever o registro do pré-treinado)")
        SAIDA = DADOS / f"registro_{a.tag}"
        tem = {p.stem for p in a.rotulos.glob("*.npz")}
        man = man[[f"{c}_{h}" in tem for c, h in zip(man.case_id, man.arch)]]
        print(f"--rotulos {a.rotulos}: {len(man)} arcadas com rótulo próprio → {SAIDA}")
    if a.n:
        man = man[man.split == "teste"].head(a.n)
    SAIDA.mkdir(exist_ok=True)
    n_vira, invertidos = 0, []
    if MANIFESTO.exists() and GT.exists():      # GT do CVAT: só para MEDIR o registro no teste (só no PC)
        imagem = dict(pd.read_csv(MANIFESTO)[["case_id", "imagem_cvat"]].itertuples(index=False))
        gt2d = coco(GT)
    else:
        print("sem o GT do CVAT (VM): registra tudo, mas sem as métricas de acerto do teste")
        imagem, gt2d = {}, {}
    linhas = []
    for n, r in enumerate(man.itertuples(), 1):
        z = np.load(DADOS / "malhas" / r.arquivo)
        arco, zl, label = z["arco"].astype(float), z["rascunho_label"].astype(int), z["label"].astype(int)
        if rot:
            # O arco do .npz foi orientado pelo lado que o PRÉ-TREINADO prevê (preparar_fusao). Refaz a orientação
            # com o rótulo próprio, pela mesma regra: classes 1..8 = lado direito do paciente → arco menor.
            zl = np.load(rot / f"{r.case_id}_{r.arch}.npz")["label3d"].astype(int)
            assert len(zl) == len(label), f"{r.case_id}_{r.arch}: {len(zl)} rótulos × {len(label)} células"
            d_n, e_n = (zl >= 1) & (zl <= 8), zl >= 9
            inv = bool(d_n.any() and e_n.any() and arco[d_n].mean() > arco[e_n].mean())
            if inv:
                arco = 1.0 - arco
                n_vira += 1
            invertidos.append(dict(case_id=r.case_id, arch=r.arch, split=r.split, arco_invertido=inv))
        if not (FEATURES_2D / f"{r.case_id}.npz").exists():   # Ampliação sem panorâmica (treino só 3D)
            linhas.append(dict(case_id=r.case_id, arch=r.arch, split=r.split, motivo="sem panorâmica"))
            continue
        f2d = np.load(FEATURES_2D / f"{r.case_id}.npz")
        cls, s = rg.manchas(arco, zl)
        fdi_tok, masc_tok, x_tok, larg_tok = rg.deteccoes(f2d, r.arch, com_mascaras=True)
        v = np.flatnonzero(~np.isnan(x_tok))                  # tokens com máscara (os que entram no registro)
        fdi2, masc, x, larg = fdi_tok[v], [masc_tok[k] for k in v], x_tok[v], larg_tok[v]
        linha = dict(case_id=r.case_id, arch=r.arch, split=r.split, n_3d=len(cls), n_2d=len(fdi2))
        if len(cls) < 4 or len(fdi2) < 4:
            linhas.append(dict(linha, motivo="menos de 4 manchas ou detecções"))
            continue
        w = float(np.median(larg))
        o3, o2 = np.argsort(s), np.argsort(x)
        fdi3 = np.array([ORDER16[r.arch][c - 1] for c in cls])
        discorda = (fdi3[o3][:, None] != fdi2[o2][None, :]).astype(float)
        custo, coef, pares = rg.registrar(s[o3], x[o2], w, discorda)
        if coef is None:
            linhas.append(dict(linha, motivo="registro sem pares"))
            continue
        coef = np.r_[np.zeros(4 - len(coef)), coef]
        pares = np.array([(cls[o3[i]], o2[j]) for i, j in pares], int)    # j: índice entre os válidos
        np.savez(SAIDA / f"{r.case_id}_{r.arch}.npz", coef=coef, x_tok=x_tok, largura=w,
                 pares=np.c_[pares[:, 0], v[pares[:, 1]]], s_mancha=s, classe_mancha=cls)
        linha.update(n_pares=len(pares), custo=custo, discordam=int(sum(fdi3[list(cls).index(a)] != fdi2[j]
                                                                            for a, j in pares)))
        if r.split == "teste" and r.case_id in imagem and imagem[r.case_id] in gt2d:
            gts = [(posicao(f), mu.decode(rle).astype(bool)) for f, rle in gt2d[imagem[r.case_id]]
                   if posicao(f) // 10 in rg.QUAD[r.arch]]
            # dente físico de cada mancha (planejador) e de cada detecção (CVAT, IoU >= 0,3)
            fis3 = {a: ORDER16[r.arch][maioria(label[zl == a]) - 1] if maioria(label[zl == a]) > 0 else 0
                    for a in cls}
            fis2 = []
            for m in masc:
                iou = [(m & g).sum() / max((m | g).sum(), 1) for _, g in gts]
                fis2.append(gts[int(np.argmax(iou))][0] if iou and max(iou) >= 0.3 else 0)
            certos = sum(fis3[a] > 0 and fis3[a] == fis2[j] for a, j in pares)
            # erro do prior nos dentes do GT presentes nos dois exames
            x_gt = {f: (rg.coroa_2d(f, m)[0] - f2d["recorte"][1]) / f2d["recorte"][3] for f, m in gts}
            erro = [abs(np.polyval(coef, np.median(arco[label == c])) - x_gt[ORDER16[r.arch][c - 1]]) / w
                    for c in np.unique(label[label > 0]) if ORDER16[r.arch][c - 1] in x_gt]
            # fusão tardia: a mancha casada recebe o número do dente 2D
            tardia = zl.copy()
            f2l = {f: i + 1 for i, f in enumerate(ORDER16[r.arch])}
            for a, j in pares:
                if fdi2[j] in f2l:
                    tardia[zl == a] = f2l[fdi2[j]]
            linha.update(pares_certos=int(certos), todos_certos=certos == len(pares),
                         erro_prior_mediano=float(np.median(erro)) if erro else np.nan,
                         erro_prior_p90=float(np.quantile(erro, 0.9)) if erro else np.nan,
                         dsc_rascunho=dsc_dentes(zl, label), dsc_tardia=dsc_dentes(tardia, label),
                         acc_rascunho=float(np.mean(zl == label)), acc_tardia=float(np.mean(tardia == label)))
        linhas.append(linha)
        if n % 50 == 0:
            print(f"[{n}/{len(man)}] {r.case_id} {r.arch}", flush=True)

    D = pd.DataFrame(linhas)
    sufixo = "" if manif == "manifesto.csv" else f"_{Path(manif).stem}"
    if rot:
        print(f"arco reorientado pelo rótulo próprio em {n_vira} arcadas (discordou do pré-treinado)")
        # Para a VM aplicar a MESMA orientação ao montar os dados limpos (montar_limpo.py).
        pd.DataFrame(invertidos).to_csv(SAIDA / f"arco{sufixo}.csv", index=False)
    D.to_csv(DADOS / (f"registro_resumo_{tag}{sufixo}.csv" if rot else f"registro_resumo{sufixo}.csv"), index=False)
    ok = D[D.motivo.isna()] if "motivo" in D else D
    print(f"\n{len(ok)} de {len(D)} arcadas registradas ({len(D) - len(ok)} puladas) → {SAIDA}")
    if "motivo" in D:
        print("puladas por motivo:", D.motivo.value_counts().to_dict())
    if ok.empty or "n_pares" not in ok:
        return
    print(f"pares por arcada: mediana {ok.n_pares.median():.0f}; rascunho 3D e v2 discordam do número em "
          f"{ok.discordam.sum() / ok.n_pares.sum():.1%} dos pares")
    if "pares_certos" not in ok:                          # manifesto sem teste (ex.: manifesto_complementar.csv)
        return
    T = ok[ok.split == "teste"].dropna(subset=["pares_certos"])
    print(f"\nTESTE ({T.case_id.nunique()} casos, {len(T)} arcadas)")
    print(f"  pares certos: {T.pares_certos.sum() / T.n_pares.sum():.1%} | arcadas com todos certos: "
          f"{T.todos_certos.mean():.1%}   (pós-processado, 143 casos: 96,5% / 79,9%)")
    print(f"  erro do prior (larguras de dente): mediana {T.erro_prior_mediano.median():.3f}, p90 das arcadas "
          f"{T.erro_prior_p90.median():.3f}; prior ruim (> 0,5): {(T.erro_prior_mediano > 0.5).sum()} arcadas"
          f"   (0,088 / 0,214)")
    print(f"  fusão tardia × rascunho 3D (malha 8k, sem pós-processamento): DSC dentes {T.dsc_tardia.mean():.4f} × "
          f"{T.dsc_rascunho.mean():.4f}; acurácia por célula {T.acc_tardia.mean():.4f} × {T.acc_rascunho.mean():.4f}")
    d = T.dsc_tardia - T.dsc_rascunho
    print(f"  arcadas em que a tardia melhora / piora / empata: {(d > 1e-4).sum()} / {(d < -1e-4).sum()} / "
          f"{(d.abs() <= 1e-4).sum()}")
    print(T.groupby("arch")[["dsc_rascunho", "dsc_tardia"]].mean().to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
