"""Inclinação mesiodistal por dente: planejador × coroa 3D × panorâmica (landmarks, passo 1, sem treino).

Pergunta: o ângulo do dente na panorâmica (que mostra a RAIZ) traz informação sobre o eixo longo que o planejador
definiu, além do que a coroa 3D já dá? E o ajuste do planejador (GT − coroa) acompanha a raiz?

Plano de comparação = o da panorâmica: o arco "desenrolado" (tangente t ao arco, no plano horizontal) × a vertical.
Sinal comum: positivo = o ápice vai para o lado ESQUERDO do paciente (direita da imagem: o lado direito do paciente
fica à esquerda da panorâmica, conferido pelos quadrantes dos tokens).
  tip_gt     eixo IE do planejador (coluna 2 da rotação do objeto do eixo do dente = initial_rotation do JSON)
  tip_coroa  linha de base 3D sem 2D: do topo da coroa (20% das células mais oclusais do dente) ao centro do colo
             (células do dente vizinhas da gengiva), como uma sugestão automática de eixos
  tip_2d     eixo principal (PCA) da máscara do dente na panorâmica (mask_rle do v2, que inclui a raiz)
Convenção do modelo: Y vertical (coroa −Y no superior, +Y no inferior), lado direito do paciente em −X, +Z anterior.
Tangente t: pela posição da bola verde (initial_location) dos vizinhos na ordem do arco (direita → esquerda).

Saída: <TCC_DADOS>/fusao_dados/eixos_por_dente.csv e um resumo (correlações e regressão com validação cruzada
por caso: tip_gt ~ tip_coroa  ×  tip_gt ~ tip_coroa + tip_2d), por tipo de dente.
Uso: python 05_landmarks/tabela_eixos.py [--n N]
"""
import argparse
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
import caminhos  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pycocotools import mask as mu  # noqa: E402

from dados_fusao import ORDER16  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
DADOS = caminhos.FUSAO_DADOS
F2D = caminhos.FEATURES_2D
IMPL = caminhos.BRUTOS
JSONS = [IMPL / "tcc_json", IMPL / "complementar_json"]
APICAL = {"upper": np.array([0.0, 1.0, 0.0]), "lower": np.array([0.0, -1.0, 0.0])}


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tip(eixo, t, apv):
    """Ângulo (graus) do eixo apical no plano (t, vertical); positivo = ápice para o lado de t."""
    e = eixo * np.sign(eixo @ apv or 1.0)
    return float(np.degrees(np.arctan2(e @ t, e @ apv)))


def eixo_coroa(z, classe, arch):
    lab, faces, bary = z["label"], z["faces"], z["bary"].astype(float)
    dente = np.flatnonzero(lab == classe)
    if len(dente) < 30:
        return None
    gv = np.zeros(len(z["verts"]), bool)
    gv[faces[lab == 0].ravel()] = True
    borda = dente[gv[faces[dente]].any(1)]
    if len(borda) < 5:
        return None
    h = bary[dente] @ (-APICAL[arch])                       # cresce no sentido da coroa
    topo = bary[dente[h >= np.quantile(h, 0.8)]].mean(0)
    v = bary[borda].mean(0) - topo
    return v / (np.linalg.norm(v) + 1e-9)


def eixo_2d(f2d, fdi, arch):
    sel = [k for k in range(len(f2d["fdi"])) if int(f2d["fdi"][k]) == fdi]
    if not sel:
        return None, 0
    k = max(sel, key=lambda j: float(f2d["score"][j]))
    H, W = (int(v) for v in f2d["tamanho"])
    m = mu.decode({"size": [H, W], "counts": str(f2d["mask_rle"][k])}).astype(bool)
    ys, xs = np.nonzero(m)
    if len(xs) < 30:
        return None, len(xs)
    c = np.c_[xs - xs.mean(), ys - ys.mean()]
    w, U = np.linalg.eigh(c.T @ c)
    d = U[:, 1]                                              # eixo maior (x para a direita, y para baixo)
    apical_img = -d[1] if arch == "upper" else d[1]          # raiz para cima no superior, para baixo no inferior
    if apical_img < 0:
        d, apical_img = -d, -apical_img
    return float(np.degrees(np.arctan2(d[0], apical_img))), len(xs)


def achar_json(fonte):
    rot = fonte.rsplit("_", 1)[0]
    for d in JSONS:
        if (d / f"{rot}.json").exists():
            return d / f"{rot}.json", rot
    return None, rot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int)
    a = ap.parse_args()
    man = pd.concat([pd.read_csv(DADOS / "manifesto.csv"),
                     pd.read_csv(DADOS / "manifesto_complementar.csv").assign(split="treino_complementar")], ignore_index=True)
    if a.n:
        man = man.head(a.n)
    linhas = []
    for n, r in enumerate(man.itertuples(), 1):
        jp, rot = achar_json(r.fonte)
        f2 = F2D / f"{r.case_id}.npz"
        if jp is None or not f2.exists():
            continue
        J = json.loads(jp.read_text(encoding="utf-8"))["dentes"]
        z, f2d = np.load(DADOS / "malhas" / r.arquivo), np.load(f2)
        ordem = ORDER16[r.arch]
        dentes = {}
        for t in J.values():
            zz = t["planejamento"]
            if zz["arch"].lower() != r.arch or zz.get("deciduous_fdi", 0) or zz["fdi"] not in ordem:
                continue
            R = quat_mat(zz["initial_rotation"])
            dentes[zz["fdi"]] = (np.array(zz["initial_location"], float), R)
        fdis = [f for f in ordem if f in dentes]
        for i, f in enumerate(fdis):
            loc, R = dentes[f]
            ant = dentes[fdis[max(i - 1, 0)]][0]
            pos = dentes[fdis[min(i + 1, len(fdis) - 1)]][0]
            t = (pos - ant) * np.array([1.0, 0.0, 1.0])          # horizontal, da direita para a esquerda do paciente
            if np.linalg.norm(t) < 1e-6:
                continue
            t /= np.linalg.norm(t)
            apv = APICAL[r.arch]
            classe = ordem.index(f) + 1
            ec = eixo_coroa(z, classe, r.arch)
            t2, npx = eixo_2d(f2d, f, r.arch)
            linhas.append(dict(case_id=r.case_id, arch=r.arch, split=r.split, rotulo=rot, fdi=f,
                               tipo=8 - min(classe - 1, 16 - classe),
                               tip_gt=tip(R[:, 2], t, apv),
                               tip_coroa=tip(ec, t, apv) if ec is not None else np.nan,
                               tip_2d=t2 if t2 is not None else np.nan, px_2d=npx,
                               ie_vertical=float(abs(R[:, 2] @ apv))))
        if n % 100 == 0:
            print(f"[{n}/{len(man)}]", flush=True)
    D = pd.DataFrame(linhas)
    D.to_csv(DADOS / "eixos_por_dente.csv", index=False)
    print(f"{len(D)} dentes de {D.case_id.nunique()} casos | com 2D {D.tip_2d.notna().sum()} | "
          f"|cos(IE, vertical)| mediano {D.ie_vertical.median():.3f} (≈1: a coluna 2 é mesmo o eixo longo)")
    resumo(D)


def resumo(D):
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold
    D = D.dropna(subset=["tip_gt", "tip_coroa", "tip_2d"])
    print("\npor tipo (1 = incisivo central … 8 = 3º molar); ρ = Spearman com o GT; RMSE (graus) por validação "
          "cruzada (5 grupos por caso) de tip_gt ~ coroa  ×  ~ coroa + 2D; ρ_aj = Spearman de (GT − coroa) com o 2D")
    for (arch, tipo), g in D.groupby(["arch", "tipo"]):
        if len(g) < 40:
            continue
        rc, r2 = spearmanr(g.tip_coroa, g.tip_gt)[0], spearmanr(g.tip_2d, g.tip_gt)[0]
        raj = spearmanr(g.tip_gt - g.tip_coroa, g.tip_2d)[0]
        err = {}
        for nome, cols in (("coroa", ["tip_coroa"]), ("coroa+2d", ["tip_coroa", "tip_2d"])):
            e = []
            for tr, te in GroupKFold(5).split(g, groups=g.case_id):
                m = LinearRegression().fit(g.iloc[tr][cols], g.iloc[tr].tip_gt)
                e.append((m.predict(g.iloc[te][cols]) - g.iloc[te].tip_gt.values) ** 2)
            err[nome] = float(np.sqrt(np.concatenate(e).mean()))
        print(f"  {arch:5s} tipo {tipo}: n {len(g):4d} | ρ coroa {rc:+.2f} | ρ 2D {r2:+.2f} | ρ_aj {raj:+.2f} | "
              f"RMSE coroa {err['coroa']:.1f}° → coroa+2D {err['coroa+2d']:.1f}° | dp GT {g.tip_gt.std():.1f}°")


if __name__ == "__main__":
    main()
