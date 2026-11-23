"""Recortes por dente para o modelo de landmarks (eixo longo IE + bola verde).

Para cada dente do GT (classe c presente no `label` da malha decimada da fusão):
  pontos   as células do dente + as células de gengiva a até MARGEM mm delas, reamostradas para N_PONTOS
  canais   xyz e normal no REFERENCIAL LOCAL + 1 se é célula do dente (0 se gengiva)
Referencial local (só da segmentação, para valer igual no teste com a segmentação prevista):
  origem   centroide das células do dente
  y        apical (a vertical do modelo: +Y no superior, −Y no inferior)
  x        DISTAL: tangente ao arco pelos centroides dos dentes vizinhos (na ordem do arco), no plano horizontal,
           apontando para distal. O espelho direita ↔ esquerda sai daí: o 46 e o 36 ficam no mesmo referencial
  z        vestibular: x × y com o sinal que aponta para fora do arco (do centro da arcada para o dente)
Alvos (GT do planejador, JSON; seção 19.5: só o IE é alvo, o resto é controle/validação): o eixo IE no referencial local (unitário, sentido apical) e os ângulos
  tip = inclinação mesiodistal (positivo = ápice para distal = coroa para mesial) e
  torque = inclinação vestibulolingual (positivo = ápice para lingual = coroa para vestibular);
  bola = initial_location − origem, no referencial local (mm), e md = eixo MD (ponta distal) + rot, só para validação.
2D: o token da panorâmica com o mesmo FDI (feat_box 1024 + feat_mask 256), score e o ângulo da raiz na máscara
  (tip_2d no mesmo sinal anatômico do tip, da tabela_eixos.py); sem token → zeros e tem_2d = False.
Tipo: posição no arco (1 = incisivo central … 8 = 3º molar), grupo (0 incisivo, 1 canino, 2 pré-molar, 3 molar), arcada.

Saída: <TCC_DADOS>/fusao_dados/landmarks/recortes.npz (arrays empilhados) e recortes.csv (uma linha por dente).
Uso: python 05_landmarks/recortes_dentes.py [--n N]
"""
import argparse
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from dados_fusao import ORDER16  # noqa: E402
from tabela_eixos import APICAL, DADOS, F2D, achar_json, eixo_2d, quat_mat  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
SAIDA = DADOS / "landmarks"
N_PONTOS, MARGEM = 512, 2.0
GRUPO = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3}


def referencial(bary, label, classe, arch, centro_arcada):
    """origem, matriz R (linhas = x distal, y apical, z vestibular) ou None."""
    pres = sorted(int(c) for c in np.unique(label) if c > 0)
    cent = {c: bary[label == c].mean(0) for c in pres}
    i = pres.index(classe)
    ant, pos = cent[pres[max(i - 1, 0)]], cent[pres[min(i + 1, len(pres) - 1)]]
    t = (pos - ant) * np.array([1.0, 0.0, 1.0])               # classes 1..16 = direita → esquerda do paciente
    if np.linalg.norm(t) < 1e-6:
        return None, None
    t /= np.linalg.norm(t)
    distal = -t if classe <= 8 else t                         # lado direito (1..8): distal = para a direita = −t
    y = APICAL[arch]
    x = distal - (distal @ y) * y
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    o = cent[classe]
    if z @ ((o - centro_arcada) * np.array([1.0, 0.0, 1.0])) < 0:   # z para fora do arco (vestibular)
        z = -z
    return o, np.stack([x, y, z])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int)
    a = ap.parse_args()
    man = pd.concat([pd.read_csv(DADOS / "manifesto.csv"),
                     pd.read_csv(DADOS / "manifesto_complementar.csv").assign(split="treino")], ignore_index=True)
    if a.n:
        man = man.head(a.n)
    rng = np.random.default_rng(0)
    P, T2, L = [], [], []
    for n, r in enumerate(man.itertuples(), 1):
        jp, rot = achar_json(r.fonte)
        if jp is None:
            continue
        J = json.loads(jp.read_text(encoding="utf-8"))["dentes"]
        gt = {t["planejamento"]["fdi"]: t["planejamento"] for t in J.values()
              if t["planejamento"]["arch"].lower() == r.arch and not t["planejamento"].get("deciduous_fdi", 0)}
        z = np.load(DADOS / "malhas" / r.arquivo)
        bary, nrm, label = z["bary"].astype(float), z["normals"].astype(float), z["label"].astype(int)
        f2p = F2D / f"{r.case_id}.npz"
        f2d = np.load(f2p) if f2p.exists() else None
        arvore_g = cKDTree(bary[label == 0])
        idx_g = np.flatnonzero(label == 0)
        centro = bary[label > 0].mean(0)
        for classe in sorted(int(c) for c in np.unique(label) if c > 0):
            fdi = ORDER16[r.arch][classe - 1]
            if fdi not in gt:
                continue
            o, R = referencial(bary, label, classe, r.arch, centro)
            if o is None:
                continue
            dente = np.flatnonzero(label == classe)
            perto = arvore_g.query_ball_point(bary[dente], MARGEM)
            gengiva = idx_g[np.unique(np.concatenate([np.asarray(p, int) for p in perto]))] if len(perto) else []
            cel = np.r_[dente, gengiva].astype(int)
            sel = rng.choice(len(cel), N_PONTOS, replace=len(cel) < N_PONTOS)
            c = cel[sel]
            pts = np.c_[(bary[c] - o) @ R.T, nrm[c] @ R.T, np.isin(c, dente).astype(float)]
            g = gt[fdi]
            ie = quat_mat(g["initial_rotation"])[:, 2]
            ie = ie * np.sign(ie @ APICAL[r.arch] or 1.0)
            ie_l = R @ ie
            tip = float(np.degrees(np.arctan2(ie_l[0], ie_l[1])))        # ápice para distal = +
            torque = float(np.degrees(np.arctan2(-ie_l[2], ie_l[1])))    # ápice para lingual = +
            # MD do planejador (coluna 0), orientado para distal: NÃO é alvo (seção 19.5), serve para validar contra
            # a reta Mesial → Distal do 3DTeethLand. rot = giro em torno da vertical (ponta distal para vestibular = +)
            md_l = R @ quat_mat(g["initial_rotation"])[:, 0]
            md_l = md_l * np.sign(md_l[0] or 1.0)
            rot = float(np.degrees(np.arctan2(md_l[2], md_l[0])))
            bola = R @ (np.array(g["initial_location"], float) - o)
            tok, tem2d, tip2d, score = np.zeros(1280, np.float16), False, np.nan, 0.0
            if f2d is not None:
                k = [j for j in range(len(f2d["fdi"])) if int(f2d["fdi"][j]) == fdi]
                if k:
                    j = max(k, key=lambda j: float(f2d["score"][j]))
                    tok = np.r_[f2d["feat_box"][j], f2d["feat_mask"][j]].astype(np.float16)
                    tem2d, score = True, float(f2d["score"][j])
                    t2, _ = eixo_2d(f2d, fdi, r.arch)
                    if t2 is not None:                                    # sinal anatômico (como o tip)
                        tip2d = t2 * (-1.0 if fdi // 10 in (1, 4) else 1.0)
            tipo = 8 - min(classe - 1, 16 - classe)
            P.append(pts.astype(np.float32))
            T2.append(tok)
            L.append(dict(case_id=r.case_id, arch=r.arch, split=r.split, fdi=fdi, tipo=tipo, grupo=GRUPO[tipo],
                          tip=tip, torque=torque, bola_x=bola[0], bola_y=bola[1], bola_z=bola[2],
                          ie_x=ie_l[0], ie_y=ie_l[1], ie_z=ie_l[2], md_x=md_l[0], md_y=md_l[1], md_z=md_l[2],
                          rot=rot, tem_2d=tem2d, score_2d=score, tip_2d=tip2d,
                          n_celulas=len(dente)))
        if n % 100 == 0:
            print(f"[{n}/{len(man)}] {len(L)} dentes", flush=True)
    SAIDA.mkdir(exist_ok=True)
    D = pd.DataFrame(L)
    np.savez_compressed(SAIDA / "recortes.npz", pontos=np.stack(P), token=np.stack(T2))
    D.to_csv(SAIDA / "recortes.csv", index=False)
    print(f"{len(D)} dentes de {D.case_id.nunique()} casos | por split {D.split.value_counts().to_dict()} | "
          f"com 2D {D.tem_2d.mean():.1%}")
    print(D.groupby(["arch", "tipo"])[["tip", "torque", "rot"]].agg(["mean", "std"]).round(1).to_string())
    print(f"|MD·IE| mediano {np.median(np.abs((D[['md_x', 'md_y', 'md_z']].values * D[['ie_x', 'ie_y', 'ie_z']].values).sum(1))):.3f} "
          f"(≈0: o MD é ⊥ ao IE)")


if __name__ == "__main__":
    main()
