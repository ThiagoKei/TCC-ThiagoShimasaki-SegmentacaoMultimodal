"""Treino e avaliação do modelo de landmarks (modelo_landmarks.py). Roda em CPU.

Dados (recortes_dentes.py e recortes_3dteethland.py, em <TCC_DADOS>/fusao_dados/landmarks/):
  recortes.{npz,csv}       casos do planejador: eixo IE (alvo) + token 2D; split treino/val/teste (teste = os 141)
  recortes_3dtl.{npz,csv}  3DTeethLand treino: 5 pontos + cúspides; dividido aqui por paciente em treino/val (85/15)

Cada passo junta `--b-eixo` dentes do planejador e `--b-pontos` do 3DTeethLand no MESMO forward (BN comum).
Perda = (1 − cos) no IE + λ · [BCE do calor + L1 do deslocamento perto do landmark] nos pontos.
Modos: conjunto (padrão) | so_eixo | so_pontos (ablação do treino conjunto). --com-2d liga a fusão tardia no eixo.
Aumento: rotação pequena (±--rot graus, eixo aleatório) e escala 0,95–1,05, aplicadas também aos alvos.

Saída: runs_landmarks/<nome>/ (best.pt pelo erro angular do IE no val — ou pelo erro dos pontos em so_pontos —,
historico.csv, args.json). --avaliar: teste dos 141 (+ 2D embaralhado entre dentes do mesmo FDI, se --com-2d) →
pred_teste.csv por dente, e val do 3DTeethLand.
Uso:  python 05_landmarks/treinar_landmarks.py --nome so3d_s42 [--com-2d] [--modo conjunto] [--epocas 60] [--n-max 2000 (teste rápido)]
      python 05_landmarks/treinar_landmarks.py --nome so3d_s42 --avaliar
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
import caminhos  # noqa: E402
from modelo_landmarks import CANAL_DENTE, D_COND, N_CANAIS, ModeloLandmarks, alvos_pontos, extrair_pontos  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
LM = caminhos.FUSAO_DADOS / "landmarks"                      # --dados troca
RUNS = caminhos.RAIZ / "runs_landmarks"
DEV = torch.device("cpu")                                      # --dispositivo cuda na VM


def mover(t):
    """Cópia do lote com os tensores no DEV (o original fica na CPU para as contas em numpy)."""
    return {k: (v.to(DEV, non_blocking=True) if torch.is_tensor(v) else v) for k, v in t.items()}
ESC = 10.0                                   # mm → unidade da rede (coordenadas e deslocamentos / 10)
UNICOS = ["Mesial", "Distal", "FacialPoint", "InnerPoint", "OuterPoint"]
GRUPOS = {0: "incisivo", 1: "canino", 2: "pré-molar", 3: "molar"}


# ── dados ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def condicao(df):
    c = np.zeros((len(df), D_COND), np.float32)
    c[np.arange(len(df)), df.tipo.values - 1] = 1
    c[np.arange(len(df)), 8 + df.grupo.values] = 1
    c[:, 12] = (df.arch.values == "upper")
    return c


def carregar(n_max=None, semente=42):
    z = np.load(LM / "recortes.npz")
    d = pd.read_csv(LM / "recortes.csv")
    plan = dict(df=d, pts=z["pontos"], tok=z["token"], cond=condicao(d),
                ie=d[["ie_x", "ie_y", "ie_z"]].values.astype(np.float32),
                tip2d=np.nan_to_num(d.tip_2d.values).astype(np.float32),
                tem2d=(d.tem_2d.values & d.tip_2d.notna().values).astype(np.float32))
    z3 = np.load(LM / "recortes_3dtl.npz")
    d3 = pd.read_csv(LM / "recortes_3dtl.csv")
    rng = np.random.default_rng(semente)
    pac = d3.id.unique()
    val_ids = set(rng.choice(pac, int(round(0.15 * len(pac))), replace=False))
    d3["split"] = np.where(d3.id.isin(val_ids), "val", "treino")
    tl = dict(df=d3, pts=z3["pontos"], cond=condicao(d3), kp=z3["kp"], kpm=z3["kp_mask"])
    idx = {}
    for nome, base in (("plan", plan), ("tl", tl)):
        for s in ("treino", "val", "teste"):
            i = np.flatnonzero(base["df"].split.values == s)
            if n_max and s == "treino":
                i = i[:n_max]
            idx[(nome, s)] = i
    return plan, tl, idx


def rot_aleatoria(rng, graus):
    if graus <= 0:
        return np.eye(3, dtype=np.float32)
    eixo = rng.normal(size=3)
    eixo /= np.linalg.norm(eixo)
    a = np.radians(rng.uniform(-graus, graus))
    K = np.array([[0, -eixo[2], eixo[1]], [eixo[2], 0, -eixo[0]], [-eixo[1], eixo[0], 0]])
    return (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K).astype(np.float32)


def perturbar_flag(xyz, flag, rng, mm):
    """Estraga a máscara "é dente" do recorte como uma segmentação errada
    faria. Erosão/dilatação de t mm (t uniforme em ±mm) mais um gradiente numa direção aleatória (±mm/2), para o
    erro não ser simétrico. Usa a distância até o outro lado da fronteira (+ dentro do dente). Limite: o recorte só
    tem gengiva a até MARGEM (2 mm) do dente do GT, e não tem os dentes vizinhos."""
    from scipy.spatial import cKDTree
    dente = flag > 0.5
    if dente.all() or not dente.any():
        return flag.copy()
    sd = np.where(dente, cKDTree(xyz[~dente]).query(xyz)[0], -cKDTree(xyz[dente]).query(xyz)[0])
    d = rng.normal(size=3)
    proj = (xyz - xyz.mean(0)) @ (d / np.linalg.norm(d))
    proj /= max(np.abs(proj).max(), 1e-6)
    t = rng.uniform(-mm, mm) + 0.5 * mm * rng.uniform(-1, 1) * proj
    return (sd > t).astype(np.float32)


def lote(base, i, rng, aug, rot, p_pert=0.0, pert_mm=1.0):
    """Tensores de um lote. Com aug: mesma rotação/escala nos pontos, normais e alvos.
    p_pert: fração dos recortes com a máscara de entrada (canal 6) estragada; o alvo "dente" é sempre a limpa."""
    pts = base["pts"][i].copy()
    alvo = {"dente": pts[:, :, 6].copy()}
    if p_pert > 0:
        for b in range(len(i)):
            if rng.random() < p_pert:
                pts[b, :, 6] = perturbar_flag(pts[b, :, :3], pts[b, :, 6], rng, pert_mm)
    if "ie" in base:
        alvo["ie"] = base["ie"][i].copy()
    if "kp" in base:
        alvo["kp"], alvo["kpm"] = base["kp"][i].copy(), base["kpm"][i]
    if aug:
        for b in range(len(i)):
            R, s = rot_aleatoria(rng, rot), rng.uniform(0.95, 1.05)
            pts[b, :, :3] = pts[b, :, :3] @ R.T * s
            pts[b, :, 3:6] = pts[b, :, 3:6] @ R.T
            if "ie" in alvo:
                alvo["ie"][b] = R @ alvo["ie"][b]
            if "kp" in alvo:
                alvo["kp"][b] = alvo["kp"][b] @ R.T * s
    t = dict(pts=torch.from_numpy(pts), cond=torch.from_numpy(base["cond"][i]))
    if "tok" in base:
        t.update(tok=torch.from_numpy(base["tok"][i].astype(np.float32)), tip2d=torch.from_numpy(base["tip2d"][i]),
                 tem2d=torch.from_numpy(base["tem2d"][i]))
    else:
        n = len(i)
        t.update(tok=torch.zeros(n, 1280), tip2d=torch.zeros(n), tem2d=torch.zeros(n))
    t.update({k: torch.from_numpy(np.asarray(v)) for k, v in alvo.items()})
    return t


def entrada(t):
    p = t["pts"].clone()
    p[..., :3] /= ESC
    return p


# ── perdas ────────────────────────────────────────────────────────────────────────────────────────────────────────
def perda_pontos(logits, desloc, t, sigma):
    logits = logits[..., :N_CANAIS]
    calor, alvo_d, valido = alvos_pontos(t["pts"][..., :3], t["kp"], t["kpm"].bool(), sigma)
    v = valido[:, None, :].expand_as(calor).float()
    l_calor = (F.binary_cross_entropy_with_logits(logits, calor, reduction="none") * v).sum() / v.sum().clamp(min=1)
    w = (calor > 0.1).float() * calor * v                          # deslocamento só perto do landmark
    l_desl = (F.smooth_l1_loss(desloc, alvo_d / ESC, reduction="none").sum(-1) * w).sum() / w.sum().clamp(min=1)
    return l_calor, l_desl


def angulos(ie):
    """ie (n, 3) no referencial local → (angulação MD = tip, inclinação VL = torque), graus (mesmo sinal do recorte)."""
    return np.degrees(np.arctan2(ie[:, 0], ie[:, 1])), np.degrees(np.arctan2(-ie[:, 2], ie[:, 1]))


# ── avaliação ─────────────────────────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def prever_eixo(modelo, plan, i, com_2d, embaralhar=False, bs=32, semente=0):
    modelo.eval()
    i = np.asarray(i)
    tok_i = i.copy()
    if embaralhar:                                                 # token de OUTRO paciente com o mesmo FDI
        rng = np.random.default_rng(semente)
        fdi = plan["df"].fdi.values[i]
        for f in np.unique(fdi):
            m = np.flatnonzero(fdi == f)
            tok_i[m] = i[rng.permutation(m)]
    out = []
    for a in range(0, len(i), bs):
        t = lote(plan, i[a:a + bs], None, False, 0)
        if embaralhar:
            j = tok_i[a:a + bs]
            t["tok"] = torch.from_numpy(plan["tok"][j].astype(np.float32))
            t["tip2d"], t["tem2d"] = torch.from_numpy(plan["tip2d"][j]), torch.from_numpy(plan["tem2d"][j])
        t = mover(t)
        _, _, ie = modelo(entrada(t), t["cond"], t["tok"], t["tip2d"], t["tem2d"]) if com_2d else \
            modelo(entrada(t), t["cond"])
        out.append(ie.cpu().numpy())
    return np.concatenate(out)


def erros_eixo(pred, gt):
    ang = np.degrees(np.arccos(np.clip((pred * gt).sum(1), -1, 1)))
    tp, tq = angulos(pred)
    tg, qg = angulos(gt)
    return ang, np.abs(tp - tg), np.abs(tq - qg)


@torch.no_grad()
def avaliar_pontos(modelo, tl, i, bs=32, raio=1.5):
    """Erro (mm) dos 5 pontos únicos e precisão/revocação das cúspides a `raio` mm (casamento guloso)."""
    modelo.eval()
    err = {k: [] for k in UNICOS}
    tp = fp = fn = 0
    for a in range(0, len(i), bs):
        t = lote(tl, i[a:a + bs], None, False, 0)
        td = mover(t)
        lg, ds, _ = modelo(entrada(td), td["cond"])
        lg, ds = lg.cpu().numpy(), ds.cpu().numpy() * ESC
        for b in range(len(lg)):
            xyz, kp, km = t["pts"][b, :, :3].numpy(), t["kp"][b].numpy(), t["kpm"][b].numpy().astype(bool)
            uni, _, cus = extrair_pontos(xyz, lg[b], ds[b])
            for c, k in enumerate(UNICOS):
                if km[c]:
                    err[k].append(float(np.linalg.norm(uni[c] - kp[c])))
            gt_c = kp[5:][km[5:]]
            usados = set()
            for q, _ in cus:
                d = np.linalg.norm(gt_c - q, axis=1) if len(gt_c) else np.array([])
                j = [j for j in np.argsort(d) if j not in usados and d[j] <= raio][:1] if len(d) else []
                if j:
                    usados.add(j[0])
                    tp += 1
                else:
                    fp += 1
            fn += len(gt_c) - len(usados)
    med = {k: float(np.median(v)) for k, v in err.items() if v}
    return med, tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def dsc(a, b):
    """DSC por recorte entre máscaras booleanas (n, N)."""
    return 2 * (a & b).sum(1) / np.maximum(a.sum(1) + b.sum(1), 1)


@torch.no_grad()
def avaliar_contorno(modelo, base, i, pert_mm=1.0, bs=32, semente=0):
    """Refino do contorno no recorte: DSC do dente da máscara de ENTRADA estragada (semente fixa) × da PREVISTA,
    contra a limpa; e da prevista com a entrada limpa (não deve piorar). Devolve arrays por recorte."""
    modelo.eval()
    rng = np.random.default_rng(semente)
    ent, ref, limpo = [], [], []
    for a in range(0, len(i), bs):
        t = lote(base, i[a:a + bs], rng, False, 0, p_pert=1.0, pert_mm=pert_mm)
        gt = t["dente"].numpy() > 0.5
        td = mover(t)
        lg = modelo(entrada(td), td["cond"])[0][..., CANAL_DENTE].cpu().numpy()
        t0 = mover(lote(base, i[a:a + bs], None, False, 0))
        lg0 = modelo(entrada(t0), t0["cond"])[0][..., CANAL_DENTE].cpu().numpy()
        ent.append(dsc(t["pts"][:, :, 6].numpy() > 0.5, gt))
        ref.append(dsc(lg > 0, gt))
        limpo.append(dsc(lg0 > 0, gt))
    return np.concatenate(ent), np.concatenate(ref), np.concatenate(limpo)


# ── treino ────────────────────────────────────────────────────────────────────────────────────────────────────────
def treinar(a):
    torch.manual_seed(a.semente)
    rng = np.random.default_rng(a.semente)
    plan, tl, idx = carregar(a.n_max, a.semente)
    usa_eixo, usa_pontos = a.modo in ("conjunto", "so_eixo"), a.modo in ("conjunto", "so_pontos")
    saida = RUNS / a.nome
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "args.json").write_text(json.dumps(vars(a), indent=2, default=str))
    modelo = ModeloLandmarks(com_2d=a.com_2d).to(DEV)
    opt =torch.optim.AdamW(modelo.parameters(), lr=a.lr, weight_decay=1e-4)
    passos = len(idx[("plan", "treino")]) // a.b_eixo if usa_eixo else len(idx[("tl", "treino")]) // a.b_pontos
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epocas * passos, pct_start=0.1)
    tr_p, tr_t = idx[("plan", "treino")], idx[("tl", "treino")]
    print(f"modo {a.modo} | 2D {a.com_2d} | planejador treino {len(tr_p)} val {len(idx[('plan', 'val')])} | "
          f"3DTeethLand treino {len(tr_t)} val {len(idx[('tl', 'val')])} | {passos} passos/época | "
          f"threads {torch.get_num_threads()}", flush=True)
    melhor, hist, parado = np.inf, [], 0
    for ep in range(a.epocas):
        modelo.train()
        t0, soma = time.time(), dict(ie=0.0, calor=0.0, desl=0.0, seg=0.0)
        perm_p, perm_t = rng.permutation(tr_p), rng.permutation(tr_t)
        for s in range(passos):
            partes = []
            if usa_eixo:
                partes.append(("plan", mover(lote(plan, perm_p[s * a.b_eixo:(s + 1) * a.b_eixo], rng, True, a.rot,
                                                  a.p_perturba, a.pert_mm))))
            if usa_pontos:
                j = (np.arange(a.b_pontos) + s * a.b_pontos) % len(perm_t)
                partes.append(("tl", mover(lote(tl, perm_t[j], rng, True, a.rot, a.p_perturba, a.pert_mm))))
            pts = torch.cat([entrada(t) for _, t in partes])
            cond = torch.cat([t["cond"] for _, t in partes])
            tok = torch.cat([t["tok"] for _, t in partes])
            tip2d = torch.cat([t["tip2d"] for _, t in partes])
            tem2d = torch.cat([t["tem2d"] for _, t in partes])
            if a.com_2d and a.p_sem_2d > 0:                        # dropout de modalidade (robustez)
                tem2d = tem2d * (torch.rand_like(tem2d) >= a.p_sem_2d)
            lg, ds, ie = modelo(pts, cond, tok, tip2d, tem2d) if a.com_2d else modelo(pts, cond)
            perda, n0 = 0.0, 0
            for nome, t in partes:
                n = len(t["pts"])
                if nome == "plan":
                    l_ie = (1 - (ie[n0:n0 + n] * t["ie"]).sum(1)).mean()
                    perda = perda + l_ie
                    soma["ie"] += l_ie.item()
                else:
                    lc, ld = perda_pontos(lg[n0:n0 + n], ds[n0:n0 + n], t, a.sigma)
                    perda = perda + a.lam * (lc + ld)
                    soma["calor"] += lc.item()
                    soma["desl"] += ld.item()
                if a.lam_seg > 0:                                  # bônus: refino do contorno (os dois conjuntos)
                    l_seg = F.binary_cross_entropy_with_logits(lg[n0:n0 + n, :, CANAL_DENTE], t["dente"])
                    perda = perda + a.lam_seg * l_seg
                    soma["seg"] += l_seg.item()
                n0 += n
            opt.zero_grad(set_to_none=True)
            perda.backward()
            opt.step()
            sched.step()
        r = dict(epoca=ep + 1, **{f"perda_{k}": v / passos for k, v in soma.items()}, seg=time.time() - t0)
        if usa_eixo:
            iv = idx[("plan", "val")]
            ang, etp, etq = erros_eixo(prever_eixo(modelo, plan, iv, a.com_2d), plan["ie"][iv])
            r.update(val_ang=float(ang.mean()), val_ang_med=float(np.median(ang)), val_tip=float(etp.mean()),
                     val_torque=float(etq.mean()))
            if a.com_2d:
                ang0, _, _ = erros_eixo(prever_eixo(modelo, plan, iv, True, embaralhar=True), plan["ie"][iv])
                r["val_ang_emb"] = float(ang0.mean())
        if a.lam_seg > 0:
            de, dr, dl = avaliar_contorno(modelo, plan if usa_eixo else tl,
                                          idx[("plan", "val")] if usa_eixo else idx[("tl", "val")], a.pert_mm)
            r.update(val_dsc_entrada=float(de.mean()), val_dsc_refino=float(dr.mean()), val_dsc_limpo=float(dl.mean()))
        if usa_pontos and ((ep + 1) % a.cada_pontos == 0 or ep + 1 == a.epocas or not usa_eixo):
            med, prec, rev = avaliar_pontos(modelo, tl, idx[("tl", "val")])
            r.update({f"val_{k}": v for k, v in med.items()}, val_cusp_prec=prec, val_cusp_rev=rev)
        crit = r["val_ang"] if usa_eixo else float(np.mean([r[f"val_{k}"] for k in UNICOS]))
        melhorou = crit < melhor
        if melhorou:
            melhor, parado = crit, 0
            torch.save(dict(modelo=modelo.state_dict(), args=vars(a), epoca=ep + 1, crit=crit), saida / "best.pt")
        else:
            parado += 1
        hist.append(r)
        pd.DataFrame(hist).to_csv(saida / "historico.csv", index=False)
        print(" | ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items())
              + (" *" if melhorou else ""), flush=True)
        if parado >= a.paciencia:
            print(f"early stopping ({a.paciencia} épocas sem melhorar)")
            break


def avaliar(a):
    saida = RUNS / a.nome
    ck = torch.load(saida / "best.pt", map_location="cpu", weights_only=False)
    args = ck["args"]
    modelo = ModeloLandmarks(com_2d=args["com_2d"])
    modelo.load_state_dict(ck["modelo"])
    modelo.to(DEV)
    plan, tl, idx = carregar(None, args["semente"])
    print(f"{a.nome}: época {ck['epoca']} (critério val {ck['crit']:.3f})")
    if args["modo"] != "so_pontos":
        it = idx[("plan", "teste")]
        d = plan["df"].iloc[it][["case_id", "arch", "fdi", "tipo", "grupo", "tip", "torque", "tem_2d"]].copy()
        pred = prever_eixo(modelo, plan, it, args["com_2d"])
        d["ang"], d["err_tip"], d["err_torque"] = erros_eixo(pred, plan["ie"][it])
        d["tip_pred"], d["torque_pred"] = angulos(pred)
        if args["com_2d"]:
            pe = prever_eixo(modelo, plan, it, True, embaralhar=True)
            d["ang_emb"], d["err_tip_emb"], d["err_torque_emb"] = erros_eixo(pe, plan["ie"][it])
        # piso: média do IE por (arcada, tipo) no treino
        tr = idx[("plan", "treino")]
        dtr = plan["df"].iloc[tr]
        media = {k: plan["ie"][tr][g.values].mean(0) for k, g in
                 dtr.reset_index(drop=True).groupby(["arch", "tipo"]).groups.items()}
        pm = np.stack([media[(r.arch, r.tipo)] for r in d.itertuples()])
        pm /= np.linalg.norm(pm, axis=1, keepdims=True)
        d["ang_media"], d["err_tip_media"], d["err_torque_media"] = erros_eixo(pm, plan["ie"][it])
        d.to_csv(saida / "pred_teste.csv", index=False)
        cols = [c for c in d.columns if c.startswith(("ang", "err_"))]
        print("\nteste (141 casos), erro médio em graus por grupo:")
        print(d.groupby(d.grupo.map(GRUPOS))[cols].mean().round(2).to_string())
        print("todos:", d[cols].mean().round(2).to_dict())
        print("% IE < 5°:", round(float((d.ang < 5).mean() * 100), 1), "| média por tipo:",
              round(float((d.ang_media < 5).mean() * 100), 1))
    if args.get("lam_seg", 0) > 0 and args["modo"] != "so_pontos":
        it = idx[("plan", "teste")]
        de, dr, dl = avaliar_contorno(modelo, plan, it, args.get("pert_mm", 1.0))
        g = plan["df"].grupo.values[it]
        print("\nrefino do contorno no recorte (teste, máscara de entrada estragada ±%.1f mm; DSC do dente):"
              % args.get("pert_mm", 1.0))
        for k, nome in GRUPOS.items():
            m = g == k
            print(f"  {nome:10s} entrada {de[m].mean():.3f} → refinada {dr[m].mean():.3f} | entrada limpa → "
                  f"{dl[m].mean():.3f}")
        print(f"  todos      entrada {de.mean():.3f} → refinada {dr.mean():.3f} | entrada limpa → {dl.mean():.3f}")
    if args["modo"] != "so_eixo":
        med, prec, rev = avaliar_pontos(modelo, tl, idx[("tl", "val")])
        print("\n3DTeethLand val: erro mediano (mm)", {k: round(v, 2) for k, v in med.items()},
              f"| cúspides @1,5 mm: precisão {prec:.3f} revocação {rev:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", required=True)
    ap.add_argument("--avaliar", action="store_true")
    ap.add_argument("--modo", choices=["conjunto", "so_eixo", "so_pontos"], default="conjunto")
    ap.add_argument("--com-2d", action="store_true")
    ap.add_argument("--p-sem-2d", type=float, default=0.0, help="dropout de modalidade no treino (0 = desligado)")
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--paciencia", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--b-eixo", type=int, default=24)
    ap.add_argument("--b-pontos", type=int, default=8)
    ap.add_argument("--lam", type=float, default=1.0, help="peso da perda dos pontos")
    ap.add_argument("--sigma", type=float, default=1.5, help="mm, largura do calor")
    ap.add_argument("--rot", type=float, default=10.0, help="graus, rotação do aumento de dados")
    ap.add_argument("--cada-pontos", type=int, default=5, help="avaliar os pontos no val a cada N épocas")
    ap.add_argument("--lam-seg", type=float, default=0.5, help="peso do refino do contorno; 0 = sem")
    ap.add_argument("--p-perturba", type=float, default=0.5, help="fração dos recortes com a máscara estragada")
    ap.add_argument("--pert-mm", type=float, default=1.0, help="mm de erosão/dilatação da máscara estragada")
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--n-max", type=int, help="limita o treino (teste rápido)")
    ap.add_argument("--dispositivo", default="cpu", help="cpu | cuda")
    ap.add_argument("--dados", help="pasta com recortes*.{npz,csv} (padrão: <TCC_DADOS>/fusao_dados/landmarks)")
    a = ap.parse_args()
    global DEV, LM
    DEV = torch.device(a.dispositivo)
    if a.dados:
        LM = Path(a.dados)
    if a.threads:
        torch.set_num_threads(a.threads)
    avaliar(a) if a.avaliar else treinar(a)


if __name__ == "__main__":
    main()
