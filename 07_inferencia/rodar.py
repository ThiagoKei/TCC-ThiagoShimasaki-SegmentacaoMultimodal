"""Modelo completo do TCC: escaneamento 3D da arcada INFERIOR + panorâmica → segmentação, numeração FDI e landmarks.

Uso:
  python 07_inferencia/rodar.py --inferior arcada_inferior.stl --panoramica panoramica.png --saida resultado/

Cadeia (ver MANUAL.md):
  1 orientação → 2 decimação (8.000 células) → 3 G4-S (só 3D: rascunho) → 4 Mask R-CNN na panorâmica (tokens)
  → 5 registro panorâmica ↔ malha → 6 G4-FR (fusão 3D + panorâmica) → 7 graph-cut (λ = 7) → 8 landmarks
  → 9 volta ao referencial e à resolução do arquivo original.
"""
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
import caminhos  # noqa: E402

PESOS = caminhos.PESOS   # g4s.pt, g4fr.pt, landmarks_s42/43/44.pt, maskrcnn_d2_flipswap.pth (--pesos troca)
LAMBDA_FR = 7.0          # λ do graph-cut escolhido no val para o G4-FR
MIN_FRAC_ILHA = 0.005    # remove_islands
CASO = 1                 # identificador interno (o dataset do treino usa um case_id inteiro)
ORDER16 = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
PALETA = ["#8c564b", "#e377c2", "#ff7f0e", "#bcbd22", "#2ca02c", "#17becf", "#1f77b4", "#9467bd",
          "#c5b0d5", "#aec7e8", "#9edae5", "#98df8a", "#dbdb8d", "#ffbb78", "#f7b6d2", "#c49c94"]
GENGIVA = "#e9d9cf"
# Candidatos de sinal dos eixos (rotações próprias): o G4-S decide qual deixa a arcada "reconhecível".
CANDIDATOS = {"original": np.diag([1.0, 1.0, 1.0]), "vertical invertida": np.diag([-1.0, -1.0, 1.0])}


class ErroEntrada(Exception):
    pass


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── entrada ──────────────────────────────────────────────────────────────────────────────────────────────────────
def ler_malha(p):
    import vedo
    p = Path(p)
    if not p.exists():
        raise ErroEntrada(f"arquivo não encontrado: {p}")
    try:
        m = vedo.load(str(p))
    except Exception as e:                                       # noqa: BLE001
        raise ErroEntrada(f"não consegui abrir {p.name} ({e}); use STL, PLY, OBJ ou VTP") from e
    if m is None or m.npoints == 0:
        raise ErroEntrada(f"{p.name} está vazio ou num formato que o vedo não lê")
    m = m.clean().triangulate()
    verts = np.asarray(m.points, np.float64)
    faces = np.asarray(m.cells, np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1000:
        raise ErroEntrada(f"{p.name}: precisa ser uma malha de triângulos com pelo menos 1.000 faces")
    ext = np.ptp(verts, 0).max()
    if ext < 1:
        raise ErroEntrada(f"{p.name}: a maior dimensão é {ext:.3f}; parece estar em metros. Exporte em milímetros.")
    if not 30 <= ext <= 200:
        log(f"AVISO: {p.name} mede {ext:.0f} unidades na maior dimensão; uma arcada tem ~60–90 mm. "
            "Confira se está em milímetros e se é uma arcada só.")
    return verts, faces


def comprimento_de_arco(pts, grau=4):
    """Coordenada s ∈ [0, 1] ao longo do arco (cópia de fusao/preparar_fusao.py)."""
    from scipy.spatial import cKDTree
    c = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    eixos = vt[:2]
    if abs(eixos[1, 0]) > abs(eixos[0, 0]):
        eixos = eixos[::-1]
    if eixos[0, 0] < 0:
        eixos = np.array([-eixos[0], eixos[1]])
    uv = c @ eixos.T
    coef = np.polyfit(uv[:, 0], uv[:, 1], grau)
    g = np.linspace(uv[:, 0].min(), uv[:, 0].max(), 3000)
    curva = np.c_[g, np.polyval(coef, g)]
    s_grid = np.r_[0, np.cumsum(np.hypot(*np.diff(curva, axis=0).T))]
    _, idx = cKDTree(curva).query(uv)
    return (s_grid[idx] / s_grid[-1]).astype(np.float32)


# ── modelos de segmentação (mesmo caminho de código do teste do TCC: DatasetFusao + passo) ──────────────────────
def carregar_seg(nome):
    import torch
    import treinar_fusao as tf
    ck = torch.load(PESOS / nome, map_location="cpu", weights_only=False)
    m = tf.novo_modelo(ck)
    m.load_state_dict(ck["model_state_dict"])
    return m.eval()


def segmentar(modelo, raiz, pasta_2d, dev):
    import torch
    from torch.utils.data import DataLoader
    import treinar_fusao as tf
    from dados_fusao import DatasetFusao, juntar
    ds = DatasetFusao(raiz, "teste", "lower", patch=None, pasta_2d=pasta_2d)
    b = next(iter(DataLoader(ds, batch_size=1, collate_fn=juntar)))
    with torch.no_grad():
        prob, _ = tf.passo(modelo.to(dev), tf.para_device(b, dev))
    return prob[0].float().cpu().numpy()


def gravar_caso(raiz, z, registro=None):
    (raiz / "malhas").mkdir(parents=True, exist_ok=True)
    (raiz / "registro").mkdir(exist_ok=True)
    with open(raiz / "manifesto.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "arch", "split", "arquivo"])
        w.writerow([CASO, "lower", "teste", f"{CASO}_lower.npz"])
    np.savez(raiz / "malhas" / f"{CASO}_lower.npz", **z)
    reg = raiz / "registro" / f"{CASO}_lower.npz"
    if registro is not None:
        np.savez(reg, **registro)
    elif reg.exists():
        reg.unlink()


def carregar_gco(sc):
    """malha.load_gco(); se outro pacote 'pygco' sombrear o do gco-wrapper, importa gco.pygco direto."""
    cut = sc.load_gco()
    if cut is not None:
        return cut
    try:
        from gco.pygco import cut_general_graph
    except Exception:                                            # noqa: BLE001
        return None

    def cut_from_graph_float(edge_nodes, edge_weights, unaries, pairwise):
        return cut_general_graph(np.ascontiguousarray(edge_nodes, dtype=np.int32),
                                 np.ascontiguousarray(edge_weights, dtype=np.float64),
                                 np.ascontiguousarray(unaries, dtype=np.float64),
                                 np.ascontiguousarray(pairwise, dtype=np.float64), algorithm="expansion")
    return cut_from_graph_float


# ── saídas ───────────────────────────────────────────────────────────────────────────────────────────────────────
def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def cor_fdi(fdi):
    return hex_rgb(GENGIVA if fdi == 0 else PALETA[ORDER16.index(fdi)])


def salvar_ply_malha(p, verts, faces, fdi_face):
    lut = {f: cor_fdi(f) for f in [0] + ORDER16}
    cf = np.array([lut[int(f)] for f in fdi_face], np.uint8)
    cv = np.zeros((len(verts), 3), np.uint8) + np.array(hex_rgb(GENGIVA), np.uint8)
    dente = fdi_face > 0                                  # vértice herda a cor de um dente que o toca
    cv[faces[~dente].ravel()] = np.repeat(cf[~dente], 3, axis=0)
    cv[faces[dente].ravel()] = np.repeat(cf[dente], 3, axis=0)
    vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    fdt = np.dtype([("n", "u1"), ("i", "<i4", 3), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("fdi", "<i4")])
    va = np.empty(len(verts), vdt)
    va["x"], va["y"], va["z"] = verts.T.astype(np.float32)
    va["r"], va["g"], va["b"] = cv.T
    fa = np.empty(len(faces), fdt)
    fa["n"], fa["i"] = 3, faces.astype(np.int32)
    fa["r"], fa["g"], fa["b"] = cf.T
    fa["fdi"] = fdi_face.astype(np.int32)
    cab = ("ply\nformat binary_little_endian 1.0\ncomment segmentacao TCC (fdi por face; 0 = gengiva)\n"
           f"element vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           f"element face {len(faces)}\nproperty list uchar int vertex_indices\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty int fdi\nend_header\n")
    with open(p, "wb") as fh:
        fh.write(cab.encode("ascii"))
        fh.write(va.tobytes())
        fh.write(fa.tobytes())


CORES_LM = {"Mesial": "#e74c3c", "Distal": "#3498db", "FacialPoint": "#f1c40f", "InnerPoint": "#9b59b6",
            "OuterPoint": "#1abc9c", "Cuspide": "#ffffff", "Centro": "#000000"}


def salvar_ply_pontos(p, pontos):
    linhas = [f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}" for (x, y, z), (r, g, b) in pontos]
    cab = ("ply\nformat ascii 1.0\ncomment landmarks TCC\n"
           f"element vertex {len(linhas)}\nproperty float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    Path(p).write_text(cab + "\n".join(linhas) + "\n", encoding="ascii")


def salvar_html(p, verts_d, faces_d, fdi_d, lms, titulo):
    import base64
    v = verts_d.astype(np.float32)
    dados = dict(
        v=base64.b64encode(v.tobytes()).decode(), f=base64.b64encode(faces_d.astype(np.uint32).tobytes()).decode(),
        c=[list(cor_fdi(int(x))) for x in fdi_d], ordem=ORDER16, paleta=PALETA,
        lm=[dict(fdi=int(f), tipo=t, p=[round(float(x), 3) for x in q], cor=CORES_LM[t]) for f, L in lms.items()
            for t, q in ([(k, L["pontos"][k]) for k in L["pontos"]] + [("Cuspide", c) for c in L["cuspides"]])],
        eixos=[dict(fdi=int(f), o=[float(x) for x in L["centro"]], d=[float(x) for x in L["eixo_IE"]])
               for f, L in lms.items()])
    html = (AQUI / "visualizar_modelo.html").read_text(encoding="utf-8")
    html = html.replace("/*TITULO*/", titulo).replace("/*DADOS*/null", json.dumps(dados))
    Path(p).write_text(html, encoding="utf-8")


# ── cadeia ───────────────────────────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Segmentação + numeração FDI + landmarks da arcada inferior (TCC).")
    ap.add_argument("--inferior", required=True, help="escaneamento da arcada INFERIOR (STL/PLY/OBJ/VTP, em mm)")
    ap.add_argument("--panoramica", required=True, help="radiografia panorâmica do mesmo paciente (PNG/JPG)")
    ap.add_argument("--saida", required=True, help="pasta de saída")
    ap.add_argument("--sem-graphcut", action="store_true", help="não aplica o graph-cut (saída crua da rede)")
    ap.add_argument("--manter-tmp", action="store_true", help="não apaga a pasta _tmp (depuração)")
    ap.add_argument("--dispositivo", default=None, help="cuda ou cpu (padrão: cuda se houver)")
    ap.add_argument("--pesos", type=Path, default=None, help="pasta dos pesos (padrão: <TCC_DADOS>/pesos)")
    a = ap.parse_args()
    global PESOS
    if a.pesos:
        PESOS = a.pesos
    try:
        rodar(a)
    except ErroEntrada as e:
        sys.exit(f"\nERRO na entrada: {e}")


def rodar(a):
    import torch
    import orientacao as orient
    import pipeline_2d as p2d
    import registro as rg
    import landmarks as lmk
    import malha as sc
    import treinar_fusao as tf

    dev = torch.device(a.dispositivo or ("cuda" if torch.cuda.is_available() else "cpu"))
    tf.configurar_precisao("fp32")
    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    tmp = saida / "_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    t0 = time.time()

    # 1. leitura e orientação
    verts, faces = ler_malha(a.inferior)
    log(f"1. malha inferior: {len(verts)} vértices, {len(faces)} faces")
    try:
        o = orient.orienta(verts, faces)
    except orient.OrientacaoIncerta as e:
        raise ErroEntrada(f"não consegui orientar a arcada automaticamente: {e}") from e
    R0, c = np.asarray(o["R"], float), np.asarray(o["centro"], float)
    log(f"   orientação: {o['metodo']} (razão das pontas do U {o['razao_ponta']:.2f})")

    # 2. decimação (no referencial orientado; a decimação não depende da rotação)
    V0 = (verts - c) @ R0.T
    vd, fd, bd, nd = sc.decimate_to_cells(V0, faces, sc.TARGET_CELLS, label="lower")
    log(f"2. decimação: {len(fd)} células")

    # 3. G4-S nos candidatos de sinal; fica o de maior confiança média (o modelo não reconhece a arcada de ponta
    #    cabeça). Depois, o rascunho.
    g4s = carregar_seg("g4s.pt")
    raiz = tmp / "dados"
    melhor = None
    confs = {}
    cands = dict(CANDIDATOS)
    if o.get("ap_incerto"):                    # forma do U não decide frente/trás: o G4-S decide também
        cands.update({"frente invertida": np.diag([-1.0, 1.0, -1.0]), "as duas invertidas": np.diag([1.0, -1.0, -1.0])})
        log("   AVISO: frente/trás ambíguos pela forma; testando também a frente invertida")
    for nome, F in cands.items():
        z = dict(verts=(vd @ F.T).astype(np.float32), faces=fd.astype(np.int32), bary=(bd @ F.T).astype(np.float32),
                 normals=(nd @ F.T).astype(np.float32), label=np.zeros(len(fd), np.int8),
                 arco=comprimento_de_arco(bd @ F.T), rascunho_label=np.zeros(len(fd), np.int8),
                 rascunho_conf=np.zeros(len(fd), np.float16))
        gravar_caso(raiz, z)
        prob = segmentar(g4s, raiz, tmp / "f2d_vazio", dev)
        confs[nome] = float(prob.max(1).mean())
        if melhor is None or confs[nome] > melhor[0]:
            melhor = (confs[nome], nome, F, z, prob)
    _, nome, F, z, prob_s = melhor
    R = F @ R0
    log(f"3. G4-S: confiança média por candidato {', '.join(f'{k} {v:.3f}' for k, v in confs.items())} → {nome}")
    if nome != "original":
        log("   (o sinal vertical da orientação geométrica foi corrigido pelo G4-S)")
    rascunho = prob_s.argmax(1).astype(np.int8)
    z["rascunho_label"], z["rascunho_conf"] = rascunho, prob_s.max(1).astype(np.float16)
    z["arco"], arco_inv = rg.orientar_arco(z["arco"], rascunho)
    n_d = len(set(rascunho[rascunho > 0].tolist()))
    log(f"   rascunho: {n_d} dentes")

    # 4. panorâmica → tokens
    img = p2d.imread(a.panoramica)
    pred = p2d.build_predictor(PESOS / "maskrcnn_d2_flipswap.pth", "cuda" if dev.type == "cuda" else "cpu")
    cap = p2d.instalar_captura(pred)
    f2d, _ = p2d.extrair(pred, cap, img)
    del pred
    (tmp / "f2d").mkdir(parents=True, exist_ok=True)
    np.savez(tmp / "f2d" / f"{CASO}.npz", **f2d)
    inf2d = sorted(int(f) for f in f2d["fdi"] if f >= 30)
    log(f"4. panorâmica {img.shape[1]}×{img.shape[0]}: {len(f2d['fdi'])} dentes detectados, inferiores {inf2d}")
    if len(inf2d) < 4:
        log("   AVISO: menos de 4 dentes inferiores na panorâmica; a fusão fica praticamente só 3D")

    # 5. registro panorâmica ↔ malha
    reg, msg = rg.registrar_arcada(z["arco"], rascunho.astype(int), f2d, "lower")
    log(f"5. registro: {msg}" + ("" if reg else " → atenção livre (sem prior)"))

    # 6. G4-FR
    gravar_caso(raiz, z, reg)
    g4fr = carregar_seg("g4fr.pt")
    prob = segmentar(g4fr, raiz, tmp / "f2d", dev)
    log(f"6. G4-FR: {len(set(prob.argmax(1)[prob.argmax(1) > 0].tolist()))} dentes")

    # 7. graph-cut
    lab = prob.argmax(1).astype(np.int64)
    cut = None if a.sem_graphcut else carregar_gco(sc)
    if cut is not None:
        lab, (src, dst) = sc.refine_graphcut(prob.astype(np.float64), fd.astype(np.int64), (bd @ F.T).astype(float),
                                             (nd @ F.T).astype(float), LAMBDA_FR, cut)
        lab = sc.remove_islands(lab, src, dst, len(lab), min_frac=MIN_FRAC_ILHA, num_classes=17)
        log(f"7. graph-cut λ = {LAMBDA_FR:g} + ilhas: {len(set(lab[lab > 0].tolist()))} dentes")
    else:
        log("7. graph-cut PULADO" + ("" if a.sem_graphcut else " (pacote gco não instalado: pip install gco-wrapper)"))

    # 8. landmarks (referencial orientado)
    bdo, ndo = (bd @ F.T).astype(float), (nd @ F.T).astype(float)
    modelos = lmk.carregar_modelos([PESOS / f"landmarks_s{s}.pt" for s in (42, 43, 44)])
    lms_o = lmk.landmarks_globais(modelos, bdo, ndo, lab)
    log(f"8. landmarks: {len(lms_o)} dentes")

    # 9. de volta ao arquivo original: p_orig = p_or @ R + c ; vetor_orig = v_or @ R
    volta = lambda p: np.asarray(p) @ R + c           # noqa: E731
    lms = {f: dict(eixo_IE=np.asarray(L["eixo_IE"]) @ R, centro=volta(L["centro"]),
                   pontos={k: volta(q) for k, q in L["pontos"].items()},
                   cuspides=[volta(q) for q in L["cuspides"]], n_celulas=L["n_celulas"]) for f, L in lms_o.items()}
    Vor = (verts - c) @ R.T
    bary_full = Vor[faces].mean(1)
    lab_full = sc.upsample(lab, bdo, bary_full, k=3, num_classes=17)
    src_f, dst_f = sc.cell_graph(faces)
    lab_full = sc.smooth_labels(lab_full, src_f, dst_f, num_classes=17)
    l2f = np.array([0] + ORDER16)
    fdi_full, fdi_d = l2f[lab_full], l2f[lab]
    salvar_ply_malha(saida / "segmentacao.ply", verts, faces, fdi_full)

    # tabelas
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    linhas = []
    for k in sorted(set(lab[lab > 0].tolist())):
        f = ORDER16[k - 1]
        cel = lab == k
        linhas.append(dict(fdi=f, celulas_8k=int(cel.sum()), faces_malha=int((fdi_full == f).sum()),
                           area_mm2=round(float(area[fdi_full == f].sum()), 1),
                           confianca=round(float(prob[cel, k].mean()), 3),
                           na_panoramica=f in inf2d, landmarks=f in lms))
    with open(saida / "dentes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]) if linhas else ["fdi"])
        w.writeheader()
        w.writerows(linhas)
    r3 = lambda p: [round(float(x), 3) for x in p]    # noqa: E731
    js = {"referencial": "o do arquivo de entrada (mesmas unidades)",
          "dentes": {str(f): dict(eixo_IE=r3(L["eixo_IE"]), centro=r3(L["centro"]),
                                  pontos={k: r3(q) for k, q in L["pontos"].items()},
                                  cuspides=[r3(q) for q in L["cuspides"]], n_celulas_8k=int(L["n_celulas"]))
                     for f, L in sorted(lms.items())}}
    (saida / "landmarks.json").write_text(json.dumps(js, indent=1), encoding="utf-8")
    pts = []
    for L in lms.values():
        pts += [(q, hex_rgb(CORES_LM[k])) for k, q in L["pontos"].items()]
        pts += [(q, hex_rgb(CORES_LM["Cuspide"])) for q in L["cuspides"]]
        pts += [(L["centro"], hex_rgb(CORES_LM["Centro"]))]
    salvar_ply_pontos(saida / "landmarks.ply", pts)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ c
    (saida / "orientacao.json").write_text(json.dumps(dict(
        metodo=o["metodo"], candidato_escolhido=nome, confianca_g4s=confs, razao_ponta=o["razao_ponta"],
        arco_invertido_pelo_rascunho=arco_inv,
        matriz_original_para_modelo=T.round(6).tolist(),
        nota="p_modelo = M @ [p_original, 1]; referencial do modelo: Y oclusal (coroas para +Y), +Z anterior"),
        indent=1), encoding="utf-8")
    vd_orig = (vd @ F.T) @ R + c
    salvar_html(saida / "visualizar.html", vd_orig, fd, fdi_d, lms, Path(a.inferior).name)
    log(f"9. pronto em {time.time() - t0:.0f} s → {saida}")
    print(f"\n   dentes (FDI): {[d['fdi'] for d in linhas]}")
    print("   arquivos: segmentacao.ply, dentes.csv, landmarks.json, landmarks.ply, orientacao.json, visualizar.html")
    if not a.manter_tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
