"""Teeth3DS+ / 3DTeethLand (dados públicos): baixar, compactar e preparar para o MeshSegNet do TCC.

Subcomandos (rodar nesta ordem):
  baixar      baixa o Teeth3DS+ (osf.io/xctdy, CC BY-NC-ND 4.0), uma parte por vez, e guarda só os scans dos
              conjuntos de teste e do 3DTeethLand (1.274 scans). Resumível: pula as partes com feito_<parte>.txt.
              Splits (GitHub do desafio, em <TCC_DADOS>/teeth3ds/splits/):
                Teeth3DS oficial  testing_{upper,lower}.txt            600 scans
                desafio 2022      private-testing-set.txt              600 scans (174 em comum com o oficial)
                3DTeethLand       3DTeethLand_{training,testing}_*.txt 340 scans com landmarks
              Landmarks (3DTeethLand_landmarks_{train,test}.zip) baixados à parte em <TCC_DADOS>/teeth3ds/.
  compactar   troca cada OBJ + JSON por um .npz (verts, faces, labels = FDI por vértice com 0 = gengiva,
              instances, jaw, id_patient), confere a leitura e apaga o OBJ e o JSON (~17,5 GB → ~4 GB).
              --vigiar repete a cada 60 s enquanto o download roda.
  preparar    gira para a convenção do modelo (normal oclusal = Y, anterior = +Z; rotação fixa ROTACAO, medida em
              20 scans de cada base), recorta 22 mm a partir das cúspides, decima para ~8.000 células
              (00_comum/malha.py), leva o rótulo vértice → face → célula (17 classes) e guarda, para avaliar na
              malha cheia, a célula decimada mais próxima de cada vértice original.
              Saída: <TCC_DADOS>/teeth3ds/prep/<ID>_<arch>.npz e prep/manifesto.csv (resumível).
              --conjunto landmarks_treino: os 240 de treino do 3DTeethLand (manifesto_landmarks_treino.csv).

Uso:
  python 01_dados/teeth3ds.py baixar
  python 01_dados/teeth3ds.py compactar [--vigiar]
  python 01_dados/teeth3ds.py preparar [--conjunto teste|landmarks_treino] [--n N]
"""
import argparse
import csv
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
import caminhos  # noqa: E402
import malha as sc  # noqa: E402
from preparar_fusao import ORDER16  # noqa: E402



# ──── baixar ────────────────────────────────────────────────────────────────────────────────────────────────────────
RAIZ = caminhos.TEETH3DS
PASTA_SPLITS = RAIZ / "splits"
PARTES = {   # nó OSF de cada parte (filhos de xctdy)
    "data_part_1": "5cmg3", "data_part_2": "xfcn9", "data_part_3": "hw7bj", "data_part_4": "n9bd7",
    "data_part_5": "7az58", "data_part_6": "2ybr4", "data_part_7": "gvjah",
}


def ids_para_baixar():
    ids = set()
    for f in ("Teeth3DS_split_testing_upper.txt", "Teeth3DS_split_testing_lower.txt",
              "3DTeethSeg_challenge_split_private-testing-set.txt",
              # 3DTeethLand (landmarks, 340 scans; 248 fora dos testes de segmentação): para a Etapa ⑤
              "3DTeethLand_training_upper.txt", "3DTeethLand_training_lower.txt",
              "3DTeethLand_testing_upper.txt", "3DTeethLand_testing_lower.txt"):
        ids |= set((PASTA_SPLITS / f).read_text().split())
    return ids


def info_osf(no):
    with urllib.request.urlopen(f"https://api.osf.io/v2/nodes/{no}/files/osfstorage/", timeout=60) as r:
        f = json.load(r)["data"][0]
    return f["links"]["download"], int(f["attributes"]["size"])


def baixar(url, destino, tamanho):
    """Download com retomada: pede só o que falta (Range) e tenta de novo se a conexão cair."""
    while True:
        feito = destino.stat().st_size if destino.exists() else 0
        if feito >= tamanho:
            return
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={feito}-", "User-Agent": "tcc-download"})
            with urllib.request.urlopen(req, timeout=120) as r, open(destino, "ab" if feito else "wb") as out:
                if feito and r.status != 206:      # servidor ignorou o Range: recomeça do zero
                    out.truncate(0)
                t0, n0 = time.time(), feito
                while True:
                    bloco = r.read(1 << 20)
                    if not bloco:
                        break
                    out.write(bloco)
                    feito += len(bloco)
                    if time.time() - t0 > 60:
                        print(f"  {destino.name}: {feito / 1e9:.2f}/{tamanho / 1e9:.2f} GB "
                              f"({(feito - n0) / (time.time() - t0) / 1024:.0f} KB/s)", flush=True)
                        t0, n0 = time.time(), feito
        except Exception as e:                      # noqa: BLE001 — queda de rede: espera e retoma
            print(f"  {destino.name}: erro {e!r}; retomando em 30 s", flush=True)
            time.sleep(30)


def main_baixar():
    ids = ids_para_baixar()
    print(f"{len(ids)} scans de teste (união dos dois splits)", flush=True)
    saida = RAIZ / "scans"
    saida.mkdir(parents=True, exist_ok=True)
    # As partes já começadas primeiro (menos para baixar), depois as outras em ordem.
    ordem = sorted(PARTES, key=lambda p: (-(RAIZ / f"{p}.zip").exists(), p))
    for parte in ordem:
        marca = RAIZ / f"feito_{parte}.txt"
        if marca.exists():
            print(f"{parte}: já feita", flush=True)
            continue
        url, tamanho = info_osf(PARTES[parte])
        zp = RAIZ / f"{parte}.zip"
        print(f"{parte}: {tamanho / 1e9:.2f} GB", flush=True)
        baixar(url, zp, tamanho)
        with zipfile.ZipFile(zp) as z:
            membros = [m for m in z.infolist() if not m.is_dir() and Path(m.filename).stem in ids]
            for m in membros:
                z.extract(m, saida)
            n_total = sum(not m.is_dir() for m in z.infolist())
        stems = sorted({Path(m.filename).stem for m in membros})
        marca.write_text("\n".join(stems))
        zp.unlink()
        print(f"{parte}: {len(membros)} de {n_total} arquivos extraídos ({len(stems)} scans); zip apagado", flush=True)
    feitos = set()
    for p in PARTES:
        m = RAIZ / f"feito_{p}.txt"
        if m.exists():
            feitos |= set(m.read_text().split())
    print(f"fim: {len(feitos)} de {len(ids)} scans de teste em {saida}", flush=True)


# ──── compactar ─────────────────────────────────────────────────────────────────────────────────────────────────────
SCANS = caminhos.TEETH3DS / "scans"


def ler_obj(p):
    v, f = [], []
    with open(p, "r") as fh:
        for linha in fh:
            if linha.startswith("v "):
                v.append(linha.split()[1:4])
            elif linha.startswith("f "):
                f.append([int(t.split("/")[0]) - 1 for t in linha.split()[1:4]])
    return np.asarray(v, np.float32), np.asarray(f, np.int32)


def npz_para_obj(npz, destino):
    z = np.load(npz)
    with open(destino, "w") as fh:
        fh.writelines(f"v {a:.6f} {b:.6f} {c:.6f}\n" for a, b, c in z["verts"])
        fh.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in z["faces"])


def compactar(obj):
    js = obj.with_suffix(".json")
    out = obj.with_suffix(".npz")
    if not js.exists() or time.time() - max(obj.stat().st_mtime, js.stat().st_mtime) < 60:
        return False                                   # ainda sendo extraído
    verts, faces = ler_obj(obj)
    d = json.loads(js.read_text())
    labels, inst = np.asarray(d["labels"], np.int16), np.asarray(d["instances"], np.int16)
    if len(labels) != len(verts) or len(inst) != len(verts) or faces.max() >= len(verts):
        print(f"  ERRO {obj.name}: {len(verts)} vértices, {len(labels)} labels, face máx {faces.max()}", flush=True)
        return False
    np.savez_compressed(out, verts=verts, faces=faces, labels=labels, instances=inst,
                        jaw=str(d.get("jaw", "")), id_patient=str(d.get("id_patient", "")))
    z = np.load(out)                                   # confere antes de apagar
    ok = (z["verts"].shape == verts.shape and z["faces"].shape == faces.shape
          and np.array_equal(z["labels"], labels) and np.array_equal(z["instances"], inst))
    if not ok:
        print(f"  ERRO {obj.name}: npz não confere; originais mantidos", flush=True)
        out.unlink()
        return False
    obj.unlink()
    js.unlink()
    return True


def passada():
    n = 0
    for obj in sorted(SCANS.glob("*/*/*.obj")):
        n += compactar(obj)
    return n


def main_compactar():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vigiar", action="store_true")
    a = ap.parse_args()
    while True:
        t0 = time.time()
        n = passada()
        total = len(list(SCANS.glob("*/*/*.npz")))
        if n:
            print(f"{time.strftime('%H:%M')} +{n} compactados em {time.time() - t0:.0f}s | {total} .npz", flush=True)
        if not a.vigiar:
            break
        log = (caminhos.TEETH3DS / "baixar.log").read_text(encoding="utf-8", errors="ignore")
        if "fim:" in log and not list(SCANS.glob("*/*/*.obj")):
            print(f"download terminou e não há OBJ pendente: {total} .npz", flush=True)
            break
        time.sleep(60)


# ──── preparar ──────────────────────────────────────────────────────────────────────────────────────────────────────
T3D = caminhos.TEETH3DS
PREP = T3D / "prep"
# Correspondência de eixos (Teeth3DS → convenção do modelo): x' = sx·x, y' = sy·z, z' = sz·y.
# As 4 rotações próprias (det = +1) com essa correspondência:
# Medido em 20 scans de cada base: coroa (+Z no Teeth3DS; −Y sup / +Y inf no modelo), lado direito do
# paciente (+X sup / −X inf no Teeth3DS; −X nos dois no modelo), incisivos (−Y → +Z). Rotações próprias:
ROTACAO = {"upper": (-1, -1, -1), "lower": (1, 1, -1)}
SENTIDO_COROA = {"upper": -1.0, "lower": 1.0}       # sinal do Y do modelo para onde a coroa aponta


def matriz(s):
    sx, sy, sz = s
    return np.array([[sx, 0, 0], [0, 0, sy], [0, sz, 0]], dtype=np.float64)


SPLITS = {"teste": "Teeth3DS_split_testing_{arch}.txt",                 # os 600 do teste oficial (segmentação)
          "landmarks_treino": "3DTeethLand_training_{arch}.txt"}      # os 240 de treino do 3DTeethLand


def ids_teste(conjunto="teste"):
    out = []
    for arch in ("upper", "lower"):
        for nome in (T3D / "splits" / SPLITS[conjunto].format(arch=arch)).read_text().split():
            out.append((nome.rsplit("_", 1)[0], arch))
    return out


def rotulo_celulas(verts, faces, labels_v, arch, bary_d):
    from scipy.spatial import cKDTree
    lf = labels_v[faces]                                          # (F, 3)
    maioria = np.where(lf[:, 1] == lf[:, 2], lf[:, 1], lf[:, 0])  # 2 de 3 iguais → esse; senão o 1º vértice
    f2l = {0: 0, **{f: i + 1 for i, f in enumerate(ORDER16[arch])}}
    fora = ~np.isin(maioria, list(f2l))
    maioria = np.where(fora, 0, maioria)
    bary_full = verts[faces].mean(1)
    _, idx = cKDTree(bary_full).query(bary_d, k=1)
    return np.array([f2l[int(f)] for f in maioria[idx]], np.int8), int(fora.sum())


def recortar(verts, faces, arch, altura):
    """Mantém as faces a até `altura` mm das pontas das cúspides (percentil 99,5 da altura oclusal), como a malha
    recorte usual da base (~18–22 mm). Sem rótulo. Devolve as faces mantidas (índices na malha cheia)."""
    if not altura:
        return np.arange(len(faces))
    h = verts[:, 1] * SENTIDO_COROA[arch]                 # cresce no sentido da coroa
    topo = np.percentile(h, 99.5)
    return np.flatnonzero((h[faces] >= topo - altura).all(1))


def preparar(sc, cid, arch, R, altura=None):
    from scipy.spatial import cKDTree
    z = np.load(T3D / "scans" / arch / cid / f"{cid}_{arch}.npz")
    verts = z["verts"].astype(np.float64) @ R.T
    faces_cheia = z["faces"].astype(np.int64)
    manter = recortar(verts, faces_cheia, arch, altura)
    usados, faces = np.unique(faces_cheia[manter], return_inverse=True)
    faces = faces.reshape(-1, 3)
    labels_v = z["labels"].astype(np.int64)
    vd, fd, bd, nd = sc.decimate_to_cells(verts[usados], faces, sc.TARGET_CELLS, label=arch)
    label, n_fora = rotulo_celulas(verts[usados], faces, labels_v[usados], arch, bd)
    perdidos = int(((labels_v > 0) & ~np.isin(np.arange(len(verts)), usados)).sum())   # vértices de dente cortados
    _, viz = cKDTree(bd).query(verts, k=1)                         # vértice original → célula decimada
    return dict(verts=vd.astype(np.float32), faces=fd.astype(np.int32), bary=bd.astype(np.float32),
                normals=nd.astype(np.float32), label=label, viz=viz.astype(np.int32),
                cortado=~np.isin(np.arange(len(verts)), usados), dente_cortado=perdidos), n_fora


def main_preparar():
    ap = argparse.ArgumentParser()
    ap.add_argument("--altura", type=float, default=22.0, help="mm mantidos a partir das cúspides (0 = sem recorte)")
    ap.add_argument("--n", type=int)
    ap.add_argument("--conjunto", choices=list(SPLITS), default="teste",
                    help="landmarks_treino = os 240 de treino do 3DTeethLand (manifesto separado)")
    a = ap.parse_args()
    # Rotação geométrica medida (ROTACAO) e recorte a --altura mm das cúspides: no teste com 6 scans por arcada e os
    # pesos pré-treinados, 22 mm deu o maior acerto nas células de dente (sup 0,73 → 0,84; inf 0,43 → 0,78), sem
    # cortar nenhum vértice de dente. A calibração por acerto (--calibrar, sem recorte) escolhia errado no inferior.
    alvo = ids_teste(a.conjunto)[: a.n] if a.n else ids_teste(a.conjunto)
    linhas, t0, feitos = [], time.time(), 0
    for i, (cid, arch) in enumerate(alvo, 1):
        dest = PREP / f"{cid}_{arch}.npz"
        if not dest.exists():
            try:
                d, n_fora = preparar(sc, cid, arch, matriz(ROTACAO[arch]), a.altura)
            except Exception as ex:                                 # noqa: BLE001
                print(f"[{i}/{len(alvo)}] {cid}_{arch}: ERRO {ex}", flush=True)
                continue
            tmp = dest.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **d)
            tmp.replace(dest)
            feitos += 1
            if feitos % 25 == 0:
                print(f"[{i}/{len(alvo)}] {(time.time() - t0) / feitos:.1f} s/scan", flush=True)
        else:
            n_fora = -1
        lab = np.load(dest)["label"]
        zd = np.load(dest)
        linhas.append(dict(id=cid, arch=arch, arquivo=dest.name, n_dentes=int(len(np.unique(lab[lab > 0]))),
                           faces_fdi_fora=n_fora, dente_cortado=int(zd["dente_cortado"]),
                           frac_cortada=float(zd["cortado"].mean())))
    nome_man = "manifesto.csv" if a.conjunto == "teste" else f"manifesto_{a.conjunto}.csv"
    with open(PREP / nome_man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    print(f"{len(linhas)} scans em {PREP}")


def main():
    subs = {"baixar": main_baixar, "compactar": main_compactar, "preparar": main_preparar}
    if len(sys.argv) < 2 or sys.argv[1] not in subs:
        sys.exit(__doc__)
    sub = sys.argv.pop(1)
    sys.argv[0] = f"teeth3ds.py {sub}"
    subs[sub]()


if __name__ == "__main__":
    main()
