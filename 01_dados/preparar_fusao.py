"""Prepara as arcadas para a fusão 2D → 3D.

Para cada .vtp do planejador (<TCC_BRUTOS>/<lote>_{upper,lower}/, com o array de célula FDI; ver DADOS.md):
  1. decima para ~8.000 células com 00_comum/malha.py (malha.decimate_to_cells), para que a
     rede veja sempre a mesma densidade;
  2. leva o FDI para cada célula decimada pelo vizinho mais próximo (KNN k=1) e o
     converte para as 17 classes "own_class" (0 = gengiva; 1..16 = posterior-direito → posterior-esquerdo);
  3. calcula a coordenada ao longo do arco ("arco", 0..1, 0 = lado direito do paciente), que é a
     codificação de posição 3D da atenção cruzada.

Coordenada do arco: COMPRIMENTO DE ARCO, normalizado por malha. Plano oclusal por PCA dos baricentros
(o eixo u é a componente mais alinhada com o x do referencial do planejamento), curva v = polinômio de grau 4 em u ajustada às
células, e s = comprimento da curva desde a ponta direita. Não usa rótulo, então vale no teste.
Por quê (estudo de viabilidade do registro): prevendo o x do dente na panorâmica, o comprimento de arco
erra 0,23 largura de dente (mediana) contra 0,74 do x cru com f afim; com f cúbica, 0,16 com todos os
dentes no ajuste (p90 0,50). O início (0 = lado direito do paciente) é decidido pelo lado que o rascunho 3D
prevê; o GT do planejador só confere (coluna orient_gt_ok do manifesto).

Split: teste = ordem <= 150 do teste_multimodal_ordem.csv (anotados no CVAT,
fora de QUALQUER treino); o resto é treino, com 15% dos case_id para validação (semente 42).
Um caso com vários ciclos usa o ciclo/versão mais recente de cada arcada.

Precisa do vedo e do scipy (00_comum/malha.py). Uso: python 01_dados/preparar_fusao.py [--lote tcc|complementar|ampliacao] [--n 5]

Versão pública: o campo rascunho_label/rascunho_conf é gravado ZERADO. No pipeline final (G5) ele é substituído pelo
rascunho do nosso próprio modelo só 3D (03_registro/montar_limpo.py), e o arco é reorientado pelo lado que esse
rascunho prevê (03_registro/registro_fusao.py --rotulos), então a orientação inicial abaixo não muda o resultado. O GT só
confere. Os 34 casos antigos (podem estar no treino do MeshSegNet pré-treinado) ficam marcados no manifesto.
Resumível: pula a arcada cujo .npz já tem rascunho_label (escrita atômica: parar no meio não corrompe nada).
Em paralelo: --partes 2 --parte 0 e --parte 1 em dois terminais (cada um grava manifesto_parteK.csv); no fim,
uma rodada sem --partes (só pula, rápida) monta o manifesto.csv completo.

Saída: <saida>/malhas/<case_id>_<arch>.npz (verts, faces, bary, normals, label, arco, rascunho_label, rascunho_conf)
       e <saida>/manifesto.csv (case_id, arch, split, arquivo, n_celulas, n_dentes, rascunho_n_dentes,
       orient_gt_ok, antigo, pesos_iniciais, fonte).
"""
import argparse
import csv
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
import caminhos  # noqa: E402
import malha  # noqa: E402

SCRIPTS = caminhos.LISTAS                                     # listas de casos e status
IMPL = caminhos.BRUTOS                                        # <lote>_{upper,lower}/*.vtp
SAIDA = caminhos.FUSAO_DADOS
ORDEM = SCRIPTS / "teste_multimodal_ordem.csv"
FEATURES_2D = caminhos.FEATURES_2D                            # indice.csv: novo = False → os 34 antigos

# Convenção de 17 classes: posterior-direito -> posterior-esquerdo, 0 = gengiva.
ORDER16 = {
    "upper": [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28],
    "lower": [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38],
}
LADO_DIREITO = {1, 4}           # quadrantes do lado direito do paciente
NOME = re.compile(r"^(\d+)_(\d+)_v(\d+)_(upper|lower)$")    # {ciclo}_{case_id}_v{versao}_{arch}
VAL_FRAC, SEMENTE = 0.15, 42


def ler_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def listar_vtps(impl, prefixo="tcc"):
    """Um .vtp por (case_id, arch): o ciclo e a versão mais recentes. (Sem pandas.)"""
    ultimo = {}
    for arch in ("upper", "lower"):
        for p in (impl / f"{prefixo}_{arch}").glob("*.vtp"):
            m = NOME.match(p.stem)
            if m and not p.name.endswith(".tmp.vtp"):
                chave, ordem = (int(m[2]), arch), (int(m[1]), int(m[3]))
                if chave not in ultimo or ordem > ultimo[chave][0]:
                    ultimo[chave] = (ordem, p)
    return [SimpleNamespace(case_id=c, arch=a, path=p) for (c, a), (_, p) in sorted(ultimo.items())]


def splits(case_ids):
    teste = {int(r["case_id"]) for r in ler_csv(ORDEM) if float(r["ordem"]) <= 150}
    resto = sorted(set(case_ids) - teste)
    rng = np.random.default_rng(SEMENTE)
    val = set(rng.choice(resto, size=int(round(VAL_FRAC * len(resto))), replace=False).tolist())
    return {c: "teste" if c in teste else "val" if c in val else "treino" for c in case_ids}


def comprimento_de_arco(pts, grau=4):
    """s em [0, 1] por célula: comprimento da curva do arco no plano oclusal, a partir do lado de u menor."""
    c = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    eixos = vt[:2]
    if abs(eixos[1, 0]) > abs(eixos[0, 0]):          # u = a componente do plano mais alinhada com o x do referencial do planejamento
        eixos = eixos[::-1]
    if eixos[0, 0] < 0:
        eixos = np.array([-eixos[0], eixos[1]])      # u cresce no mesmo sentido do x do referencial do planejamento
    from scipy.spatial import cKDTree
    uv = c @ eixos.T
    coef = np.polyfit(uv[:, 0], uv[:, 1], grau)
    g = np.linspace(uv[:, 0].min(), uv[:, 0].max(), 3000)
    curva = np.c_[g, np.polyval(coef, g)]
    s_grid = np.r_[0, np.cumsum(np.hypot(*np.diff(curva, axis=0).T))]
    # ponto MAIS PRÓXIMO da curva, não só pelo u: nos molares o arco corre de frente para trás, e as faces
    # vestibular e lingual do mesmo dente teriam u diferentes
    _, idx = cKDTree(curva).query(uv)
    return (s_grid[idx] / s_grid[-1]).astype(np.float32)


def preparar(path, arch):
    import vedo
    from scipy.spatial import cKDTree

    m = vedo.load(str(path))
    try:
        faces = np.asarray(m.cells, dtype=np.int64)
    except ValueError:                           # polígonos com mais de 3 lados (3 em 10 no teste com --n 5)
        faces = None
    if faces is None or faces.ndim != 2 or faces.shape[1] != 3:
        m = m.triangulate()                      # o vtkTriangleFilter copia o FDI do polígono para cada triângulo
        faces = np.asarray(m.cells, dtype=np.int64)
    verts = np.asarray(m.points, dtype=np.float64)
    fdi = np.asarray(m.celldata["FDI"]).ravel().astype(np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(fdi) != len(faces):
        raise ValueError("malha não triangular depois de triangular")
    # Decíduo (51..85) → a posição permanente que ele ocupa (53→13, 75→35), como o planejamento: no JSON o
    # dente é o objeto tooth_13 (fdi = 13) com deciduous_fdi = 53; só o FDI das faces do .vtp traz o 53.
    deciduo = (fdi >= 51) & (fdi <= 85)
    fdi = np.where(deciduo, (fdi // 10 - 4) * 10 + fdi % 10, fdi)
    f2l = {0: 0, **{f: i + 1 for i, f in enumerate(ORDER16[arch])}}
    desconhecidos = set(np.unique(fdi).tolist()) - set(f2l)
    if desconhecidos:
        raise ValueError(f"FDI fora da convenção: {sorted(desconhecidos)}")

    bary_full = verts[faces].mean(axis=1)
    verts_d, faces_d, bary_d, normals_d = malha.decimate_to_cells(verts, faces, malha.TARGET_CELLS, label=arch)
    _, idx = cKDTree(bary_full).query(bary_d, k=1)
    fdi_d = fdi[idx]
    label = np.array([f2l[int(f)] for f in fdi_d], dtype=np.int8)
    deciduo_d = deciduo[idx]

    # Rascunho 3D: ainda não existe nesta etapa (vem do G5-S, 03_registro/montar_limpo.py). Fica zerado.
    rascunho_label = np.zeros(len(faces_d), np.int8)
    rascunho_conf = np.zeros(len(faces_d), np.float16)

    # Orientação inicial do arco: a do eixo u do plano oclusal (comprimento_de_arco). O registro_fusao.py --rotulos
    # a refaz pelo lado que o rascunho prevê. O GT do planejador só confere (orient_gt_ok).
    arco = comprimento_de_arco(bary_d)
    dentes = fdi_d > 0
    direita = dentes & np.isin(fdi_d // 10, list(LADO_DIREITO))
    esquerda = dentes & ~direita
    orient_gt_ok = bool(not direita.any() or not esquerda.any() or arco[direita].mean() < arco[esquerda].mean())
    return dict(verts=verts_d.astype(np.float32), faces=faces_d.astype(np.int32), bary=bary_d.astype(np.float32),
                normals=normals_d.astype(np.float32), label=label, arco=arco,
                rascunho_label=rascunho_label, rascunho_conf=rascunho_conf, orient_gt_ok=orient_gt_ok,
                deciduo=deciduo_d)


def linha_manifesto(dados, r, split, antigos, pesos):
    label, zl = dados["label"], dados["rascunho_label"]
    return dict(case_id=r.case_id, arch=r.arch, split=split[r.case_id], arquivo=f"{r.case_id}_{r.arch}.npz",
                n_celulas=len(label), n_dentes=int(len(np.unique(label[label > 0]))),
                rascunho_n_dentes=int(len(np.unique(zl[zl > 0]))), orient_gt_ok=bool(dados["orient_gt_ok"]),
                n_deciduos=int(len(np.unique(label[(label > 0) & dados["deciduo"]]))) if "deciduo" in dados else 0,
                antigo=r.case_id in antigos, pesos_iniciais=pesos, fonte=r.path.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="só os N primeiros casos (teste rápido)")
    ap.add_argument("--impl", type=Path, default=IMPL)
    ap.add_argument("--saida", type=Path, default=SAIDA)
    ap.add_argument("--partes", type=int, default=1, help="dividir os casos em N partes (processos em paralelo)")
    ap.add_argument("--parte", type=int, default=0, help="qual parte (0..N-1) este processo faz")
    ap.add_argument("--lote", choices=["tcc", "complementar", "ampliacao"], default=None,
                    help="complementar = 121 casos usados no treino do MeshSegNet pré-treinado (TODOS no treino, "
                         "manifesto_complementar.csv, antigo = True); ampliacao = a ampliação (TODOS no treino, depois "
                         "redividida por redividir_novos.py; manifesto_ampliacao.csv)")
    a = ap.parse_args()
    lote = a.lote or "tcc"

    import time
    pesos = {arch: "nenhum" for arch in ("upper", "lower")}
    antigos = {int(r["case_id"]) for r in ler_csv(FEATURES_2D / "indice.csv") if r["novo"] == "False"}
    vtps = listar_vtps(a.impl, lote)
    casos = sorted({r.case_id for r in vtps})[: a.n] if a.n else sorted({r.case_id for r in vtps})
    vtps = [r for r in vtps if r.case_id in set(casos)]
    if lote != "tcc":
        split = {c: "treino" for c in casos}
        antigos = set(casos) if lote == "complementar" else set()
        status = SCRIPTS / f"{lote}_status.csv"
        # trava contra vazamento: nenhum caso do lote pode estar no val/teste da fusão
        fusao = {int(r["case_id"]) for r in ler_csv(a.saida / "manifesto.csv") if r["split"] != "treino"}
        vaza = fusao & set(casos)
        if vaza:
            raise SystemExit(f"{len(vaza)} casos do lote {lote} estão no val/teste da fusão: {sorted(vaza)[:5]}")
    else:
        split = splits(casos)                    # sempre com TODOS os casos: o split não depende da parte
        status = ORDEM
    # Casos com status pulado_* (panorâmica ruim, normais invertidas, sem panorâmica...) ficam de fora, como no
    # extrair_features_2d.py. Filtrados DEPOIS do split, para ele não mudar.
    pulados = {int(r["case_id"]) for r in ler_csv(status) if (r["status"] or "").startswith("pulado")}
    vtps = [r for r in vtps if r.case_id not in pulados]
    if a.partes > 1:
        meus = set(casos[a.parte::a.partes])
        vtps = [r for r in vtps if r.case_id in meus]
    base = "manifesto" if lote == "tcc" else f"manifesto_{lote}"
    nome_manifesto = f"{base}.csv" if a.partes == 1 else f"{base}_parte{a.parte}.csv"
    (a.saida / "malhas").mkdir(parents=True, exist_ok=True)

    def gravar(linhas):
        with open(a.saida / nome_manifesto, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0]))
            w.writeheader()
            w.writerows(linhas)

    linhas, t0, feitas = [], time.time(), 0
    for i, r in enumerate(vtps, 1):
        destino = a.saida / "malhas" / f"{r.case_id}_{r.arch}.npz"
        if destino.exists():                     # retomada: arcada pronta (com o rascunho) é pulada
            try:
                with np.load(destino) as z:
                    if "rascunho_label" in z.files:
                        linhas.append(linha_manifesto({k: z[k] for k in z.files}, r, split, antigos, pesos[r.arch]))
                        continue
            except Exception:
                pass                             # .npz truncado por uma parada no meio: refaz
        try:
            dados = preparar(r.path, r.arch)
        except Exception as ex:                  # um caso ruim não para a rodada
            print(f"[{i}/{len(vtps)}] {r.path.name}: ERRO {ex}", flush=True)
            continue
        tmp = destino.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **dados)
        tmp.replace(destino)                     # escrita atômica: uma parada nunca deixa .npz pela metade
        linhas.append(linha_manifesto(dados, r, split, antigos, pesos[r.arch]))
        feitas += 1
        l = linhas[-1]
        print(f"[{i}/{len(vtps)}] {r.path.name}: {l['n_celulas']} células, {l['n_dentes']} dentes "
              f"(rascunho {l['rascunho_n_dentes']}), {l['split']}{', ANTIGO' if l['antigo'] else ''}"
              f"{'' if l['orient_gt_ok'] else ', ORIENTAÇÃO ≠ GT'} | {(time.time() - t0) / feitas:.1f} s/arcada",
              flush=True)
        if feitas % 25 == 0:
            gravar(linhas)

    gravar(linhas)
    from collections import Counter
    print(f"\n{len(linhas)} arcadas de {len({l['case_id'] for l in linhas})} casos → {a.saida}")
    print("por split e arcada:", dict(sorted(Counter((l["split"], l["arch"]) for l in linhas).items())))
    ok = [l["orient_gt_ok"] for l in linhas]
    print(f"orientação inicial do arco igual à do GT: {sum(ok) / len(ok):.1%} das arcadas ({len(ok) - sum(ok)} diferentes)")
    print(f"antigos (podem estar no treino do MeshSegNet pré-treinado): {len({l['case_id'] for l in linhas if l['antigo']})} casos")


if __name__ == "__main__":
    main()
