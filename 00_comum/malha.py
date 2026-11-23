"""Operações sobre a malha usadas em todo o pipeline (decimação, features, grafo, graph-cut, pós-processamento).

As fórmulas seguem o MeshSegNet público (Lian et al., IEEE TMI 2020; https://github.com/Tai-Hsien/MeshSegNet,
licença MIT, ver LICENSE_MeshSegNet):
  - features de 15 canais por célula (9 coordenadas dos vértices + baricentro + normal), normalizadas como em
    `step5_predict.py`;
  - adjacências A_S (raio 0,1) e A_L (raio 0,2) no baricentro normalizado, normalizadas por linha;
  - refino por graph-cut (alfa-expansão) com o termo de borda baseado no ângulo diedro e na distância entre
    baricentros, como em `step6_predict_with_post_processing_pygco.py`;
  - volta à malha cheia por voto dos k vizinhos mais próximos (o `KNeighborsClassifier(n_neighbors=3)` do step6).
Acrescentados aqui: remoção de ilhas pequenas e uma suavização por difusão dos rótulos na malha cheia.

Constantes do TCC: 17 classes (gengiva + 16 dentes, terceiros molares inclusos) e malha decimada para 8.000
células; o λ do graph-cut é escolhido no conjunto de validação (`04_segmentacao_fusao/escolher_lambda.py`).
"""
from __future__ import annotations

import numpy as np

NUM_CLASSES = 17
NUM_CHANNELS = 15
TARGET_CELLS = 8000


# ---------------------------------------------------------------------------------------------------------------
# Decimação
# ---------------------------------------------------------------------------------------------------------------
def _sem_degeneradas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Tira triângulos com vértice repetido ou área zero (o vtkQuadricDecimation para ao encontrar um deles)."""
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    area2 = np.linalg.norm(np.cross(verts[b] - verts[a], verts[c] - verts[a]), axis=1)
    ok = (a != b) & (b != c) & (a != c) & (area2 > 0.0)
    return faces[ok]


def decimate_to_cells(verts, faces, target_cells=TARGET_CELLS, *, label=""):
    """Decima a malha para ~`target_cells` triângulos (vedo / vtkQuadricDecimation).

    Devolve (verts, faces, baricentros, normais por célula) da malha decimada. A decimação depende da ordem dos
    vértices: passe a malha na ordem original do arquivo.
    """
    import vedo

    faces = _sem_degeneradas(np.asarray(verts), np.asarray(faces))
    if len(faces) == 0:
        raise ValueError(f"{label}: malha sem triângulos válidos")
    m = vedo.Mesh([verts, faces])
    if m.ncells > target_cells:
        m.decimate(fraction=target_cells / m.ncells)
    if m.ncells > 1.5 * target_cells:
        raise RuntimeError(f"{label}: a decimação não reduziu a malha ({m.ncells} células para {target_cells})")
    v = np.asarray(m.points, dtype=np.float64)
    f = np.asarray(m.cells, dtype=np.int64)
    bary = np.asarray(m.cell_centers().points, dtype=np.float64)
    m.compute_normals()
    normais = np.asarray(m.celldata["Normals"], dtype=np.float32)
    return v, f, bary, normais


# ---------------------------------------------------------------------------------------------------------------
# Entrada da rede
# ---------------------------------------------------------------------------------------------------------------
def extract_features(verts, faces, barycenters, normals) -> np.ndarray:
    """Matriz (N, 15) de features por célula, normalizada como no MeshSegNet público."""
    pts = np.asarray(verts, np.float64)
    centro = pts.mean(0)
    pts = pts - centro
    cel = pts[faces].reshape(len(faces), 9).astype(np.float32)
    bar = (np.asarray(barycenters, np.float64) - centro).astype(np.float32)
    nor = np.array(normals, dtype=np.float32, copy=True)

    media, desvio = pts.mean(0), pts.std(0)
    minimo, maximo = pts.min(0), pts.max(0)
    n_media, n_desvio = nor.mean(0), nor.std(0)
    desvio = np.where(desvio == 0, 1.0, desvio)
    n_desvio = np.where(n_desvio == 0, 1.0, n_desvio)
    faixa = np.where(maximo - minimo == 0, 1.0, maximo - minimo)
    for i in range(3):
        for k in (0, 3, 6):                                   # os três vértices da célula
            cel[:, i + k] = (cel[:, i + k] - media[i]) / desvio[i]
        bar[:, i] = (bar[:, i] - minimo[i]) / faixa[i]
        nor[:, i] = (nor[:, i] - n_media[i]) / n_desvio[i]
    return np.column_stack((cel, bar, nor)).astype(np.float32)


def adjacency_sparse(features: np.ndarray):
    """A_S (d < 0,1) e A_L (d ≤ 0,2) no baricentro normalizado, com laço, normalizadas por linha (torch CSR)."""
    import torch
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree

    n = len(features)
    xyz = np.ascontiguousarray(features[:, 9:12], dtype=np.float64)
    pares = cKDTree(xyz).query_pairs(0.2, output_type="ndarray")
    i, j = (pares[:, 0], pares[:, 1]) if len(pares) else (np.zeros(0, np.int64),) * 2
    perto = np.linalg.norm(xyz[i] - xyz[j], axis=1) < 0.1
    laco = np.arange(n)

    def montar(i, j):
        m = csr_matrix((np.ones(n + 2 * len(i), np.float32), (np.r_[laco, i, j], np.r_[laco, j, i])), shape=(n, n))
        m.sum_duplicates()
        m.data[:] = 1.0
        m.data /= np.repeat(np.diff(m.indptr), np.diff(m.indptr)).astype(np.float32)
        return torch.sparse_csr_tensor(torch.from_numpy(m.indptr.astype(np.int64)),
                                       torch.from_numpy(m.indices.astype(np.int64)),
                                       torch.from_numpy(m.data), size=(n, n))

    return montar(i[perto], j[perto]), montar(i, j)


# ---------------------------------------------------------------------------------------------------------------
# Grafo de células e graph-cut
# ---------------------------------------------------------------------------------------------------------------
def cell_graph(faces: np.ndarray):
    """Pares (src, dst) de células que compartilham uma aresta (arestas com exatamente duas células)."""
    faces = np.asarray(faces, np.int64)
    nf = len(faces)
    arestas = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    arestas.sort(axis=1)
    cel = np.tile(np.arange(nf), 3)
    # ordem estável: por aresta e, dentro dela, pela ordem em que a célula aparece (face, depois lado da face)
    lado = np.repeat(np.arange(3), nf)
    ordem = np.lexsort((lado, cel, arestas[:, 1], arestas[:, 0]))
    a, c = arestas[ordem], cel[ordem]
    novo = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    grupo = np.cumsum(novo) - 1
    tam = np.bincount(grupo)
    inicio = np.flatnonzero(novo)
    duas = tam == 2
    src, dst = c[inicio[duas]], c[inicio[duas] + 1]
    # mesma ordem de saída do percurso face a face: pela primeira vez em que a aresta aparece
    primeira = np.minimum.reduceat(c * 3 + lado[ordem], inicio)[duas]
    o = np.argsort(primeira, kind="stable")
    return src[o].astype(np.int64), dst[o].astype(np.int64)


def load_gco():
    """Função de corte (alfa-expansão, energias float64) do pacote gco-wrapper; None se não estiver instalado."""
    for nome, tipo in (("float", float), ("int", int), ("bool", bool)):   # o gco usa aliases antigos do numpy
        if not hasattr(np, nome):
            setattr(np, nome, tipo)
    try:
        from gco import cut_general_graph
    except Exception:
        try:
            from gco.pygco import cut_general_graph        # algumas versões do gco-wrapper expõem só aqui
        except Exception:
            return None

    def cortar(arestas, pesos, unarios, par):
        return cut_general_graph(np.ascontiguousarray(arestas, np.int32), np.ascontiguousarray(pesos, np.float64),
                                 np.ascontiguousarray(unarios, np.float64), np.ascontiguousarray(par, np.float64),
                                 algorithm="expansion")

    return cortar


def refine_graphcut(prob, faces, barycenters, normals, lambda_c, cut_from_graph):
    """Refino da borda por graph-cut (MeshSegNet, step6): unário −log10 P; Potts com peso por aresta
    −log10(θ/π)·φ, multiplicado por β = 1 + |n_i·n_j| nas arestas convexas (θ ≤ 90°), vezes λ."""
    prob = np.clip(prob, 1e-6, 1.0)
    unarios = -np.log10(prob).astype(np.float64)
    par = 1.0 - np.eye(prob.shape[1])
    src, dst = cell_graph(faces)
    ni, nj = normals[src], normals[dst]
    dot = np.sum(ni * nj, axis=1)
    cos = np.clip(dot / (np.linalg.norm(ni, axis=1) * np.linalg.norm(nj, axis=1) + 1e-12), -1.0, 0.9999)
    theta = np.arccos(cos)
    phi = np.linalg.norm(barycenters[src] - barycenters[dst], axis=1)
    w = -np.log10(theta / np.pi) * phi
    w = np.where(theta <= np.pi / 2, (1.0 + np.abs(dot)) * w, w)
    w = np.maximum(w * lambda_c, 0.0)
    rot = cut_from_graph(np.column_stack([src, dst]), w, unarios, par)
    return rot.astype(np.int64), (src, dst)


# ---------------------------------------------------------------------------------------------------------------
# Pós-processamento
# ---------------------------------------------------------------------------------------------------------------
def remove_islands(labels, src, dst, n, min_frac=0.005, num_classes=NUM_CLASSES):
    """Componentes conexas de mesmo rótulo com menos de max(3, min_frac·n) células recebem o rótulo mais comum
    entre os vizinhos de fora (até 5 rodadas)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    def adj(s, d):
        return csr_matrix((np.ones(2 * len(s)), (np.r_[s, d], np.r_[d, s])), shape=(n, n))

    viz = adj(src, dst)
    minimo = max(3, int(min_frac * n))
    for _ in range(5):
        mudou = False
        igual = labels[src] == labels[dst]
        _, comp = connected_components(adj(src[igual], dst[igual]), directed=False)
        for c in np.flatnonzero(np.bincount(comp) < minimo):
            membros = np.flatnonzero(comp == c)
            fora = viz[membros].indices
            fora = fora[~np.isin(fora, membros)]
            if len(fora) == 0:
                continue
            novo = np.bincount(labels[fora], minlength=int(num_classes)).argmax()
            if novo != labels[membros[0]]:
                labels[membros] = novo
                mudou = True
        if not mudou:
            break
    return labels


def upsample(labels_d, bary_d, bary_full, k=3, num_classes=NUM_CLASSES):
    """Rótulo de cada face da malha cheia = voto da maioria das k células decimadas mais próximas
    (empate → menor rótulo)."""
    from scipy.spatial import cKDTree

    k = min(k, len(bary_d))
    _, idx = cKDTree(bary_d).query(bary_full, k=k)
    if k == 1:
        return labels_d[idx]
    viz = labels_d[idx]
    passo = max(int(num_classes), int(labels_d.max()) + 1)
    n = len(viz)
    cont = np.bincount((viz + passo * np.arange(n)[:, None]).ravel(), minlength=passo * n).reshape(n, passo)
    return cont.argmax(1).astype(np.int64)


def smooth_labels(labels_full, src, dst, iters=15, lam=0.5, num_classes=NUM_CLASSES):
    """Suaviza a borda dos rótulos na malha cheia: difusão do one-hot para a média dos vizinhos
    (x ← x + λ(Āx − x), `iters` vezes) e argmax."""
    from scipy.sparse import csr_matrix, diags

    n = len(labels_full)
    if iters <= 0 or len(src) == 0:
        return labels_full
    a = csr_matrix((np.ones(2 * len(src), np.float32), (np.r_[src, dst], np.r_[dst, src])), shape=(n, n))
    grau = np.asarray(a.sum(1)).ravel()
    grau[grau == 0] = 1.0
    media = diags(1.0 / grau) @ a
    nc = max(int(num_classes), int(labels_full.max()) + 1)
    x = np.zeros((n, nc), np.float32)
    x[np.arange(n), np.clip(labels_full, 0, nc - 1)] = 1.0
    for _ in range(int(iters)):
        x += lam * (media @ x - x)
    return x.argmax(1).astype(np.int64)
