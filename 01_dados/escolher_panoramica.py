"""Escolhe a panorâmica de verdade entre as candidatas de um caso.

Entrada: ds_complementar/candidatas/<case_id>__<k>.png e candidatas.csv (todas as panorâmicas candidatas de
cada caso, já pré-processadas com 01_dados/preprocessar_panoramica.py).
k = ordem de preferência (Panoramica antes de Indefinidos, mais recente primeiro); a ATUAL (a que está em processados/)
é a de menor k que passa no filtro heurístico. A pasta Panoramica às vezes tem também periapical, tele ou foto.

Critério automático: o Mask R-CNN v2 (o mesmo pipeline do extrair_features_2d.py) em cada candidata. Uma panorâmica
de verdade dá ~25–30 dentes nas DUAS arcadas; periapical/foto/tele dá poucos ou uma arcada só.
  pontuação = nº de dentes, com as duas arcadas obrigatórias (min(sup, inf) ≥ 4); empate → a menor k.
Também monta uma folha por caso (candidatas lado a lado, com k, dentes sup/inf e o filtro) em
ds_complementar/folhas/<case_id>.png, para conferência visual.

Uso: python 01_dados/escolher_panoramica.py                # mede, escolhe e monta as folhas (não troca nada)
     python 01_dados/escolher_panoramica.py --aplicar      # copia a escolhida para processados/ quando ≠ a atual
                                                                 # (a antiga vai para processados_antes/)
Saída: ds_complementar/escolha.csv
"""
import argparse
import shutil
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pipeline_2d as p2d  # noqa: E402
from extrair_features_2d import PESOS  # noqa: E402
from montar_cvat_teste_multimodal import BASE  # noqa: E402

RAIZ = BASE / "ds_complementar"                  # BASE = caminhos.PANORAMICAS
sys.stdout.reconfigure(encoding="utf-8")


def ler(p):
    return cv2.imdecode(np.frombuffer(Path(p).read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)


def gravar(p, img):
    Path(p).write_bytes(cv2.imencode(".png", img)[1].tobytes())


def folha(cid, linhas):
    tiles = []
    for r in linhas:
        f = RAIZ / "candidatas" / f"{cid}__{r['k']}.png"
        if f.exists():
            im = cv2.cvtColor(cv2.resize(ler(f), (384, 384)), cv2.COLOR_GRAY2BGR)
        else:                                   # candidata que não decodificou: quadro preto com o motivo
            im = np.zeros((384, 384, 3), np.uint8)
        cor = (0, 200, 0) if r["escolhida"] else (0, 0, 230)
        cv2.rectangle(im, (0, 0), (383, 383), cor, 6 if r["escolhida"] else 2)
        txt = f"k={r['k']} {r['n_sup']}+{r['n_inf']} dentes"
        cv2.putText(im, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(im, str(r["filtro"])[:30], (8, 372), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(im)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    linhas_img = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    (RAIZ / "folhas").mkdir(exist_ok=True)
    gravar(RAIZ / "folhas" / f"{cid}.png", np.vstack(linhas_img))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--raiz", default=None, help="pasta das panorâmicas candidatas (padrão: ds_complementar)")
    ap.add_argument("--decisoes", default=None,
                    help="com --aplicar, CSV (case_id, decisao) da revisão: 'trocar' aplica a escolhida, "
                         "'manter' não mexe, 'sem_panoramica' tira o caso de processados/ (vai para processados_antes/)")
    a = ap.parse_args()
    global RAIZ
    if a.raiz:
        RAIZ = BASE / a.raiz
    C = pd.read_csv(RAIZ / "candidatas.csv", dtype={"case_id": str})
    if not a.aplicar:
        pred = cap = None                       # o Mask R-CNN só carrega se faltar alguma candidata no cache
        # Retomável: cada candidata medida vai para escolha_cache.csv na hora; na próxima rodada, pula.
        cache_path = RAIZ / "escolha_cache.csv"
        cache = {}
        if cache_path.exists():
            for c in pd.read_csv(cache_path, dtype={"case_id": str}).itertuples():
                cache[(c.case_id, int(c.k))] = dict(n_sup=int(c.n_sup), n_inf=int(c.n_inf), score=float(c.score))
        print(f"{len(cache)} candidatas já medidas | {len(C)} no total", flush=True)
        res = []
        for i, r in enumerate(C.itertuples(), 1):
            chave = (r.case_id, int(r.k))
            if chave in cache:
                res.append(cache[chave])
                continue
            f = RAIZ / "candidatas" / f"{r.case_id}__{r.k}.png"
            if not f.exists():
                m = dict(n_sup=0, n_inf=0, score=0.0)
            else:
                if pred is None:
                    pred = p2d.build_predictor(PESOS, "cpu")
                    cap = p2d.instalar_captura(pred)
                d, _ = p2d.extrair(pred, cap, p2d.imread(f))
                arc = np.asarray(d["arcada"])
                m = dict(n_sup=int((arc == 0).sum()), n_inf=int((arc == 1).sum()),
                         score=float(np.mean(d["score"])) if len(arc) else 0.0)
            res.append(m)
            novo = not cache_path.exists()
            pd.DataFrame([dict(case_id=r.case_id, k=int(r.k), **m)]).to_csv(cache_path, mode="a", header=novo,
                                                                           index=False)
            if i % 50 == 0:
                print(f"[{i}/{len(C)}]", flush=True)
        C = pd.concat([C.reset_index(drop=True), pd.DataFrame(res)], axis=1)
        C["n"] = C.n_sup + C.n_inf
        C["pontos"] = np.where(np.minimum(C.n_sup, C.n_inf) >= 4, C.n, -1)
        C["escolhida"] = False
        C["atual"] = False
        for cid, g in C.groupby("case_id"):
            ok = g[g.filtro == "ok"]
            if len(ok):
                C.loc[ok.k.idxmin(), "atual"] = True
        for cid, g in C.groupby("case_id"):
            melhor = g.sort_values(["pontos", "k"], ascending=[False, True]).index[0]
            C.loc[melhor, "escolhida"] = True
            folha(cid, C.loc[g.index].to_dict("records"))
        C.to_csv(RAIZ / "escolha.csv", index=False)
        mud = C[C.escolhida & ~C.atual]
        print(f"{C.case_id.nunique()} casos | a escolhida muda em {len(mud)}: {mud.case_id.tolist()}")
        sem = C.groupby("case_id").pontos.max()
        print(f"casos em que nenhuma candidata tem as duas arcadas: {sem[sem < 0].index.tolist()}")
        print(f"folhas em {RAIZ / 'folhas'}")
        return
    C = pd.read_csv(RAIZ / "escolha.csv", dtype={"case_id": str})
    (RAIZ / "processados_antes").mkdir(exist_ok=True)
    alvo = C[C.escolhida & ~C.atual]
    if a.decisoes:
        D = pd.read_csv(RAIZ / a.decisoes, dtype={"case_id": str})
        for cid in D[D.decisao == "sem_panoramica"].case_id:
            atual = RAIZ / "processados" / f"{cid}.png"
            if atual.exists():
                shutil.move(str(atual), RAIZ / "processados_antes" / atual.name)
                print(f"{cid}: sem panorâmica → tirado de processados/")
        alvo = alvo[alvo.case_id.isin(D[D.decisao == "trocar"].case_id)]
        print(f"{len(alvo)} trocas aprovadas na revisão ({a.decisoes})")
    for r in alvo.itertuples():
        atual = RAIZ / "processados" / f"{r.case_id}.png"
        if atual.exists():
            shutil.move(str(atual), RAIZ / "processados_antes" / atual.name)
        shutil.copy2(RAIZ / "candidatas" / f"{r.case_id}__{r.k}.png", atual)
        print(f"{r.case_id}: processados ← candidata k={r.k}")


if __name__ == "__main__":
    main()
