"""Treino e avaliação da fusão 2D → 3D. Para a GPU alugada (roda em CPU só para teste).

Receita de treino do MeshSegNet de referência mantida onde não há motivo para mudar: Adam com AMSGrad, sem
weight decay, perda Dice generalizada, patch balanceado (25% gengiva), best.pt pelo maior val_DSC,
early stopping de 30 épocas, batch >= 2 (o BatchNorm do vetor global quebra com 1).

O que muda na fusão:
  - parte de pesos pré-treinados (--pesos-iniciais) nas camadas herdadas;
  - fase 1 (--congelar épocas): só as camadas novas treinam, com o MeshSegNet congelado (BN em eval),
    para o 2D aprender a "falar" com features 3D estáveis; fase 2: tudo junto, LR menor no MeshSegNet;
  - --fusao nenhuma é a comparação obrigatória: MESMOS casos, MESMA receita, sem 2D.

Validação e teste na malha decimada INTEIRA (batch 1, sem patch), nas 17 classes, com o 2D e com o
2D desligado (sem_2d) no MESMO modelo: a diferença mede quanto a panorâmica contribui.
O graph-cut e a remoção de ilhas ficam no escolher_lambda.py.

Uso (receitas da G5; o 04_segmentacao_fusao/rodar_g5.sh roda todas):
  BASE="--arch lower --lr 1e-2 --bn-livre --batch 8 --rot-max 10 --com-complementar --com-ampliacao
        --paciencia 100000 --pasta-2d <TCC_DADOS>/features_2d_flipswap"
  G5-S (só 3D, do zero):
    python 04_segmentacao_fusao/treinar_fusao.py $BASE --fusao nenhuma --pesos-iniciais nenhum --semente 42 \
        --passos 13000 --dados <TCC_DADOS>/fusao_dados --saida <TCC_DADOS>/runs5/G5-S.s42
  G5-FR (fusão residual, a partir do G5-S):
    python 04_segmentacao_fusao/treinar_fusao.py $BASE --fusao atencao --prior --portao-celula --peso-atencao 0.1 \
        --congelar 0 --residual --peso-erro-3d 5 --inicio-fusao 0 --semente 42 --lr 1e-3 --passos 6500 \
        --pesos-iniciais <TCC_DADOS>/runs5/G5-S.s42/best.pt --dados <TCC_DADOS>/fusao_dados_limpo_g5 --saida <TCC_DADOS>/runs5/G5-FR.s42
  Teste:
    python 04_segmentacao_fusao/treinar_fusao.py --arch lower --saida <TCC_DADOS>/runs5/G5-FR.s42 --avaliar --com-ampliacao \
        --dados <TCC_DADOS>/fusao_dados_limpo_g5 --pasta-2d <TCC_DADOS>/features_2d_flipswap
Saída: <saida>/best.pt, latest.pt, historico.csv; com --avaliar, teste_por_classe.csv e o resumo.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
import caminhos  # noqa: E402
from dados_fusao import FEATURES_2D, DatasetFusao, juntar  # noqa: E402
from modelo_fusao import MeshSegNetFusao, carregar_pesos_iniciais  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

DADOS = caminhos.FUSAO_DADOS
ORDER16 = {
    "upper": [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28],
    "lower": [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38],
}


def adjacencias(cells, curto=0.1, longo=0.2):
    """A_S/A_L (raios 0,1 e 0,2), montadas no device."""
    bary = cells[:, 9:12, :].transpose(1, 2).contiguous()
    d = torch.cdist(bary, bary, compute_mode="donot_use_mm_for_euclid_dist")
    a_l = (d < longo).to(d.dtype)
    d.lt_(curto)
    a_s = d
    a_s /= a_s.sum(-1, keepdim=True)
    a_l /= a_l.sum(-1, keepdim=True)
    return a_s, a_l


def dice_generalizada(prob, labels, eps=1.0):
    oh = torch.nn.functional.one_hot(labels, num_classes=prob.shape[-1]).to(prob.dtype)
    inter = (prob * oh).sum(dim=(0, 1))
    den = prob.sum(dim=(0, 1)) + oh.sum(dim=(0, 1))
    return 1.0 - ((2.0 * inter + eps) / (den + eps)).mean()


def ce_erro_rascunho(prob, labels, erro, peso_erro):
    """Entropia cruzada com peso `peso_erro` nas células em que o rascunho 3D erra e 1 nas outras.
    O Dice generalizado é uma média por classe e quase não enxerga ~7% de células trocadas dentro de classes
    grandes; este termo põe o gradiente justamente onde o 2D teria algo a corrigir."""
    logp = torch.log(prob.clamp_min(1e-8))
    ce = torch.nn.functional.nll_loss(logp.reshape(-1, logp.shape[-1]), labels.reshape(-1), reduction="none")
    w = 1.0 + (peso_erro - 1.0) * erro.reshape(-1).to(ce.dtype)
    return (w * ce).sum() / w.sum()


def perda_atencao(pesos, labels, fdi_2d, ignorar, arch):
    """Supervisão da atenção (média das cabeças). Cada célula de dente deve olhar para o token
    2D com o mesmo número FDI; gengiva e dentes sem token (não detectado ou no dropout) olham para o nulo (0).
    O GT entra só aqui, no treino: no teste a atenção casa célula e token pelo conteúdo e pelo prior.
    pesos (B, N, 1+T); labels (B, N) em 0..16; fdi_2d (B, T); ignorar (B, 1+T)."""
    l2f = torch.tensor([0] + ORDER16[arch], device=labels.device)
    fdi_cel = l2f[labels]                                                            # (B, N)
    casa = (fdi_cel.unsqueeze(-1) == fdi_2d.unsqueeze(1)) & ~ignorar[:, 1:].unsqueeze(1) & (fdi_cel > 0).unsqueeze(-1)
    alvo = torch.where(casa.any(-1), casa.float().argmax(-1) + 1, torch.zeros_like(labels))
    return -torch.log(pesos.clamp_min(1e-8)).gather(-1, alvo.unsqueeze(-1)).mean()


def dsc_por_classe(pred, labels, n_classes=17, smooth=1.0):
    """DSC por classe numa arcada (smooth 1) e quais classes existem no GT."""
    dsc, presente = np.zeros(n_classes), np.zeros(n_classes, bool)
    for c in range(n_classes):
        p, g = pred == c, labels == c
        presente[c] = g.any()
        dsc[c] = (2.0 * (p & g).sum() + smooth) / (p.sum() + g.sum() + smooth)
    return dsc, presente


def para_device(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def ce_pesada_3d(prob, p3d, labels, peso):
    """CE da P_fus com peso `peso` nas células em que a P_3D (do mesmo modelo) erra e 1 nas
    demais — a fusão é cobrada sobretudo onde o 3D falha, e também punida se estragar o que o 3D acertou."""
    w = 1 + (peso - 1) * (p3d.detach().argmax(-1) != labels).float()
    lp = torch.log(prob.clamp_min(1e-8)).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return -(w * lp).sum() / w.sum()


PRECISAO = "fp32"      # 'fp32' (G1–G4), 'tf32' ou 'bf16'; definida por configurar_precisao()


def configurar_precisao(p):
    """fp32 = como no G4. tf32 = matmul/bmm/conv em TF32 nos tensor cores (Ampere/Ada); a cdist das adjacências
    usa o modo sem matmul e fica exata. bf16 = tf32 + autocast bfloat16 no forward (softmax e perdas em fp32)."""
    global PRECISAO
    PRECISAO = p
    rapido = p in ("tf32", "bf16")
    torch.backends.cuda.matmul.allow_tf32 = rapido
    torch.backends.cudnn.allow_tf32 = rapido


def passo(modelo, b, sem_2d=False):
    usa_bf16 = PRECISAO == "bf16" and b["cells"].is_cuda
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=usa_bf16):
        a_s, a_l = adjacencias(b["cells"])
        prior = (b["centro"], b["x_tok"], b["sigma"], b["tem_prior"]) if "centro" in b else None
        prob, pesos = modelo(b["cells"], a_s, a_l, b["arco"], b["tokens"], b["valido"], b["glob"], sem_2d=sem_2d,
                             prior=prior, cls_tok=b.get("cls_tok"), rascunho_cel=b.get("rascunho_label"))
    if usa_bf16:
        prob = prob.float()
        pesos = pesos.float() if torch.is_tensor(pesos) else pesos
        if torch.is_tensor(getattr(modelo, "ultimo_p3d", None)):
            modelo.ultimo_p3d = modelo.ultimo_p3d.float()
    return prob, pesos


def novo_modelo(ck_ou_args):
    """MeshSegNetFusao a partir dos args do treino ou de um checkpoint (testar, robustez, diagnóstico)."""
    g = ck_ou_args.get if isinstance(ck_ou_args, dict) else (lambda k, d=None: getattr(ck_ou_args, k, d))
    return MeshSegNetFusao(fusao=g("fusao"), usar_probs=g("usar_probs", True), prior=g("prior", False),
                           portao_celula=g("portao_celula", False), portao_disc=g("portao_disc", 2.0),
                           portao_b0=g("portao_b0", -3.0), portao_x=g("portao_x", False),
                           assimetrica=g("assimetrica", False), residual=g("residual", False))


@torch.no_grad()
def avaliar(modelo, loader, dev, arch, sem_2d=False):
    """Uma linha por (caso, classe): DSC, se a classe está no GT, e o FDI correspondente."""
    modelo.eval()
    linhas = []
    l2f = [0] + ORDER16[arch]
    for b in loader:
        b = para_device(b, dev)
        prob, _ = passo(modelo, b, sem_2d)
        pred = prob.argmax(-1)[0].cpu().numpy()
        dsc, pres = dsc_por_classe(pred, b["labels"][0].cpu().numpy())
        linhas += [dict(case_id=b["case_id"][0], classe=c, fdi=l2f[c], dsc=dsc[c], no_gt=pres[c],
                        sem_2d=sem_2d) for c in range(len(dsc))]
    return pd.DataFrame(linhas)


def resumo(df):
    """dsc_17: média das 17 classes com as ausentes (métrica usual do MeshSegNet); dsc_dentes: só dentes do GT."""
    return dict(dsc_17=df.groupby("case_id").dsc.mean().mean(),
                dsc_dentes=df[(df.classe > 0) & df.no_gt].groupby("case_id").dsc.mean().mean())


def congelar_backbone(modelo, congelado):
    novos = {id(p) for p in modelo.parametros_novos()}
    for p in modelo.parameters():
        if id(p) not in novos:
            p.requires_grad_(not congelado)


def treinar(a):
    torch.manual_seed(a.semente)
    np.random.seed(a.semente)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configurar_precisao(a.precisao)
    a.saida.mkdir(parents=True, exist_ok=True)

    tr = DatasetFusao(a.dados, "treino", a.arch, patch=a.patch, augment=True, pasta_2d=a.pasta_2d,
                      dificil=a.dificil, com_complementar=a.com_complementar, rot_max=a.rot_max, rot_incl=a.rot_incl,
                      com_ampliacao=a.com_ampliacao, sem_dobra=a.sem_dobra)
    va = DatasetFusao(a.dados, "val", a.arch, patch=None, pasta_2d=a.pasta_2d, com_ampliacao=a.com_ampliacao)  # G5: val 52 + 226
    trl = DataLoader(tr, batch_size=a.batch, shuffle=True, drop_last=True, collate_fn=juntar,
                     num_workers=a.workers, pin_memory=dev.type == "cuda")
    val = DataLoader(va, batch_size=1, collate_fn=juntar, num_workers=a.workers)
    if a.passos:     # Orçamento em passos de otimização; vira épocas inteiras (arredonda p/ cima)
        a.epocas = -(-a.passos // len(trl))
        print(f"orçamento: {a.passos} passos = {a.epocas} épocas × {len(trl)} passos/época "
              f"(= {a.epocas * len(trl)} passos)")
    print(f"{a.arch} | fusão={a.fusao} | treino={len(tr)} val={len(va)} | patch={a.patch} batch={a.batch} | {dev}"
          f" | precisão={a.precisao}"
          f" | portão0={a.portao_inicial} difícil={a.dificil} ce={a.peso_ce}×(erro {a.peso_erro})"
          f" atenção={a.peso_atencao} | sem2D={a.peso_sem_2d}"
          + (f" | ASSIMÉTRICA (erro 3D ×{a.peso_erro_3d}, fusão a partir da ép. {a.inicio_fusao})" if a.assimetrica
             else "")
          + (f" | RESIDUAL (erro 3D ×{a.peso_erro_3d}, fusão a partir da ép. {a.inicio_fusao})" if a.residual
             else "") + " | rotação="
          + ("3D uniforme" if a.rot_max is None else f"±{a.rot_max}° vertical, ±{a.rot_incl}° inclinação"))

    modelo = MeshSegNetFusao(fusao=a.fusao, p_sem_2d=a.p_sem_2d, p_token=a.p_token, usar_probs=not a.sem_probs,
                             prior=a.prior, p_sem_prior=a.p_sem_prior, portao_inicial=a.portao_inicial,
                             portao_celula=a.portao_celula, portao_disc=a.portao_disc, portao_b0=a.portao_b0,
                             portao_x=a.portao_x, assimetrica=a.assimetrica, residual=a.residual)
    if a.pesos_iniciais is None:
        raise SystemExit("--pesos-iniciais: informe 'nenhum' (do zero) ou o checkpoint inicial (ex.: o best.pt do G5-S)")
    if a.pesos_iniciais != "nenhum":
        carregar_pesos_iniciais(modelo, a.pesos_iniciais)
    modelo.to(dev)
    novos = {id(p) for p in modelo.parametros_novos()}
    grupos = [{"params": [p for p in modelo.parameters() if id(p) not in novos], "lr": a.lr}]
    if novos:
        grupos.append({"params": modelo.parametros_novos(), "lr": a.lr_novos, "amsgrad": not a.sem_amsgrad_novos})
    opt = torch.optim.Adam(grupos, amsgrad=True, weight_decay=0.0)

    ep0, melhor, parado, hist = 0, -1.0, 0, []
    ultimo = a.saida / "latest.pt"
    if a.resume and ultimo.exists():
        ck = torch.load(ultimo, map_location=dev, weights_only=False)
        modelo.load_state_dict(ck["modelo"])
        opt.load_state_dict(ck["opt"])
        ep0, melhor, parado, hist = ck["epoca"], ck["melhor"], ck["parado"], ck["hist"]
        print(f"retomando da época {ep0} (melhor val_DSC {melhor:.4f})")

    for ep in range(ep0, a.epocas):
        fase1 = ep < a.congelar and novos
        congelar_backbone(modelo, bool(fase1))
        modelo.train()
        # BN do MeshSegNet com as estatísticas pré-treinadas (eval) na fase 1 e, sem --bn-livre, também na fase 2:
        # no treino curto, deixá-lo re-estimar com batch 2 × patch 2000 derrubou o val de 0,903 para
        # 0,789, e só trocar de volta as running_mean/var pré-treinadas devolveu 0,899 (os pesos estavam bons).
        if fase1 or not a.bn_livre:
            for nome, m in modelo.named_modules():
                if isinstance(m, torch.nn.BatchNorm1d) and nome.split(".")[0] not in {"tokens", "cruzada", "film", "cab_fus", "delta"}:
                    m.eval()
        t0, soma, n = time.time(), 0.0, 0
        for b in trl:
            b = para_device(b, dev)
            prob, pesos = passo(modelo, b)
            dupla = a.assimetrica or a.residual
            fusao_ativa = not dupla or ep >= a.inicio_fusao
            if dupla:                  # L_3D sempre; L_fus (Dice + CE pesada nos erros da P_3D) depois do aquecimento
                p3 = modelo.ultimo_p3d
                perda = dice_generalizada(p3, b["labels"])
                if a.residual:   # Z4d: CE simples também na P_3D; só com Dice, a L_fus (CE ×5) mandava ~20× mais
                    perda = perda + ce_pesada_3d(p3, p3, b["labels"], 1.0)   # gradiente ao tronco (teste_residual)
                if fusao_ativa:
                    perda = perda + dice_generalizada(prob, b["labels"]) + ce_pesada_3d(prob, p3, b["labels"],
                                                                                         a.peso_erro_3d)
            else:
                perda = dice_generalizada(prob, b["labels"])
            if a.peso_atencao > 0 and pesos is not None and fusao_ativa:
                perda = perda + a.peso_atencao * perda_atencao(pesos, b["labels"], b["fdi_2d"],
                                                               modelo.ultimo_ignorar, a.arch)
            if a.peso_ce > 0:
                perda = perda + a.peso_ce * ce_erro_rascunho(prob, b["labels"], b["erro_rascunho"], a.peso_erro)
            if a.peso_sem_2d > 0 and a.fusao != "nenhuma":
                # O mesmo lote de novo com o 2D desligado, para o caminho só 3D treinar em
                # 100% das amostras (com só o dropout de modalidade, o Z2 ficou 0,004 sem 2D).
                prob0, _ = passo(modelo, b, sem_2d=True)
                perda = perda + a.peso_sem_2d * dice_generalizada(prob0, b["labels"])
            opt.zero_grad(set_to_none=True)
            perda.backward()
            opt.step()
            soma, n = soma + perda.item() * len(b["case_id"]), n + len(b["case_id"])
        celula = a.fusao == "atencao" and a.portao_celula
        if celula:
            modelo.cruzada.zerar_g()
        r = resumo(avaliar(modelo, val, dev, a.arch))
        if celula:     # g médio no val (com 2D) e g nas células em que o 2D discorda do rascunho 3D (disc > 0,5)
            cz = modelo.cruzada
            portao, g_disc = cz.soma_g / max(cz.n_g, 1), cz.soma_g_disc / max(cz.n_g_disc, 1)
            frac_disc = cz.n_g_disc / max(cz.n_g, 1)
        r0 = resumo(avaliar(modelo, val, dev, a.arch, sem_2d=True)) if novos else r
        if not celula:
            portao = float(torch.tanh(modelo.cruzada.portao.detach())) if a.fusao == "atencao" else float("nan")
            g_disc = frac_disc = float("nan")
        lam = float(torch.nn.functional.softplus(modelo.cruzada.rho.detach()).mean()) if a.prior else float("nan")
        hist.append(dict(epoca=ep + 1, perda=soma / max(n, 1), val_dsc17=r["dsc_17"], val_dentes=r["dsc_dentes"],
                         val_dentes_sem_2d=r0["dsc_dentes"], portao=portao, lambda_prior=lam, fase=1 if fase1 else 2,
                         g_disc=g_disc, frac_disc=frac_disc))
        melhorou = r["dsc_17"] > melhor
        if melhorou:
            melhor, parado = r["dsc_17"], 0
            torch.save({"model_state_dict": modelo.state_dict(), "num_classes": 17, "fusao": a.fusao,
                        "arch": a.arch, "epoca": ep + 1, "val_dsc": float(melhor),        # float: o testar() usa weights_only
                        "usar_probs": not a.sem_probs, "prior": a.prior, "portao_celula": a.portao_celula,
                        "portao_disc": a.portao_disc, "portao_b0": a.portao_b0, "portao_x": a.portao_x,
                        "assimetrica": a.assimetrica, "residual": a.residual, "sem_dobra": a.sem_dobra,
                        "pesos_iniciais": str(a.pesos_iniciais), "precisao": a.precisao},          # G5: 'nenhum' ou o best.pt do G5-S (cascata)
                       a.saida / "best.pt")
        else:
            parado += 1
        torch.save(dict(modelo=modelo.state_dict(), opt=opt.state_dict(), epoca=ep + 1, melhor=melhor,
                        parado=parado, hist=hist), ultimo)
        pd.DataFrame(hist).to_csv(a.saida / "historico.csv", index=False)
        print(f"ép {ep + 1:3d} | fase {1 if fase1 else 2} | perda {soma / max(n, 1):.4f} | val DSC17 {r['dsc_17']:.4f} "
              f"dentes {r['dsc_dentes']:.4f} (sem 2D {r0['dsc_dentes']:.4f}) | portão {portao:+.3f} | λ {lam:.2f} | "
              + (f"g disc {g_disc:.3f} ({100 * frac_disc:.1f}% céls) | " if celula else "")
              + f"{time.time() - t0:.0f}s{' *' if melhorou else ''}", flush=True)
        if parado >= a.paciencia:
            print(f"early stopping: {a.paciencia} épocas sem melhorar")
            break


def testar(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configurar_precisao(a.precisao)
    if a.pretreinado:       # referência: os pesos pré-treinados (só malhas 3D), sem ajuste fino
        a.saida.mkdir(parents=True, exist_ok=True)
        if not a.pesos_iniciais or a.pesos_iniciais == "nenhum":
            raise SystemExit("--pretreinado exige --pesos-iniciais com o checkpoint de referência")
        modelo = carregar_pesos_iniciais(MeshSegNetFusao(fusao="nenhuma"), a.pesos_iniciais).to(dev)
        ck = {"fusao": "nenhuma", "epoca": 0}
    else:
        ck = torch.load(a.saida / "best.pt", map_location=dev, weights_only=True)
        modelo = novo_modelo(ck).to(dev)
        modelo.load_state_dict(ck["model_state_dict"])
    te = DataLoader(DatasetFusao(a.dados, "teste", a.arch, patch=None, pasta_2d=a.pasta_2d,
                                 com_ampliacao=a.com_ampliacao),          # G5: teste 139 + 279
                    batch_size=1, collate_fn=juntar)
    df = avaliar(modelo, te, dev, a.arch)
    if ck["fusao"] != "nenhuma":
        df = pd.concat([df, avaliar(modelo, te, dev, a.arch, sem_2d=True)])
    df.to_csv(a.saida / "teste_por_classe.csv", index=False)
    for s, g in df.groupby("sem_2d"):
        r = resumo(g)
        print(f"teste {a.arch} {'SEM 2D' if s else 'com 2D'}: DSC17 {r['dsc_17']:.4f} | dentes {r['dsc_dentes']:.4f} "
              f"| {g.case_id.nunique()} arcadas (época {ck['epoca']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["upper", "lower"], required=True)
    ap.add_argument("--fusao", choices=["nenhuma", "film", "atencao"], default="atencao")
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--dados", type=Path, default=DADOS)
    ap.add_argument("--pasta-2d", type=Path, default=FEATURES_2D)
    ap.add_argument("--pesos-iniciais", default=None, help="checkpoint inicial; 'nenhum' = do zero")
    ap.add_argument("--epocas", type=int, default=200)
    ap.add_argument("--passos", type=int, default=None,
                    help="orçamento em passos; substitui --epocas por ceil(passos / passos por época)")
    ap.add_argument("--congelar", type=int, default=10, help="épocas só com as camadas novas")
    ap.add_argument("--bn-livre", action="store_true",
                    help="na fase 2, o BN do MeshSegNet re-estima as estatísticas (padrão: fica com as pré-treinadas)")
    ap.add_argument("--lr", type=float, default=1e-4, help="lr do MeshSegNet")
    ap.add_argument("--lr-novos", type=float, default=1e-3, help="camadas da fusão")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patch", type=int, default=7000)
    ap.add_argument("--p-sem-2d", type=float, default=0.2)
    ap.add_argument("--p-token", type=float, default=0.1)
    ap.add_argument("--sem-probs", action="store_true", help="ablação: tokens 2D sem as probs FDI")
    ap.add_argument("--prior", action="store_true",
                    help="atenção GUIADA (o modelo proposto): prior do registro nos logits; sem ela, atenção livre")
    ap.add_argument("--p-sem-prior", type=float, default=0.1, help="dropout do prior no treino (por amostra)")
    ap.add_argument("--portao-inicial", type=float, default=0.0,
                    help="tanh do portão no início (0 = começa como o MeshSegNet; 0 travou a fusão)")
    ap.add_argument("--sem-amsgrad-novos", action="store_true", help="Adam sem AMSGrad nas camadas da fusão")
    ap.add_argument("--dificil", type=float, default=0.0,
                    help="fração do patch tirada das células em que o rascunho 3D erra (0 = como antes)")
    ap.add_argument("--peso-ce", type=float, default=0.0,
                    help="peso da entropia cruzada somada ao Dice (0 = só Dice, como antes)")
    ap.add_argument("--peso-erro", type=float, default=5.0, help="peso da CE nas células em que o rascunho 3D erra")
    ap.add_argument("--peso-atencao", type=float, default=0.0,
                    help="supervisão da atenção: célula → token 2D com o mesmo FDI (0 = desligada)")
    ap.add_argument("--portao-celula", action="store_true",
                    help="portão POR CÉLULA guiado pela discordância 2D × rascunho 3D no lugar do escalar")
    ap.add_argument("--portao-disc", type=float, default=2.0, help="peso inicial da discordância no portão por célula")
    ap.add_argument("--portao-b0", type=float, default=-3.0, help="viés inicial do portão por célula (σ(−3) ≈ 0,05)")
    ap.add_argument("--portao-x", action="store_true", help="o portão por célula também lê as features da célula")
    ap.add_argument("--com-complementar", action="store_true",
                    help="soma ao treino os 121 casos do treino do MeshSegNet pré-treinado (manifesto_complementar.csv)")
    ap.add_argument("--assimetrica", action="store_true",
                    help="perda assimétrica (tronco só aprende com a P_3D; fusão com stop-gradient)")
    ap.add_argument("--sem-dobra", default=None,
                    help="'k/K' treina sem a dobra k do treino (rascunhos fora da dobra)")
    ap.add_argument("--residual", action="store_true",
                    help="P_fus = softmax(z_3D + Δ), Δ zerado no início; L_3D + L_fus sem stop-gradient")
    ap.add_argument("--peso-erro-3d", type=float, default=5.0,
                    help="peso da CE da P_fus nas células em que a P_3D erra (com --assimetrica)")
    ap.add_argument("--inicio-fusao", type=int, default=20,
                    help="épocas só com a L_3D antes de ligar a L_fus (com --assimetrica)")
    ap.add_argument("--peso-sem-2d", type=float, default=0.0,
                    help="perda dupla, L(com 2D) + α·L(sem 2D) em cada lote; 0 = desligada")
    ap.add_argument("--com-ampliacao", action="store_true",
                    help="soma ao treino os casos da ampliação (manifesto_ampliacao.csv)")
    ap.add_argument("--rot-max", type=float, default=None,
                    help="graus de giro em torno da vertical (Y) na augmentation; omitido = rotação 3D "
                         "uniforme (como Z1/Z2)")
    ap.add_argument("--rot-incl", type=float, default=5.0, help="graus de inclinação (X e Z) com --rot-max")
    ap.add_argument("--paciencia", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--precisao", choices=["fp32", "tf32", "bf16"], default="fp32",
                    help="fp32 = como no G4; tf32/bf16 = mais rápido na L4/T4")
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--avaliar", action="store_true", help="só roda o best.pt no teste")
    ap.add_argument("--pretreinado", action="store_true",
                    help="com --avaliar: avalia os pesos pré-treinados sem ajuste fino (a referência da tabela)")
    a = ap.parse_args()
    testar(a) if a.avaliar else treinar(a)


if __name__ == "__main__":
    main()
