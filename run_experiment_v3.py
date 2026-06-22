import os, warnings, argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import wilcoxon
import pandas as pd

warnings.filterwarnings("ignore")

from dataset_generator import (
    load_cifar10_clean, load_cifar10c,
    load_openml_dataset, perturb_tabular, make_dataloader,
    CORRUPTION_TYPES,
)
from model_simulator import build_model, ViTWrapper
from bfmp_metrics import (
    entropy_spike_score, activation_drift_distance,
    output_distribution_dispersion, attention_collapse_detector_vit,
    odd_trajectory, ess_trajectory, compound_failure_signal,
    LearnedCFS, FailureHorizonRegressor, compute_failure_horizon,
)
from baselines import (
    MCDropoutPredictor, ODINPredictor,
    AutoencoderBaseline, AutoencoderPredictor,
)

INFER_BATCH = 64



def batched_softmax_acts(model, imgs_tensor, device, bs=INFER_BATCH):
    model.eval()
    all_sm, all_acts = [], {}
    for start in range(0, imgs_tensor.shape[0], bs):
        xb = imgs_tensor[start:start+bs].to(device)
        with torch.no_grad():
            sm = F.softmax(model(xb), dim=-1).cpu().numpy()
        ac = {k: v.cpu().numpy() for k, v in model.get_activations(xb).items()}
        all_sm.append(sm)
        for k, v in ac.items():
            all_acts.setdefault(k, []).append(v)
        del xb; torch.cuda.empty_cache()
    return (np.concatenate(all_sm, 0),
            {k: np.concatenate(v, 0) for k, v in all_acts.items()})


def batched_attn_entropy(model, imgs_tensor, device, bs=INFER_BATCH):
    model.eval()
    all_H = []
    for start in range(0, imgs_tensor.shape[0], bs):
        xb = imgs_tensor[start:start+bs].to(device)
        all_H.append(model.get_attention_entropy(xb).cpu().numpy())
        del xb; torch.cuda.empty_cache()
    return np.concatenate(all_H, 0)


def batched_uncertainty(fn, imgs_tensor, device, bs=INFER_BATCH):
    out = []
    for start in range(0, imgs_tensor.shape[0], bs):
        xb = imgs_tensor[start:start+bs].to(device)
        u  = fn(xb)
        out.append(u if isinstance(u, np.ndarray) else u.cpu().numpy())
        del xb; torch.cuda.empty_cache()
    return np.concatenate(out, 0)



def train_model(model, dataloader, n_epochs=30, lr=1e-3, device="cpu"):
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    model.to(device).train()
    for epoch in range(n_epochs):
        total, correct, rloss = 0, 0, 0.0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss   = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            rloss   += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total   += y.size(0)
        sched.step()
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:3d}/{n_epochs}  "
                  f"loss={rloss/total:.4f}  acc={100*correct/total:.1f}%")
    model.eval()
    return model

def extract_trajectory_cifar(model, corruption, n_samples=200,
                              data_dir="./data", device="cpu"):
    imgs_clean, lbls_clean = load_cifar10_clean(data_dir, train=False,
                                                n_samples=n_samples)
    sm_clean, acts_clean   = batched_softmax_acts(model, imgs_clean, device)

    softmax_seq = [sm_clean]
    acts_seq    = [acts_clean]
    img_seq     = [imgs_clean]

    for sev in [1, 2, 3, 4, 5]:
        imgs_c, _ = load_cifar10c(corruption, sev, data_dir, n_samples)
        sm_c, ac_c = batched_softmax_acts(model, imgs_c, device)
        softmax_seq.append(sm_c)
        acts_seq.append(ac_c)
        img_seq.append(imgs_c)

    labels   = lbls_clean.numpy()
    ess_traj = ess_trajectory(softmax_seq)
    add_traj = np.stack([activation_drift_distance(acts_clean, acts_seq[t])
                         for t in range(len(acts_seq))])
    odd_traj = odd_trajectory(softmax_seq)

    if isinstance(model, ViTWrapper):
        attn_seq = [batched_attn_entropy(model, imgs, device) for imgs in img_seq]
        acd_traj = attention_collapse_detector_vit(attn_seq)
    else:
        acd_traj = odd_traj

    horizon = compute_failure_horizon(softmax_seq, labels)

    return {
        "softmax_seq": softmax_seq, "acts_seq": acts_seq,
        "img_seq": img_seq, "labels": labels,
        "ess_traj": ess_traj, "add_traj": add_traj,
        "odd_traj": odd_traj, "acd_traj": acd_traj,
        "horizon": horizon,
    }


def evaluate_detection(signal, horizon, k=1):
    y_true  = (horizon.flatten() <= k).astype(int)
    y_score = signal.flatten()
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"AUROC": float("nan"), "AUPR": float("nan")}
    return {
        "AUROC": roc_auc_score(y_true, y_score),
        "AUPR":  average_precision_score(y_true, y_score),
    }


def run_single_seed(arch, dataset, seed, n_samples=200, n_epochs=30,
                    device="cpu", data_dir="./data", results_dir="./results"):
    print(f"\n{'='*60}\n arch={arch}  dataset={dataset}  seed={seed}\n{'='*60}")
    torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(results_dir, exist_ok=True)


    if dataset == "cifar10c":
        imgs_clean, lbls_clean = load_cifar10_clean(data_dir, train=True,
                                                    n_samples=5000)
        train_dl  = DataLoader(TensorDataset(imgs_clean, lbls_clean),
                               batch_size=64, shuffle=True)
        n_classes, in_feat = 10, None
    else:
        X_tr, X_te, y_tr, y_te = load_openml_dataset(dataset,
                                                      cache_dir=data_dir)
        in_feat   = X_tr.shape[1]
        n_classes = int(y_tr.max()) + 1
        train_dl  = make_dataloader(X_tr.astype(np.float32), y_tr,
                                    batch_size=256, shuffle=True)


    model = build_model(arch, in_features=in_feat, n_classes=n_classes,
                        device=device)
    print(f"[train] {arch.upper()} params: "
          f"{sum(p.numel() for p in model.parameters()):,}")
    model = train_model(model, train_dl, n_epochs=n_epochs, device=device)

    if dataset == "cifar10c":
        traj = extract_trajectory_cifar(
            model, CORRUPTION_TYPES[0], n_samples, data_dir, device)
    else:
        noise_levels = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
        softmax_seq, acts_seq = [], []
        for nl in noise_levels:
            X_n = perturb_tabular(X_te, nl, seed=seed) if nl > 0 else X_te
            sm, ac = batched_softmax_acts(
                model, torch.from_numpy(X_n.astype(np.float32)), device)
            softmax_seq.append(sm); acts_seq.append(ac)
        labels   = y_te
        ess_t    = ess_trajectory(softmax_seq)
        add_t    = np.stack([activation_drift_distance(acts_seq[0], acts_seq[t])
                             for t in range(len(acts_seq))])
        odd_t    = odd_trajectory(softmax_seq)
        horizon  = compute_failure_horizon(softmax_seq, labels)
        traj = {"softmax_seq": softmax_seq, "acts_seq": acts_seq,
                "img_seq": None, "labels": labels,
                "ess_traj": ess_t, "add_traj": add_t,
                "odd_traj": odd_t, "acd_traj": odd_t,
                "horizon": horizon}

    ess_traj = traj["ess_traj"]
    add_traj = traj["add_traj"]
    odd_traj = traj["odd_traj"]
    horizon  = traj["horizon"]
    T        = ess_traj.shape[0]
    split    = max(1, T // 2)

    lcfs = LearnedCFS(k=1, window=2, C=1.0)
    lcfs.fit(ess_traj[:split], add_traj[:split],
             odd_traj[:split], horizon[:split])
    cfs_score = lcfs.predict_proba(ess_traj[split:], add_traj[split:],
                                    odd_traj[split:])
    det_bfmp  = evaluate_detection(cfs_score, horizon[split:])

    ablation = lcfs.ablation_scores(
        ess_traj[split:], add_traj[split:],
        odd_traj[split:], horizon[split:]
    )


    fhr = FailureHorizonRegressor(alpha=1.0, window=2)
    fhr.fit(ess_traj[:split], add_traj[:split],
            odd_traj[:split], horizon[:split])
    fhr_sc = fhr.score(ess_traj[split:], add_traj[split:],
                       odd_traj[split:], horizon[split:])

    if seed == 0 and dataset == "cifar10c":
        tp = os.path.join(results_dir, f"{arch}_{dataset}_trajectory.npz")
        np.savez(tp,
                 ess_traj=ess_traj, add_traj=add_traj,
                 odd_traj=odd_traj, horizon=horizon,
                 cfs_traj=cfs_score,
                 ablation_ess=ablation.get("ESS only", float("nan")),
                 ablation_add=ablation.get("ADD only", float("nan")),
                 ablation_odd=ablation.get("ODD only", float("nan")),
                 ablation_naive=ablation.get("ESS+ADD (naive)", float("nan")),
                 ablation_lcfs=ablation.get("Full LCFS", float("nan")))
        print(f"[saved] trajectory → {tp}")

 
    det_mcd = det_odin = det_ae = {"AUROC": float("nan"), "AUPR": float("nan")}
    if dataset == "cifar10c":
        img_seq = traj["img_seq"]

        mcd   = MCDropoutPredictor(model, n_samples=10, device=device)
        mcd_u = np.stack([batched_uncertainty(
            lambda x: mcd.predict_uncertainty(x), imgs, device)
            for imgs in img_seq])
        det_mcd = evaluate_detection(mcd_u, horizon)

        odin   = ODINPredictor(model, device=device)
        odin_u = np.stack([batched_uncertainty(
            lambda x: odin.predict_uncertainty(x), imgs, device)
            for imgs in img_seq])
        det_odin = evaluate_detection(odin_u, horizon)

        ae_m = AutoencoderBaseline(in_channels=3).to(device)
        ae_p = AutoencoderPredictor(ae_m, device=device)
        ae_p.fit(DataLoader(
            TensorDataset(img_seq[0],
                          torch.zeros(img_seq[0].shape[0], dtype=torch.long)),
            batch_size=64), n_epochs=5)
        ae_u = np.stack([batched_uncertainty(
            lambda x: ae_p.predict_uncertainty(x), imgs, device)
            for imgs in img_seq])
        det_ae = evaluate_detection(ae_u, horizon)

    return {
        "seed": seed, "arch": arch, "dataset": dataset,
        "BFMP_AUROC":  det_bfmp["AUROC"], "BFMP_AUPR":  det_bfmp["AUPR"],
        "FHR_MAE":     fhr_sc["MAE"],     "FHR_RMSE":   fhr_sc["RMSE"],
        "FHR_R2":      fhr_sc["R2"],
        "MCD_AUROC":   det_mcd["AUROC"],  "MCD_AUPR":   det_mcd["AUPR"],
        "ODIN_AUROC":  det_odin["AUROC"], "ODIN_AUPR":  det_odin["AUPR"],
        "AE_AUROC":    det_ae["AUROC"],   "AE_AUPR":    det_ae["AUPR"],
        "ABL_ESS":     ablation.get("ESS only", float("nan")),
        "ABL_ADD":     ablation.get("ADD only", float("nan")),
        "ABL_ODD":     ablation.get("ODD only", float("nan")),
        "ABL_NAIVE":   ablation.get("ESS+ADD (naive)", float("nan")),
        "ABL_LCFS":    ablation.get("Full LCFS", float("nan")),
    }


def aggregate_and_print(results):
    df = pd.DataFrame(results)
    print("\n" + "="*65 + "\n RESULTS (mean ± std)\n" + "="*65)
    summary = {}
    for col in df.columns:
        if col in ("seed", "arch", "dataset"): continue
        vals = df[col].dropna().values
        if len(vals):
            summary[col] = {"mean": float(vals.mean()), "std": float(vals.std())}
            print(f"  {col:<22s}: {vals.mean():.4f} ± {vals.std():.4f}")

    print("\n Wilcoxon signed-rank tests (BFMP vs baselines):")
    for bl in ("MCD", "ODIN", "AE"):
        bv = df["BFMP_AUROC"].dropna().values
        bbase = df[f"{bl}_AUROC"].dropna().values
        n = min(len(bv), len(bbase))
        if n >= 3:
            try:
                stat, p = wilcoxon(bv[:n], bbase[:n])
                sig = "✓ p<0.05" if p < 0.05 else "✗ n.s."
                print(f"  BFMP vs {bl}: W={stat:.1f} p={p:.4f} {sig}")
            except Exception as e:
                print(f"  BFMP vs {bl}: {e}")
    return pd.DataFrame(summary).T


def generate_latex(summary, arch, dataset):
    lines = [r"\begin{table}[ht]", r"\centering",
             rf"\caption{{Failure detection ({arch.upper()}, {dataset}, 5 seeds).}}",
             r"\label{tab:main}", r"\begin{tabular}{lcc}",
             r"\toprule", r"Method & AUROC & AUPR \\", r"\midrule"]
    for method, ca, cp in [
        ("BFMP/LCFS (ours)", "BFMP_AUROC", "BFMP_AUPR"),
        ("MC Dropout",       "MCD_AUROC",  "MCD_AUPR"),
        ("ODIN",             "ODIN_AUROC", "ODIN_AUPR"),
        ("Autoencoder",      "AE_AUROC",   "AE_AUPR"),
    ]:
        if ca in summary.index:
            a, as_ = summary.loc[ca,"mean"], summary.loc[ca,"std"]
            p, ps  = summary.loc[cp,"mean"], summary.loc[cp,"std"]
            b = r"\textbf{" if "ours" in method else ""
            e = "}"         if "ours" in method else ""
            lines.append(f"{method} & {b}{a:.3f}$\\pm${as_:.3f}{e} & "
                         f"{b}{p:.3f}$\\pm${ps:.3f}{e} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",      default="cnn",
                        choices=["mlp","cnn","vit"])
    parser.add_argument("--dataset",   default="cifar10c",
                        choices=["cifar10c","credit-g","adult","phoneme"])
    parser.add_argument("--seeds",     type=int,  default=5)
    parser.add_argument("--n_samples", type=int,  default=200)
    parser.add_argument("--n_epochs",  type=int,  default=30)
    parser.add_argument("--device",    default="cuda"
                        if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_dir",  default="./data")
    parser.add_argument("--out_dir",   default="./results")
    args = parser.parse_args()

    print(f"[info] device={args.device}  n_samples={args.n_samples}")
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = []
    for seed in range(args.seeds):
        r = run_single_seed(
            arch=args.arch, dataset=args.dataset, seed=seed,
            n_samples=args.n_samples, n_epochs=args.n_epochs,
            device=args.device, data_dir=args.data_dir,
            results_dir=args.out_dir,
        )
        all_results.append(r)

    summary  = aggregate_and_print(all_results)
    csv_path = os.path.join(args.out_dir,
                            f"{args.arch}_{args.dataset}_results.csv")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"\n[saved] {csv_path}")

    latex    = generate_latex(summary, args.arch, args.dataset)
    tex_path = os.path.join(args.out_dir,
                            f"{args.arch}_{args.dataset}_table.tex")
    with open(tex_path, "w") as f: f.write(latex)
    print(f"[saved] {tex_path}\n\n{latex}")


if __name__ == "__main__":
    main()