from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bfmp_metrics import (
    entropy_spike_score, activation_drift_distance,
    output_distribution_dispersion,
    attention_collapse_detector_vit,
    compound_failure_signal,
    FailureHorizonRegressor,
    compute_failure_horizon,
    ess_trajectory, add_trajectory, odd_trajectory,
)
from model_simulator import ViTWrapper

class ArchitectureSignatureExtractor:

    def __init__(self, model: nn.Module, arch: str, device: str = "cpu"):
        self.model  = model.to(device)
        self.device = device
        self.arch   = arch.lower()
        assert self.arch in ("mlp", "cnn", "vit"), f"Unknown arch: {arch}"

    def extract(self, X: torch.Tensor) -> Dict[str, np.ndarray]:

        X = X.to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            sm     = F.softmax(logits, dim=-1).cpu().numpy()
            acts   = self.model.get_activations(X)
            acts   = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                      for k, v in acts.items()}

        attn_H = None
        if self.arch == "vit" and isinstance(self.model, ViTWrapper):
            attn_H = self.model.get_attention_entropy(X).cpu().numpy()  # (N, n_blocks)

        return {
            "softmax":      sm,
            "activations":  acts,
            "attn_entropy": attn_H,
        }

    def signature_name(self) -> str:
        
        sigs = {
            "mlp": "MLP-Sig: [ESS, ADD-dense, ODD]",
            "cnn": "CNN-Sig: [ESS, ADD-conv, ODD]",
            "vit": "ViT-Sig: [ESS, ADD-attn, ACD]",
        }
        return sigs[self.arch]


class BFMPPipeline:

    def __init__(
        self,
        model:  nn.Module,
        arch:   str,
        device: str = "cpu",
        fhr_alpha:  float = 1.0,
        fhr_window: int   = 3,
    ):
        self.extractor = ArchitectureSignatureExtractor(model, arch, device)
        self.arch      = arch
        self.device    = device
        self.fhr       = FailureHorizonRegressor(alpha=fhr_alpha, window=fhr_window)

        self._clean_acts:     Optional[Dict[str, np.ndarray]] = None
        self._clean_softmax:  Optional[np.ndarray]            = None
        self._trajectory_cache: List[Dict]                    = []

    def calibrate(self, clean_loader) -> None:

        acts_list, sm_list = [], []
        for X, _ in clean_loader:
            signals = self.extractor.extract(X)
            sm_list.append(signals["softmax"])
            for k, v in signals["activations"].items():
                acts_list.append({k: v})

        
        self._clean_softmax = np.concatenate(sm_list, axis=0)
        sample_sig = self.extractor.extract(
            next(iter(clean_loader))[0]
        )
        self._clean_acts = {
            k: v.mean(axis=0, keepdims=True) * np.ones((1, v.shape[-1]))
               if v.ndim == 2 else v
            for k, v in sample_sig["activations"].items()
        }
        print(f"[BFMP] Calibrated.  Arch signature: {self.extractor.signature_name()}")

    # ------------------------------------------------------------------ #
    def extract_signals(self, X: torch.Tensor) -> Dict[str, np.ndarray]:
        sig     = self.extractor.extract(X)
        sm      = sig["softmax"]
        acts    = sig["activations"]

        ess = entropy_spike_score(sm)
        odd = output_distribution_dispersion(sm)

        if self._clean_acts is not None:
            N   = sm.shape[0]
            clean_expanded = {
                k: np.repeat(v, N, axis=0)
                   if v.shape[0] == 1 else v
                for k, v in self._clean_acts.items()
            }
            add = activation_drift_distance(clean_expanded, acts)
        else:
            add = np.zeros(sm.shape[0])

        if self.arch == "vit" and sig["attn_entropy"] is not None:
            H_t    = sig["attn_entropy"]             
            H_max  = H_t.max(axis=0, keepdims=True)
            acd    = np.clip(H_max - H_t, 0, None).mean(axis=1)  
        else:
            acd = odd.copy()                          
        ess_n = (ess - ess.min()) / (ess.max() - ess.min() + 1e-8)
        add_n = (add - add.min()) / (add.max() - add.min() + 1e-8)
        cfs   = ess_n * add_n

        return {
            "ESS":  ess,
            "ADD":  add,
            "ODD":  odd,
            "ACD":  acd,
            "CFS":  cfs,
        }

    def monitor_trajectory(
        self,
        X_seq: List[torch.Tensor],
    ) -> Dict[str, np.ndarray]:

        T = len(X_seq)
        ess_list, add_list, odd_list, acd_list, cfs_list = [], [], [], [], []

        for X in X_seq:
            s = self.extract_signals(X)
            ess_list.append(s["ESS"])
            add_list.append(s["ADD"])
            odd_list.append(s["ODD"])
            acd_list.append(s["ACD"])
            cfs_list.append(s["CFS"])

        return {
            "ESS": np.stack(ess_list),
            "ADD": np.stack(add_list),
            "ODD": np.stack(odd_list),
            "ACD": np.stack(acd_list),
            "CFS": np.stack(cfs_list),
        }

    def fit_fhr(
        self,
        traj:    Dict[str, np.ndarray],
        horizon: np.ndarray,
    ) -> None:

        self.fhr.fit(traj["ESS"], traj["ADD"], traj["ODD"], horizon)
        print("[BFMP] FHR fitted.")

    def predict_horizon(
        self,
        traj: Dict[str, np.ndarray],
    ) -> np.ndarray:
    
        return self.fhr.predict(traj["ESS"], traj["ADD"], traj["ODD"])

    def evaluate(
        self,
        traj:    Dict[str, np.ndarray],
        horizon: np.ndarray,
        k:       int = 1,
    ) -> Dict[str, float]:
        from sklearn.metrics import roc_auc_score, average_precision_score

        y_true  = (horizon.flatten() <= k).astype(int)
        y_score = traj["CFS"].flatten()

        auroc  = roc_auc_score(y_true, y_score)  if y_true.sum() > 0 else float("nan")
        aupr   = average_precision_score(y_true, y_score) if y_true.sum() > 0 else float("nan")
        fhr_sc = self.fhr.score(traj["ESS"], traj["ADD"], traj["ODD"], horizon)

        return {
            "AUROC": auroc,
            "AUPR":  aupr,
            **fhr_sc,
        }
