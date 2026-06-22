from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance


class TemperatureScaling(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) / self.temperature

    def fit(
        self,
        val_loader,
        n_epochs: int  = 50,
        lr:       float = 0.01,
        device:   str  = "cpu",
    ):
        """Optimise T* on a validation set using NLL loss."""
        self.to(device)
        self.model.eval()
        opt = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=n_epochs)

        logits_list, labels_list = [], []
        with torch.no_grad():
            for X, y in val_loader:
                logits_list.append(self.model(X.to(device)).cpu())
                labels_list.append(y)
        logits_all = torch.cat(logits_list)
        labels_all = torch.cat(labels_list)

        def eval_fn():
            opt.zero_grad()
            loss = F.cross_entropy(logits_all / self.temperature.cpu(), labels_all)
            loss.backward()
            return loss

        opt.step(eval_fn)
        print(f"[TempScale] Optimal T* = {self.temperature.item():.4f}")
        return self

    def predict_uncertainty(self, X: torch.Tensor) -> np.ndarray:
        """Returns 1 - max_softmax_prob (lower confidence = higher uncertainty)."""
        with torch.no_grad():
            logits = self.forward(X)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()
        return 1.0 - probs.max(axis=1)

class EnergyScorePredictor:
    def __init__(self, model: nn.Module, temperature: float = 1.0, device: str = "cpu"):
        self.model       = model.to(device)
        self.temperature = temperature
        self.device      = device

    def predict_uncertainty(self, X: torch.Tensor) -> np.ndarray:
        X = X.to(self.device)
        with torch.no_grad():
            logits = self.model(X)
            energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
        return energy.cpu().numpy()

    def predict_uncertainty_trajectory(self, X_seq: List[torch.Tensor]) -> np.ndarray:
        return np.stack([self.predict_uncertainty(X) for X in X_seq])


class MahalanobisPredictor:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model    = model.to(device)
        self.device   = device
        self.class_means: Optional[np.ndarray]  = None
        self.precision:   Optional[np.ndarray]  = None
        self._fitted = False

    def _extract_penultimate(self, X: torch.Tensor) -> np.ndarray:
        X = X.to(self.device)
        acts = self.model.get_activations(X)
        last_key = list(acts.keys())[-1]
        return acts[last_key].cpu().numpy() if isinstance(acts[last_key], torch.Tensor) \
               else acts[last_key]

    def fit(self, dataloader, n_classes: int):
        
        feats_by_class = {c: [] for c in range(n_classes)}
        for X, y in dataloader:
            z = self._extract_penultimate(X)
            for i, c in enumerate(y.numpy()):
                feats_by_class[c].append(z[i])

        all_feats = []
        class_means = []
        for c in range(n_classes):
            if feats_by_class[c]:
                fc = np.stack(feats_by_class[c])
                class_means.append(fc.mean(axis=0))
                all_feats.append(fc - fc.mean(axis=0))
            else:
                class_means.append(np.zeros_like(class_means[0]) if class_means else np.zeros(1))

        self.class_means = np.stack(class_means)      
        all_feats = np.concatenate(all_feats, axis=0) 

        cov_estimator = EmpiricalCovariance(assume_centered=True)
        cov_estimator.fit(all_feats)
        self.precision = cov_estimator.precision_     
        self._fitted   = True
        print(f"[Mahal] Fitted on {len(all_feats)} samples, {n_classes} classes.")

    def predict_uncertainty(self, X: torch.Tensor) -> np.ndarray:
        assert self._fitted, "Call fit() first"
        z = self._extract_penultimate(X)               
        scores = []
        for z_i in z:
            dists = []
            for mu_c in self.class_means:
                d = z_i - mu_c
                dist = d @ self.precision @ d
                dists.append(dist)
            scores.append(min(dists))
        return np.array(scores)

    def predict_uncertainty_trajectory(self, X_seq: List[torch.Tensor]) -> np.ndarray:
        return np.stack([self.predict_uncertainty(X) for X in X_seq])


class ConformalPredictor:

    def __init__(self, model: nn.Module, alpha: float = 0.1, device: str = "cpu"):
        self.model     = model.to(device)
        self.alpha     = alpha
        self.device    = device
        self.threshold = None

    def calibrate(self, cal_loader):
        scores = []
        self.model.eval()
        with torch.no_grad():
            for X, y in cal_loader:
                logits = self.model(X.to(self.device))
                probs  = F.softmax(logits, dim=-1).cpu().numpy()
                for i, yi in enumerate(y.numpy()):
                    scores.append(1.0 - probs[i, yi])
        scores = np.array(scores)
        n      = len(scores)
        level  = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.threshold = np.quantile(scores, min(level, 1.0))
        print(f"[Conformal] Threshold q̂ = {self.threshold:.4f} at α={self.alpha}")

    def predict_set(self, X: torch.Tensor) -> List[List[int]]:

        assert self.threshold is not None, "Call calibrate() first"
        with torch.no_grad():
            logits = self.model(X.to(self.device))
            probs  = F.softmax(logits, dim=-1).cpu().numpy()
        sets = []
        for p in probs:
            pred_set = [c for c, pc in enumerate(p) if 1.0 - pc <= self.threshold]
            sets.append(pred_set)
        return sets

    def predict_uncertainty(self, X: torch.Tensor) -> np.ndarray:
        sets = self.predict_set(X)
        return np.array([len(s) for s in sets], dtype=float)

    def predict_uncertainty_trajectory(self, X_seq: List[torch.Tensor]) -> np.ndarray:
        return np.stack([self.predict_uncertainty(X) for X in X_seq])
