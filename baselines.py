from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class MCDropoutPredictor:
    def __init__(self, model: nn.Module, n_samples: int = 30, device: str = "cpu"):
        self.model     = model.to(device)
        self.n_samples = n_samples
        self.device    = device

    def predict_uncertainty(self, X: torch.Tensor) -> np.ndarray:
        X        = X.to(self.device)
        samples  = self.model.mc_dropout_forward(X, self.n_samples)  # (S, N, C)
        mean_p   = samples.mean(dim=0).cpu().numpy()                  # (N, C)
        eps      = 1e-10
        p        = np.clip(mean_p, eps, 1.0)
        entropy  = -(p * np.log(p)).sum(axis=1)
        return entropy

    def predict_uncertainty_trajectory(
        self, X_seq: List[torch.Tensor]
    ) -> np.ndarray:
        return np.stack([self.predict_uncertainty(X) for X in X_seq])


class DeepEnsemblePredictor:

    def __init__(
        self,
        model_builder,           # callable () → nn.Module
        n_members: int = 5,
        device:    str = "cpu",
    ):
        self.device  = device
        self.members = [model_builder().to(device) for _ in range(n_members)]
        self._trained = [False] * n_members

    def fit_member(
        self,
        member_idx: int,
        dataloader,
        n_epochs:   int = 10,
        lr:         float = 1e-3,
    ):
        model = self.members[member_idx]
        opt   = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for epoch in range(n_epochs):
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                loss = F.cross_entropy(model(X), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._trained[member_idx] = True
        model.eval()

    def fit_all(self, dataloader, n_epochs: int = 10, lr: float = 1e-3):
        for i in range(len(self.members)):
            print(f"[DeepEnsemble] Training member {i+1}/{len(self.members)}")
            self.fit_member(i, dataloader, n_epochs, lr)

    def predict_uncertainty(self, X: torch.Tensor) -> np.ndarray:
        X     = X.to(self.device)
        preds = []
        for m in self.members:
            m.eval()
            with torch.no_grad():
                preds.append(F.softmax(m(X), dim=-1).cpu().numpy())
        preds  = np.stack(preds)     
        return preds.var(axis=0).mean(axis=1)  

    def predict_uncertainty_trajectory(
        self, X_seq: List[torch.Tensor]
    ) -> np.ndarray:
        return np.stack([self.predict_uncertainty(X) for X in X_seq])


class ODINPredictor:

    def __init__(
        self,
        model:       nn.Module,
        temperature: float = 1000.0,
        epsilon:     float = 0.0014,
        device:      str   = "cpu",
    ):
        self.model       = model.to(device)
        self.temperature = temperature
        self.epsilon     = epsilon
        self.device      = device

    def predict_uncertainty(self, X: torch.Tensor) -> np.ndarray:
        X = X.to(self.device).requires_grad_(True)

        logits  = self.model(X)
        scaled  = logits / self.temperature
        labels  = scaled.argmax(dim=1)
        loss    = F.cross_entropy(scaled, labels)
        loss.backward()
        with torch.no_grad():
            X_hat = X - self.epsilon * X.grad.sign()
            logits_hat = self.model(X_hat)
            scores = F.softmax(logits_hat / self.temperature, dim=-1).max(dim=1).values

        return (1.0 - scores.cpu().numpy())

    def predict_uncertainty_trajectory(
        self, X_seq: List[torch.Tensor]
    ) -> np.ndarray:
        return np.stack([self.predict_uncertainty(X) for X in X_seq])


class AutoencoderBaseline(nn.Module):

    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32,  4, 2, 1), nn.ReLU(),
            nn.Conv2d(32,          64,  4, 2, 1), nn.ReLU(),
            nn.Conv2d(64,         128,  4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
        )
        self.decoder_fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z    = self.encoder(x)
        h    = self.decoder_fc(z).reshape(-1, 128, 4, 4)
        return self.decoder_conv(h)

    def reconstruction_error(self, x: torch.Tensor) -> np.ndarray:
        """Returns MSE reconstruction error per sample (N,)."""
        with torch.no_grad():
            recon = self.forward(x)
            err   = F.mse_loss(recon, x, reduction="none")
            return err.mean(dim=(1, 2, 3)).cpu().numpy()


class AutoencoderPredictor:
    def __init__(self, model: AutoencoderBaseline, device: str = "cpu"):
        self.model  = model.to(device)
        self.device = device

    def fit(self, dataloader, n_epochs: int = 20, lr: float = 1e-3):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for epoch in range(n_epochs):
            total_loss = 0.0
            for X, _ in dataloader:
                X    = X.to(self.device)
                recon = self.model(X)
                loss  = F.mse_loss(recon, X)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            print(f"  [AE] Epoch {epoch+1:2d}/{n_epochs}  loss={total_loss:.4f}")
        self.model.eval()

    def predict_uncertainty(self, X: torch.Tensor) -> np.ndarray:
        return self.model.reconstruction_error(X.to(self.device))

    def predict_uncertainty_trajectory(
        self, X_seq: List[torch.Tensor]
    ) -> np.ndarray:
        return np.stack([self.predict_uncertainty(X) for X in X_seq])
