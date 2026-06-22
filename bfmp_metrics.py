from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def _norm(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


def _rolling_slope(traj: np.ndarray, w: int = 2) -> np.ndarray:
    T, N  = traj.shape
    slope = np.zeros_like(traj)
    for t in range(w, T):
        slope[t] = (traj[t] - traj[t - w]) / w
    return slope



def entropy_spike_score(softmax: np.ndarray) -> np.ndarray:
    C   = softmax.shape[1]
    eps = 1e-10
    p   = np.clip(softmax, eps, 1.0)
    H   = -(p * np.log(p)).sum(axis=1)
    return H / np.log(C)


def ess_trajectory(softmax_seq: List[np.ndarray]) -> np.ndarray:
    return np.stack([entropy_spike_score(sm) for sm in softmax_seq])

def activation_drift_distance(
    acts_clean:     Dict[str, np.ndarray],
    acts_perturbed: Dict[str, np.ndarray],
    layer_weights:  Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Weighted L2 drift: ADD = Σ_l w_l ||φ_l(x) - φ_l(x_0)||_2"""
    layers = list(acts_clean.keys())
    w = layer_weights or {l: 1.0 / len(layers) for l in layers}
    add = np.zeros(next(iter(acts_clean.values())).shape[0])
    for l in layers:
        diff = acts_perturbed[l] - acts_clean[l]
        add += w.get(l, 0.0) * np.linalg.norm(diff, axis=1)
    return add


def attention_collapse_detector_vit(
    attn_entropy_seq: List[np.ndarray],
) -> np.ndarray:
    """ACD_t = mean_blocks(H_max - H_t). Returns (T, N)."""
    H_max = np.stack(attn_entropy_seq).max(axis=0)
    return np.stack([
        np.clip(H_max - H_t, 0, None).mean(axis=1)
        for H_t in attn_entropy_seq
    ])



def output_distribution_dispersion(softmax: np.ndarray) -> np.ndarray:
    """ODD = 1 - (p_max - p_mean)  ∈ [0,1]. High = uncertain."""
    return 1.0 - (softmax.max(axis=1) - softmax.mean(axis=1))


def odd_trajectory(softmax_seq: List[np.ndarray]) -> np.ndarray:
    return np.stack([output_distribution_dispersion(sm) for sm in softmax_seq])



def compound_failure_signal_naive(
    ess_traj: np.ndarray,
    add_traj: np.ndarray,
) -> np.ndarray:
    """Original naive CFS = norm(ESS) * norm(ADD). Kept for ablation."""
    return _norm(ess_traj) * _norm(add_traj)


class LearnedCFS:

    def __init__(self, k: int = 1, window: int = 2, C: float = 1.0):
        self.k      = k
        self.window = window
        self.pipe   = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(C=C, max_iter=1000,
                                          class_weight="balanced",
                                          random_state=42)),
        ])
        self._fitted = False

    def _build_features(
        self,
        ess_traj: np.ndarray,   
        add_traj: np.ndarray,
        odd_traj: np.ndarray,
    ) -> np.ndarray:

        T, N = ess_traj.shape
        w    = self.window

        ess_s = _rolling_slope(ess_traj, w)
        add_s = _rolling_slope(add_traj, w)
        odd_s = _rolling_slope(odd_traj, w)

        ess_n = _norm(ess_traj)
        add_n = _norm(add_traj)
        odd_n = _norm(odd_traj)


        feats = np.stack([
            ess_n, add_n, odd_n,
            ess_s, add_s, odd_s,
            ess_n * add_n,   
            ess_n * odd_n,   
            add_n * odd_n,   
        ], axis=-1)          

        return feats.reshape(T * N, 9).astype(np.float32)

    def fit(
        self,
        ess_traj: np.ndarray,
        add_traj: np.ndarray,
        odd_traj: np.ndarray,
        horizon:  np.ndarray,   # (T, N) steps-to-failure
    ) -> "LearnedCFS":
        X = self._build_features(ess_traj, add_traj, odd_traj)
        y = (horizon.flatten() <= self.k).astype(int)

        # Need both classes present
        if y.sum() == 0 or y.sum() == len(y):
            print("[LCFS] Warning: degenerate labels — falling back to ADD signal")
            self._fitted    = False
            self._add_traj  = add_traj
            return self

        self.pipe.fit(X, y)
        self._fitted = True

        coef = self.pipe.named_steps["clf"].coef_[0]
        names = ["ESS","ADD","ODD","dESS","dADD","dODD",
                 "ESS×ADD","ESS×ODD","ADD×ODD"]
        top = sorted(zip(names, coef), key=lambda x: abs(x[1]), reverse=True)
        print("[LCFS] Feature importances:")
        for name, c in top:
            print(f"       {name:<12s}  {c:+.4f}")
        return self

    def predict_proba(
        self,
        ess_traj: np.ndarray,
        add_traj: np.ndarray,
        odd_traj: np.ndarray,
    ) -> np.ndarray:
        if not self._fitted:
            return _norm(add_traj)   # fallback

        T, N = ess_traj.shape
        X    = self._build_features(ess_traj, add_traj, odd_traj)
        prob = self.pipe.predict_proba(X)[:, 1]   # P(failure)
        return prob.reshape(T, N)

    def ablation_scores(
        self,
        ess_traj: np.ndarray,
        add_traj: np.ndarray,
        odd_traj: np.ndarray,
        horizon:  np.ndarray,
    ) -> Dict[str, float]:
        from sklearn.metrics import roc_auc_score

        y = (horizon.flatten() <= self.k).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            return {}

        def auroc(score_2d):
            s = score_2d.flatten()
            try:
                return float(roc_auc_score(y, s))
            except Exception:
                return float("nan")

        ess_n = _norm(ess_traj)
        add_n = _norm(add_traj)
        odd_n = _norm(odd_traj)

        results = {
            "ESS only":  auroc(ess_n),
            "ADD only":  auroc(add_n),
            "ODD only":  auroc(odd_n),
            "ESS+ADD (naive)": auroc(ess_n * add_n),
            "Full LCFS": auroc(self.predict_proba(ess_traj, add_traj, odd_traj)),
        }

        print("\n[LCFS Ablation]")
        for k, v in results.items():
            print(f"  {k:<22s}: {v:.4f}")
        return results



def compound_failure_signal(
    ess_traj: np.ndarray,
    add_traj: np.ndarray,
) -> np.ndarray:
    """Alias for naive CFS — kept for backward compatibility."""
    return compound_failure_signal_naive(ess_traj, add_traj)


class FailureHorizonRegressor:

    def __init__(self, alpha: float = 1.0, window: int = 2):
        self.alpha   = alpha
        self.window  = window
        self.scaler  = StandardScaler()
        self.model   = Ridge(alpha=alpha, fit_intercept=True)
        self._fitted = False

    def _build_features(self, ess_traj, add_traj, odd_traj) -> np.ndarray:
        T, N  = ess_traj.shape
        w     = self.window
        ess_s = _rolling_slope(ess_traj, w)
        add_s = _rolling_slope(add_traj, w)
        odd_s = _rolling_slope(odd_traj, w)
        ess_n = _norm(ess_traj)
        add_n = _norm(add_traj)
        odd_n = _norm(odd_traj)
        feats = np.stack([
            ess_n, add_n, odd_n,
            ess_s, add_s, odd_s,
            ess_n * add_n, ess_n * odd_n, add_n * odd_n,
        ], axis=-1)
        return feats.reshape(T * N, 9).astype(np.float32)

    def fit(self, ess_traj, add_traj, odd_traj, horizon):
        X = self._build_features(ess_traj, add_traj, odd_traj)
        y = horizon.flatten().astype(np.float32)
        self.model.fit(self.scaler.fit_transform(X), y)
        self._fitted = True

    def predict(self, ess_traj, add_traj, odd_traj) -> np.ndarray:
        assert self._fitted
        X = self._build_features(ess_traj, add_traj, odd_traj)
        p = np.clip(self.model.predict(self.scaler.transform(X)), 0, None)
        return p.reshape(ess_traj.shape)

    def score(self, ess_traj, add_traj, odd_traj, horizon) -> Dict[str, float]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        pred = self.predict(ess_traj, add_traj, odd_traj).flatten()
        true = horizon.flatten()
        return {
            "MAE":  float(mean_absolute_error(true, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
            "R2":   float(r2_score(true, pred)),
        }


def compute_failure_horizon(
    softmax_seq: List[np.ndarray],
    labels:      np.ndarray,
) -> np.ndarray:

    T          = len(softmax_seq)
    N          = labels.shape[0]
    pred_class = np.stack([sm.argmax(axis=1) for sm in softmax_seq])
    correct    = (pred_class == labels[None, :]).astype(int)

    horizon = np.zeros((T, N), dtype=np.int32)
    for t in range(T):
        for n in range(N):
            future = correct[t:, n]
            fails  = np.where(future == 0)[0]
            horizon[t, n] = int(fails[0]) if len(fails) else (T - t)
    return horizon