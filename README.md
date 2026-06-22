# Behavioural Failure Mode Prediction in Machine Learning Systems

**BFMP** is a trajectory-based framework that predicts ML model failure *before* it occurs, by monitoring the evolution of internal model signals across increasing perturbation severity.

## What this does

Most ML monitoring tools detect failure *after* the model has already started making wrong predictions. BFMP detects the warning signs *before* — by feeding a model progressively corrupted versions of an input and watching how its internal behaviour changes across that trajectory.

Think of it like checking a patient's temperature before they collapse, not after.

---

## Key results

| Method | AUROC ↑ | AUPR ↑ |
|---|---|---|
| **BFMP/LCFS (ours)** | **0.675 ± 0.09** | **0.954 ± 0.01** |
| MC Dropout | 0.559 ± 0.06 | 0.946 ± 0.01 |
| Deep Ensembles | 0.551 ± 0.07 | 0.943 ± 0.01 |
| ODIN | 0.556 ± 0.05 | 0.939 ± 0.01 |
| Autoencoder | 0.538 ± 0.05 | 0.944 ± 0.01 |

*CNN on CIFAR-10-C, 5 seeds, all differences p < 0.05 (Wilcoxon)*

- **ADD alone** achieves AUROC 0.815 ± 0.038 (7 corruptions × 5 seeds)
- **Pretrained DeiT-Small**: ADD reaches AUROC **0.977 ± 0.006**
- **FHR** predicts failure horizon with MAE = 0.55 steps

---

## The four main contributions

1. **BFMP** — trajectory-based monitoring framework evaluated on CIFAR-10-C and OpenML-CC18
2. **LCFS** (Learned Compound Failure Signal) — logistic regressor on 9 features that outperforms naive ESS×ADD multiplication (0.734 vs 0.680)
3. **FHR** (Failure Horizon Regressor) — continuous regression of steps-to-failure, not just binary yes/no
4. **Architecture-gated signals** — ACD for ViT (real attention), ODD for MLP/CNN (softmax spread); using softmax as an attention proxy in an MLP is a category error we explicitly avoid

---

## Repository structure

```
.
├── dataset_generator.py      # CIFAR-10-C loader + OpenML tabular datasets
├── model_simulator.py        # CNN, MLP, ViT (pretrained DeiT-Small with attention hooks)
├── bfmp_metrics.py           # ESS, ADD, ODD, ACD, LCFS, FHR implementations
├── bfmp_v3.py                # BFMPPipeline class (calibrate → monitor → predict)
├── baselines.py              # MC Dropout, Deep Ensembles, ODIN, Autoencoder
├── prior_works_baselines.py  # XGBoost, LSTM, ZScore, prior work comparisons
├── run_experiment_v3.py      # Main experiment runner (5 seeds, all baselines)
├── run_all_corruptions.py    # All 15 CIFAR-10-C corruption types × 5 seeds
├── pretrained_vit_acd.py     # ImageNet-pretrained ViT ACD vs ODD validation
├── analyze_v3.py             # Figure generation from real results CSVs
├── figures/                  # Generated paper figures (PDF)
├── results/                  # Experiment CSVs, LaTeX tables, significance reports
└── data/                     # NOT included — download separately (see below)
```

---

## Setup

```bash
git clone https:github.com/lohaarja/Behavioural-Failure-Mode-Prediction-in-Machine-Learning-Systems
cd bfmp
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
scikit-learn>=1.3.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
pandas>=2.0.0
tqdm>=4.65.0
openml>=0.14.0
requests>=2.28.0
```

---

## Data

**CIFAR-10-C** is not included in this repository (600MB).

Download from Zenodo:
```bash
mkdir -p data/cifar10c
wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar -P data/cifar10c/
cd data/cifar10c && tar -xf CIFAR-10-C.tar
```

Or the code will attempt to download it automatically on first run.

**CIFAR-10** (clean): downloaded automatically via torchvision on first run.

**OpenML credit-g**: downloaded automatically via `pip install openml`.

---

## Running experiments

### Quick start — CNN on Gaussian noise, 3 seeds

```bash
python run_experiment_v3.py --arch cnn --dataset cifar10c --seeds 3 --device cuda
```

### Full experiment — all 15 corruptions, 5 seeds (≈90 minutes on T4)

```bash
python run_all_corruptions.py --arch cnn --device cuda --seeds 5
```

### Pretrained ViT ACD validation (≈45 minutes on T4)

```bash
python pretrained_vit_acd.py --device cuda --seeds 3 --epochs 20
```

## Reproducing paper numbers

All experiments use explicit random seeds passed to both `torch.manual_seed` and `numpy.random.seed`. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic CUDA behaviour:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python run_all_corruptions.py --arch cnn --seeds 5 --device cuda
python generate_final_paper.py --arch cnn
```

The script `generate_final_paper.py` reads actual CSVs and outputs:
- `results/cnn_table2_final.tex` — main comparison table
- `results/cnn_table4_ablation_final.tex` — multi-seed ablation
- `results/cnn_paper_patch.txt` — exact numbers to update in the paper

---

## Signal definitions

| Signal | Formula | Architecture |
|---|---|---|
| ESS | H(p) / log C | All |
| ADD | Σ‖φ(xₜ) − φ(x₀)‖₂ | All |
| ODD | 1 − (max p − p̄) | MLP, CNN |
| ACD | mean[H_max − H(aₜ)]₊ | ViT only |
| LCFS | σ(w⊤fₜ + b) | All |
| FHR | max(0, v⊤fₜ + c) | All |

**Important:** ODD is *not* attention. Softmax in an MLP is a class probability vector, not a spatial attention map. ACD is only computed for ViT where true Q@K^T/sqrt(d) self-attention exists.

---
