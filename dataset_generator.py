import os
import hashlib
import urllib.request
import tarfile
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as T
from sklearn.preprocessing import StandardScaler


CIFAR10C_URL = (
    "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
)
CIFAR10C_SHA256 = "56de9a84c9f46c74696dde898cfcf2f1"   

CORRUPTION_TYPES = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

SEVERITY_LEVELS = [1, 2, 3, 4, 5]


def _download_cifar10c(data_dir: str = "./data/cifar10c") -> str:
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "CIFAR-10-C.tar")
   
    extracted_labels = os.path.join(data_dir, "CIFAR-10-C", "labels.npy")
    if not os.path.exists(extracted_labels):
        if not os.path.exists(tar_path):
            print(f"[dataset] Downloading CIFAR-10-C → {tar_path}")
            urllib.request.urlretrieve(CIFAR10C_URL, tar_path)
        print("[dataset] Extracting …")
        with tarfile.open(tar_path) as tf:
            tf.extractall(data_dir)
        print("[dataset] Done.")
    return data_dir


def load_cifar10c(
    corruption: str = "gaussian_noise",
    severity: int = 3,
    data_dir: str = "./data/cifar10c",
    n_samples: int = 10_000,
    as_tensor: bool = True,
):

    assert corruption in CORRUPTION_TYPES, f"Unknown corruption: {corruption}"
    assert 1 <= severity <= 5, "Severity must be 1-5"

    _download_cifar10c(data_dir)
    subdir = os.path.join(data_dir, "CIFAR-10-C")

    images = np.load(os.path.join(subdir, f"{corruption}.npy"))  
    labels = np.load(os.path.join(subdir, "labels.npy"))         

    start = (severity - 1) * 10_000
    end   = start + min(n_samples, 10_000)
    images = images[start:end]
    labels = labels[start:end]

    images = images.astype(np.float32) / 255.0
    images = np.transpose(images, (0, 3, 1, 2))

    if as_tensor:
        return torch.from_numpy(images), torch.from_numpy(labels.astype(np.int64))
    return images, labels


def load_cifar10_clean(
    data_dir: str = "./data",
    train: bool = False,
    n_samples: int = 10_000,
):

    transform = T.Compose([T.ToTensor()])
    split = "train" if train else None
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=transform
    )
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    subset  = torch.utils.data.Subset(dataset, indices)
    loader  = DataLoader(subset, batch_size=len(indices))
    images, labels = next(iter(loader))
    return images, labels


def cifar10c_trajectory_dataset(
    model,
    data_dir: str = "./data",
    corruption: str = "gaussian_noise",
    n_per_severity: int = 500,
    device: str = "cpu",
):

    model.eval()
    results = []
    for sev in SEVERITY_LEVELS:
        imgs, lbls = load_cifar10c(corruption, sev, data_dir, n_per_severity)
        imgs = imgs.to(device)
        with torch.no_grad():
            logits  = model(imgs)
            softmax = torch.softmax(logits, dim=-1).cpu().numpy()
        results.append({
            "severity": sev,
            "images":   imgs.cpu().numpy(),
            "labels":   lbls.numpy(),
            "logits":   logits.cpu().numpy(),
            "softmax":  softmax,
        })
    return results



def load_svhn(
    data_dir: str = "./data",
    split: str = "test",
    n_samples: int = 10_000,
):

    transform = T.Compose([T.ToTensor()])
    dataset   = torchvision.datasets.SVHN(
        root=data_dir, split=split, download=True, transform=transform
    )
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    subset  = torch.utils.data.Subset(dataset, indices)
    loader  = DataLoader(subset, batch_size=len(indices))
    images, labels = next(iter(loader))
    return images, labels


OPENML_TASK_IDS = {
    "credit-g":       31,
    "adult":          7592,
    "phoneme":        9952,
    "higgs-small":    146606,
    "blood-transfus": 10101,
    "steel-plates":   146212,
    "australian":     146818,
    "bank-marketing": 14965,
}


def load_openml_dataset(
    name: str = "credit-g",
    test_size: float = 0.2,
    random_state: int = 42,
    cache_dir: str = "./data/openml",
):

    try:
        import openml
        from sklearn.model_selection import train_test_split

        os.makedirs(cache_dir, exist_ok=True)
        openml.config.cache_directory = os.path.abspath(cache_dir)

        task_id = OPENML_TASK_IDS[name]
        task    = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

        # pandas → numpy, handle categoricals
        if hasattr(X, "to_numpy"):
            # fill NaN with column median
            for col in X.columns:
                if X[col].dtype.kind in "fc":
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(X[col].mode()[0])
                    X[col] = X[col].astype("category").cat.codes
            X = X.to_numpy(dtype=np.float32)
        else:
            X = np.array(X, dtype=np.float32)

        if hasattr(y, "to_numpy"):
            y = y.astype("category").cat.codes.to_numpy(dtype=np.int64)
        else:
            y = np.array(y, dtype=np.int64)

        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    except ImportError:
        raise ImportError("Install openml:  pip install openml")


def perturb_tabular(X: np.ndarray, noise_std: float = 0.5, seed: int = 42):
    """Add Gaussian noise to tabular features (simulates distribution shift)."""
    rng     = np.random.default_rng(seed)
    X_noisy = X + rng.normal(0, noise_std, X.shape).astype(np.float32)
    return X_noisy

def make_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int = 256, shuffle: bool = False):
    tx = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
    ty = torch.from_numpy(y) if isinstance(y, np.ndarray) else y
    ds = TensorDataset(tx, ty)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)