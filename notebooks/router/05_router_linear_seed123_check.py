from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / "data").exists() and (candidate / "notebooks").exists():
            return candidate
    raise FileNotFoundError("No pude encontrar la raíz del proyecto.")


PROJECT_ROOT = find_project_root(Path.cwd())
WORKING_DIR = PROJECT_ROOT / "data" / "working"

EMB_DIR = WORKING_DIR / "router" / "embeddings" / "vit_tiny_patch16_224_balanced_seed123"
OUT_DIR = WORKING_DIR / "router" / "ablation" / "vit_tiny_patch16_224_balanced_seed123_linear_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LINEAR_EPOCHS = 200
LINEAR_BATCH_SIZE = 128
LINEAR_LR = 1e-3
LINEAR_WEIGHT_DECAY = 1e-4
LINEAR_PATIENCE = 20


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


def load_arrays():
    Z_train = np.load(EMB_DIR / "Z_train.npy")
    y_train = np.load(EMB_DIR / "y_train_expert.npy")
    Z_val = np.load(EMB_DIR / "Z_val.npy")
    y_val = np.load(EMB_DIR / "y_val_expert.npy")
    Z_test = np.load(EMB_DIR / "Z_test.npy")
    y_test = np.load(EMB_DIR / "y_test_expert.npy")
    meta_train = pd.read_csv(EMB_DIR / "meta_train.csv")
    return Z_train, y_train, Z_val, y_val, Z_test, y_test, meta_train


def build_label_map(meta_train):
    tmp = meta_train[["expert_id", "expert_name"]].drop_duplicates().sort_values("expert_id")
    return dict(zip(tmp["expert_id"], tmp["expert_name"]))


class LinearRouter(nn.Module):
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return self.fc(x)


def train_linear_router(Z_train, y_train, Z_val, y_val, n_classes):
    Xtr = torch.tensor(Z_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xva = torch.tensor(Z_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=LINEAR_BATCH_SIZE,
        shuffle=True,
    )

    model = LinearRouter(Z_train.shape[1], n_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LINEAR_LR, weight_decay=LINEAR_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, LINEAR_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(Xva.to(DEVICE))
            pred_val = torch.softmax(logits_val, dim=1).argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, pred_val)

        if epoch == 1 or epoch % 10 == 0:
            print(f"[Linear-check] epoch={epoch:03d} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= LINEAR_PATIENCE:
            print(f"[Linear-check] Early stopping. Mejor epoch={best_epoch}, val_acc={best_val_acc:.4f}")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val_acc, best_epoch


@torch.no_grad()
def predict(model, X):
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    logits = model(X_t)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred = probs.argmax(axis=1)
    return pred, probs


def main():
    Z_train, y_train, Z_val, y_val, Z_test, y_test, meta_train = load_arrays()

    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train)
    Z_val_s = scaler.transform(Z_val)
    Z_test_s = scaler.transform(Z_test)

    id_to_name = build_label_map(meta_train)
    class_names = [id_to_name[i] for i in sorted(id_to_name.keys())]

    model, best_val_acc, best_epoch = train_linear_router(
        Z_train_s, y_train, Z_val_s, y_val, n_classes=len(class_names)
    )

    pred_val, _ = predict(model, Z_val_s)
    pred_test, _ = predict(model, Z_test_s)

    val_acc = accuracy_score(y_val, pred_val)
    test_acc = accuracy_score(y_test, pred_test)

    results = {
        "model_name": "linear_seed123_check",
        "val_acc": float(val_acc),
        "test_acc": float(test_acc),
        "best_epoch": int(best_epoch),
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }

    with open(OUT_DIR / "linear_seed123_check.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nRESULTADO FINAL")
    print(results)


if __name__ == "__main__":
    main()