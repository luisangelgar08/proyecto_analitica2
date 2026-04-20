from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# Config
# =========================================================
PROJECT_ROOT = Path("/home/uaodeepia/luis/proyecto_analitica2")
EMB_DIR = PROJECT_ROOT / "data" / "working" / "router" / "embeddings" / "vit_tiny_patch16_224_balanced"

OUT_DIR = PROJECT_ROOT / "data" / "working" / "router" / "aux_training" / "vit_tiny_patch16_224_balanced"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALPHA = 0.02              # luego puedes probar 0.01, 0.02, 0.05
N_EXPERTS = 5
EPOCHS = 80
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 20


# =========================================================
# Seed
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)


# =========================================================
# Model
# =========================================================
class LinearRouter(nn.Module):
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return self.fc(x)


# =========================================================
# Load data
# =========================================================
def load_arrays():
    Z_train = np.load(EMB_DIR / "Z_train.npy")
    y_train = np.load(EMB_DIR / "y_train_expert.npy")
    Z_val = np.load(EMB_DIR / "Z_val.npy")
    y_val = np.load(EMB_DIR / "y_val_expert.npy")
    Z_test = np.load(EMB_DIR / "Z_test.npy")
    y_test = np.load(EMB_DIR / "y_test_expert.npy")

    meta_train = pd.read_csv(EMB_DIR / "meta_train.csv")
    meta_val = pd.read_csv(EMB_DIR / "meta_val.csv")
    meta_test = pd.read_csv(EMB_DIR / "meta_test.csv")

    return Z_train, y_train, Z_val, y_val, Z_test, y_test, meta_train, meta_val, meta_test


def build_label_map(meta_train):
    tmp = meta_train[["expert_id", "expert_name"]].drop_duplicates().sort_values("expert_id")
    return dict(zip(tmp["expert_id"], tmp["expert_name"]))


# =========================================================
# Auxiliary loss
# =========================================================
def compute_aux_loss_from_probs(probs: torch.Tensor, n_experts: int):
    """
    probs: [B, N]
    f_i: fracción dura usando argmax
    P_i: probabilidad media suave
    """
    hard_assign = torch.argmax(probs, dim=1)               # [B]
    fi = torch.bincount(hard_assign, minlength=n_experts).float()
    fi = fi / fi.sum().clamp_min(1.0)                      # [N]
    Pi = probs.mean(dim=0)                                 # [N]
    aux = n_experts * torch.sum(fi * Pi)
    return aux, fi, Pi


# =========================================================
# Epoch-level metrics
# =========================================================
@torch.no_grad()
def compute_epoch_balance(model, X_np, y_np, n_experts):
    X = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
    logits = model(X)
    probs = torch.softmax(logits, dim=1)

    pred = torch.argmax(probs, dim=1).cpu().numpy()
    acc = accuracy_score(y_np, pred)

    hard_assign = torch.argmax(probs, dim=1)
    fi = torch.bincount(hard_assign, minlength=n_experts).float()
    fi = fi / fi.sum().clamp_min(1.0)
    Pi = probs.mean(dim=0)

    fi_np = fi.cpu().numpy()
    Pi_np = Pi.cpu().numpy()

    ratio = float(fi_np.max() / fi_np.min()) if fi_np.min() > 0 else np.inf

    return {
        "acc": float(acc),
        "pred": pred,
        "fi": fi_np,
        "Pi": Pi_np,
        "ratio": ratio,
    }


# =========================================================
# Training
# =========================================================
def main():
    Z_train, y_train, Z_val, y_val, Z_test, y_test, meta_train, meta_val, meta_test = load_arrays()

    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train)
    Z_val_s = scaler.transform(Z_val)
    Z_test_s = scaler.transform(Z_test)

    id_to_name = build_label_map(meta_train)
    class_names = [id_to_name[i] for i in sorted(id_to_name.keys())]

    print("=" * 80)
    print("RESUMEN DE ENTRADA")
    print("=" * 80)
    print("Z_train:", Z_train_s.shape, "y_train:", y_train.shape)
    print("Z_val  :", Z_val_s.shape, "y_val  :", y_val.shape)
    print("Z_test :", Z_test_s.shape, "y_test :", y_test.shape)
    print("Classes:", class_names)
    print("DEVICE :", DEVICE)
    print("ALPHA  :", ALPHA)

    Xtr = torch.tensor(Z_train_s, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = LinearRouter(Z_train_s.shape[1], len(class_names)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_epoch = -1
    best_val_acc = -1.0
    patience_counter = 0

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch_losses = []
        batch_ce = []
        batch_aux = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            logits = model(xb)
            probs = torch.softmax(logits, dim=1)

            ce = criterion(logits, yb)
            aux, _, _ = compute_aux_loss_from_probs(probs, N_EXPERTS)
            loss = ce + ALPHA * aux

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            batch_ce.append(ce.item())
            batch_aux.append(aux.item())

        # métricas de época
        model.eval()
        train_stats = compute_epoch_balance(model, Z_train_s, y_train, N_EXPERTS)
        val_stats = compute_epoch_balance(model, Z_val_s, y_val, N_EXPERTS)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)),
            "train_ce": float(np.mean(batch_ce)),
            "train_aux": float(np.mean(batch_aux)),
            "train_acc": train_stats["acc"],
            "val_acc": val_stats["acc"],
            "train_ratio": train_stats["ratio"],
            "val_ratio": val_stats["ratio"],
        }

        for i, name in enumerate(class_names):
            row[f"train_f_{name}"] = float(train_stats["fi"][i])
            row[f"train_P_{name}"] = float(train_stats["Pi"][i])
            row[f"val_f_{name}"] = float(val_stats["fi"][i])
            row[f"val_P_{name}"] = float(val_stats["Pi"][i])

        history.append(row)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[epoch {epoch:03d}] "
                f"train_loss={row['train_loss']:.4f} | "
                f"train_acc={row['train_acc']:.4f} | "
                f"val_acc={row['val_acc']:.4f} | "
                f"train_ratio={row['train_ratio']:.4f} | "
                f"val_ratio={row['val_ratio']:.4f}"
            )

        # early stopping por val_acc
        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping en epoch {epoch}. Mejor epoch={best_epoch}, best_val_acc={best_val_acc:.4f}")
            break

    # cargar mejor modelo por val_acc
    model.load_state_dict(best_state)
    model.eval()

    train_stats = compute_epoch_balance(model, Z_train_s, y_train, N_EXPERTS)
    val_stats = compute_epoch_balance(model, Z_val_s, y_val, N_EXPERTS)
    test_stats = compute_epoch_balance(model, Z_test_s, y_test, N_EXPERTS)

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUT_DIR / f"router_aux_history_alpha_{str(ALPHA).replace('.', 'p')}.csv", index=False)

    # guardar scaler
    np.save(OUT_DIR / "scaler_mean.npy", scaler.mean_)
    np.save(OUT_DIR / "scaler_scale.npy", scaler.scale_)

    # guardar expert id map
    expert_map = meta_train[["expert_id", "expert_name"]].drop_duplicates().sort_values("expert_id")
    expert_map.to_csv(OUT_DIR / "expert_id_map.csv", index=False)

    # guardar checkpoint
    ckpt = {
        "model_state_dict": model.state_dict(),
        "input_dim": int(Z_train_s.shape[1]),
        "n_classes": int(len(class_names)),
        "alpha": float(ALPHA),
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "scaler_mean_path": str(OUT_DIR / "scaler_mean.npy"),
        "scaler_scale_path": str(OUT_DIR / "scaler_scale.npy"),
    }
    ckpt_path = OUT_DIR / f"linear_router_aux_alpha_{str(ALPHA).replace('.', 'p')}.pt"
    torch.save(ckpt, ckpt_path)

    summary = {
        "alpha": float(ALPHA),
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "train_acc": float(train_stats["acc"]),
        "val_acc": float(val_stats["acc"]),
        "test_acc": float(test_stats["acc"]),
        "train_ratio": float(train_stats["ratio"]),
        "val_ratio": float(val_stats["ratio"]),
        "test_ratio": float(test_stats["ratio"]),
        "train_fi": {class_names[i]: float(train_stats["fi"][i]) for i in range(len(class_names))},
        "val_fi": {class_names[i]: float(val_stats["fi"][i]) for i in range(len(class_names))},
        "test_fi": {class_names[i]: float(test_stats["fi"][i]) for i in range(len(class_names))},
        "checkpoint_path": str(ckpt_path),
        "history_csv": str(OUT_DIR / f"router_aux_history_alpha_{str(ALPHA).replace('.', 'p')}.csv"),
    }

    with open(OUT_DIR / f"router_aux_summary_alpha_{str(ALPHA).replace('.', 'p')}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("RESUMEN FINAL AUX TRAINING")
    print("=" * 80)
    print(summary)


if __name__ == "__main__":
    main()