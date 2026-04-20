from pathlib import Path
import json
import time
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =========================================================
# 0) Verificación de raíz del proyecto
# =========================================================
def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / "data").exists() and (candidate / "notebooks").exists():
            return candidate
    raise FileNotFoundError("No pude encontrar la raíz del proyecto.")


PROJECT_ROOT = find_project_root(Path.cwd())
DATA_DIR = PROJECT_ROOT / "data"
WORKING_DIR = DATA_DIR / "working"

EMB_DIR = WORKING_DIR / "router" / "embeddings" / "vit_tiny_patch16_224_balanced"
OUT_DIR = WORKING_DIR / "router" / "ablation" / "vit_tiny_patch16_224_balanced"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("EMB_DIR:", EMB_DIR)
print("OUT_DIR:", OUT_DIR)


# =========================================================
# 1) Configuración
# =========================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Linear router
LINEAR_EPOCHS = 200
LINEAR_BATCH_SIZE = 128
LINEAR_LR = 1e-3
LINEAR_WEIGHT_DECAY = 1e-4
LINEAR_PATIENCE = 20

# GMM
GMM_N_COMPONENTS = 5
GMM_COVARIANCE_TYPE = "full"   # intentará full; si falla, cae a diag
GMM_MAX_ITER = 300
GMM_REG_COVAR = 1e-5

# kNN
KNN_K = 5
KNN_USE_PCA = False
KNN_PCA_DIM = 32

# Latencia
LATENCY_REPEATS = 5


# =========================================================
# 2) Semillas
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# =========================================================
# 3) Utilidades generales
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


def build_label_map(meta_train, y_train):
    tmp = meta_train[["expert_id", "expert_name"]].drop_duplicates().sort_values("expert_id")
    id_to_name = dict(zip(tmp["expert_id"], tmp["expert_name"]))
    class_ids = sorted(np.unique(y_train).tolist())
    class_names = [id_to_name[i] for i in class_ids]
    return class_ids, class_names, id_to_name


def save_confusion_and_report(y_true, y_pred, class_names, prefix):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(OUT_DIR / f"{prefix}_confusion_matrix.csv")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).T
    report_df.to_csv(OUT_DIR / f"{prefix}_classification_report.csv")

    return cm_df, report_df


def measure_latency_per_sample(predict_fn, X, repeats=5):
    # warmup
    _ = predict_fn(X)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = predict_fn(X)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_time = float(np.mean(times))
    ms_per_sample = (mean_time / len(X)) * 1000.0
    return ms_per_sample


def save_predictions(meta_df, y_true, y_pred, probs, prefix, class_names):
    out = meta_df.copy()
    out["y_true"] = y_true
    out["y_pred"] = y_pred

    if probs is not None:
        for i, name in enumerate(class_names):
            out[f"prob_{name}"] = probs[:, i]

    out.to_csv(OUT_DIR / f"{prefix}_predictions.csv", index=False)


# =========================================================
# 4) Preprocesado común de embeddings
# =========================================================
def prepare_embeddings(Z_train, Z_val, Z_test):
    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train)
    Z_val_s = scaler.transform(Z_val)
    Z_test_s = scaler.transform(Z_test)

    np.save(OUT_DIR / "Z_train_scaled.npy", Z_train_s)
    np.save(OUT_DIR / "Z_val_scaled.npy", Z_val_s)
    np.save(OUT_DIR / "Z_test_scaled.npy", Z_test_s)

    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with open(OUT_DIR / "scaler_params.json", "w", encoding="utf-8") as f:
        json.dump(scaler_params, f)

    return Z_train_s, Z_val_s, Z_test_s


# =========================================================
# 5) Router A: Linear + Softmax
# =========================================================
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
    yva = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=LINEAR_BATCH_SIZE,
        shuffle=True,
    )

    model = LinearRouter(d_model=Z_train.shape[1], n_classes=n_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LINEAR_LR,
        weight_decay=LINEAR_WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(1, LINEAR_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            logits_val = model(Xva.to(DEVICE))
            probs_val = torch.softmax(logits_val, dim=1).cpu().numpy()
            pred_val = probs_val.argmax(axis=1)
            val_acc = accuracy_score(y_val, pred_val)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
        })

        if epoch == 1 or epoch % 10 == 0:
            print(f"[Linear] epoch={epoch:03d} train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= LINEAR_PATIENCE:
            print(f"[Linear] Early stopping en epoch {epoch}. Mejor epoch={best_epoch}, best_val_acc={best_val_acc:.4f}")
            break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(OUT_DIR / "linear_history.csv", index=False)

    model.load_state_dict(best_state)
    model.eval()

    return model, best_val_acc, best_epoch


@torch.no_grad()
def linear_predict(model, X):
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    logits = model(X_t)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred = probs.argmax(axis=1)
    return pred, probs


# =========================================================
# 6) Router B: GMM + mapeo componente->clase
# =========================================================
def build_component_to_class_map(train_components, y_train, n_classes):
    contingency = np.zeros((n_classes, n_classes), dtype=np.int64)
    for comp in range(n_classes):
        idx = train_components == comp
        labels = y_train[idx]
        for c in range(n_classes):
            contingency[comp, c] = np.sum(labels == c)

    # Queremos asignación comp->class maximizando coincidencias
    row_ind, col_ind = linear_sum_assignment(-contingency)
    comp_to_class = {int(comp): int(cls) for comp, cls in zip(row_ind, col_ind)}
    return comp_to_class, contingency


def remap_gmm_probs(probs, comp_to_class, n_classes):
    mapped = np.zeros((probs.shape[0], n_classes), dtype=np.float64)
    for comp_idx, class_idx in comp_to_class.items():
        mapped[:, class_idx] = probs[:, comp_idx]
    return mapped


def fit_gmm_router(Z_train, y_train, n_classes):
    cov_type = GMM_COVARIANCE_TYPE
    try:
        gmm = GaussianMixture(
            n_components=GMM_N_COMPONENTS,
            covariance_type=cov_type,
            max_iter=GMM_MAX_ITER,
            reg_covar=GMM_REG_COVAR,
            random_state=SEED,
        )
        gmm.fit(Z_train)
    except Exception as e:
        print(f"[GMM] Falló covariance_type='{cov_type}'. Cambio a 'diag'. Error: {e}")
        cov_type = "diag"
        gmm = GaussianMixture(
            n_components=GMM_N_COMPONENTS,
            covariance_type=cov_type,
            max_iter=GMM_MAX_ITER,
            reg_covar=GMM_REG_COVAR,
            random_state=SEED,
        )
        gmm.fit(Z_train)

    train_components = gmm.predict(Z_train)
    comp_to_class, contingency = build_component_to_class_map(train_components, y_train, n_classes)

    contingency_df = pd.DataFrame(contingency)
    contingency_df.to_csv(OUT_DIR / "gmm_component_class_contingency.csv", index=False)

    with open(OUT_DIR / "gmm_component_to_class_map.json", "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in comp_to_class.items()}, f, indent=2)

    return gmm, comp_to_class, cov_type


def gmm_predict(gmm, comp_to_class, X, n_classes):
    probs_raw = gmm.predict_proba(X)
    probs = remap_gmm_probs(probs_raw, comp_to_class, n_classes)
    pred = probs.argmax(axis=1)
    return pred, probs


# =========================================================
# 7) Router C: Gaussian Naive Bayes
# =========================================================
def fit_nb_router(Z_train, y_train):
    nb = GaussianNB()
    nb.fit(Z_train, y_train)
    return nb


def nb_predict(nb, X):
    probs = nb.predict_proba(X)
    pred = probs.argmax(axis=1)
    return pred, probs


# =========================================================
# 8) Router D: k-NN con FAISS (o fallback)
# =========================================================
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False
    warnings.warn("FAISS no está disponible. Se usará fallback con sklearn.")


class KNNRouter:
    def __init__(self, k=5, use_pca=False, pca_dim=32):
        self.k = k
        self.use_pca = use_pca
        self.pca_dim = pca_dim
        self.pca = None
        self.index = None
        self.y_train = None
        self.backend = None

    def fit(self, X_train, y_train):
        X = X_train.copy().astype(np.float32)
        self.y_train = np.asarray(y_train).copy()

        if self.use_pca:
            self.pca = PCA(n_components=self.pca_dim, random_state=SEED)
            X = self.pca.fit_transform(X).astype(np.float32)

        if FAISS_AVAILABLE:
            Xn = X.copy()
            faiss.normalize_L2(Xn)
            self.index = faiss.IndexFlatIP(Xn.shape[1])
            self.index.add(Xn)
            self.backend = "faiss"
        else:
            from sklearn.neighbors import NearestNeighbors
            self.index = NearestNeighbors(n_neighbors=self.k, metric="cosine")
            self.index.fit(X)
            self.backend = "sklearn"

    def predict(self, X_query, n_classes):
        X = X_query.copy().astype(np.float32)

        if self.pca is not None:
            X = self.pca.transform(X).astype(np.float32)

        if self.backend == "faiss":
            Xn = X.copy()
            faiss.normalize_L2(Xn)
            D, I = self.index.search(Xn, self.k)
        else:
            distances, indices = self.index.kneighbors(X, n_neighbors=self.k, return_distance=True)
            I = indices

        neighbor_labels = self.y_train[I]  # [B, k]

        probs = np.zeros((len(X), n_classes), dtype=np.float32)
        pred = np.zeros(len(X), dtype=np.int64)

        for i in range(len(X)):
            counts = np.bincount(neighbor_labels[i], minlength=n_classes).astype(np.float32)
            probs[i] = counts / counts.sum()
            pred[i] = probs[i].argmax()

        return pred, probs


# =========================================================
# 9) Evaluación unificada
# =========================================================
def evaluate_router(
    model_name,
    predict_fn,
    X_val,
    y_val,
    meta_val,
    X_test,
    y_test,
    meta_test,
    class_names,
):
    pred_val, probs_val = predict_fn(X_val)
    pred_test, probs_test = predict_fn(X_test)

    val_acc = accuracy_score(y_val, pred_val)
    test_acc = accuracy_score(y_test, pred_test)

    latency_ms_per_sample = measure_latency_per_sample(
        lambda X: predict_fn(X)[0],
        X_test,
        repeats=LATENCY_REPEATS,
    )

    save_confusion_and_report(y_val, pred_val, class_names, prefix=f"{model_name}_val")
    save_confusion_and_report(y_test, pred_test, class_names, prefix=f"{model_name}_test")

    save_predictions(meta_val, y_val, pred_val, probs_val, prefix=f"{model_name}_val", class_names=class_names)
    save_predictions(meta_test, y_test, pred_test, probs_test, prefix=f"{model_name}_test", class_names=class_names)

    return {
        "model_name": model_name,
        "val_acc": float(val_acc),
        "test_acc": float(test_acc),
        "latency_ms_per_sample": float(latency_ms_per_sample),
    }


# =========================================================
# 10) Main
# =========================================================
def main():
    Z_train, y_train, Z_val, y_val, Z_test, y_test, meta_train, meta_val, meta_test = load_arrays()

    class_ids, class_names, id_to_name = build_label_map(meta_train, y_train)
    n_classes = len(class_ids)

    print("\n" + "=" * 80)
    print("RESUMEN DE ENTRADA")
    print("=" * 80)
    print("Z_train:", Z_train.shape, "y_train:", y_train.shape)
    print("Z_val  :", Z_val.shape, "y_val  :", y_val.shape)
    print("Z_test :", Z_test.shape, "y_test :", y_test.shape)
    print("Classes:", class_names)
    print("DEVICE :", DEVICE)

    Z_train_s, Z_val_s, Z_test_s = prepare_embeddings(Z_train, Z_val, Z_test)

    results = []

    # -----------------------------------------------------
    # A) Linear + Softmax
    # -----------------------------------------------------
    print("\n" + "=" * 80)
    print("A) LINEAR + SOFTMAX")
    print("=" * 80)

    linear_model, best_val_acc_linear, best_epoch_linear = train_linear_router(
        Z_train_s, y_train, Z_val_s, y_val, n_classes=n_classes
    )

    torch.save(
        {
            "model_state_dict": linear_model.state_dict(),
            "input_dim": Z_train_s.shape[1],
            "n_classes": n_classes,
            "best_val_acc": best_val_acc_linear,
            "best_epoch": best_epoch_linear,
        },
        OUT_DIR / "linear_router.pt",
    )

    res_linear = evaluate_router(
        model_name="linear",
        predict_fn=lambda X: linear_predict(linear_model, X),
        X_val=Z_val_s,
        y_val=y_val,
        meta_val=meta_val,
        X_test=Z_test_s,
        y_test=y_test,
        meta_test=meta_test,
        class_names=class_names,
    )
    res_linear["best_epoch"] = int(best_epoch_linear)
    res_linear["n_params"] = int(sum(p.numel() for p in linear_model.parameters()))
    results.append(res_linear)
    print(res_linear)

    # -----------------------------------------------------
    # B) GMM
    # -----------------------------------------------------
    print("\n" + "=" * 80)
    print("B) GMM")
    print("=" * 80)

    gmm_model, gmm_comp_to_class, gmm_cov_type_used = fit_gmm_router(
        Z_train_s, y_train, n_classes=n_classes
    )

    res_gmm = evaluate_router(
        model_name="gmm",
        predict_fn=lambda X: gmm_predict(gmm_model, gmm_comp_to_class, X, n_classes),
        X_val=Z_val_s,
        y_val=y_val,
        meta_val=meta_val,
        X_test=Z_test_s,
        y_test=y_test,
        meta_test=meta_test,
        class_names=class_names,
    )
    res_gmm["covariance_type_used"] = gmm_cov_type_used
    results.append(res_gmm)
    print(res_gmm)

    # -----------------------------------------------------
    # C) Naive Bayes
    # -----------------------------------------------------
    print("\n" + "=" * 80)
    print("C) GAUSSIAN NAIVE BAYES")
    print("=" * 80)

    nb_model = fit_nb_router(Z_train_s, y_train)

    res_nb = evaluate_router(
        model_name="naive_bayes",
        predict_fn=lambda X: nb_predict(nb_model, X),
        X_val=Z_val_s,
        y_val=y_val,
        meta_val=meta_val,
        X_test=Z_test_s,
        y_test=y_test,
        meta_test=meta_test,
        class_names=class_names,
    )
    results.append(res_nb)
    print(res_nb)

    # -----------------------------------------------------
    # D) k-NN
    # -----------------------------------------------------
    print("\n" + "=" * 80)
    print("D) k-NN")
    print("=" * 80)

    knn_model = KNNRouter(k=KNN_K, use_pca=KNN_USE_PCA, pca_dim=KNN_PCA_DIM)
    knn_model.fit(Z_train_s, y_train)

    res_knn = evaluate_router(
        model_name="knn",
        predict_fn=lambda X: knn_model.predict(X, n_classes=n_classes),
        X_val=Z_val_s,
        y_val=y_val,
        meta_val=meta_val,
        X_test=Z_test_s,
        y_test=y_test,
        meta_test=meta_test,
        class_names=class_names,
    )
    res_knn["backend"] = knn_model.backend
    res_knn["k"] = KNN_K
    res_knn["use_pca"] = KNN_USE_PCA
    res_knn["pca_dim"] = KNN_PCA_DIM if KNN_USE_PCA else None
    results.append(res_knn)
    print(res_knn)

    # -----------------------------------------------------
    # Resumen final
    # -----------------------------------------------------
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["val_acc", "test_acc"], ascending=[False, False]).reset_index(drop=True)
    results_df.to_csv(OUT_DIR / "ablation_summary.csv", index=False)

    winner = results_df.iloc[0].to_dict()
    with open(OUT_DIR / "router_winner.json", "w", encoding="utf-8") as f:
        json.dump(winner, f, indent=2, ensure_ascii=False)

    with open(OUT_DIR / "ablation_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "device": DEVICE,
                "linear_epochs": LINEAR_EPOCHS,
                "linear_lr": LINEAR_LR,
                "linear_weight_decay": LINEAR_WEIGHT_DECAY,
                "linear_patience": LINEAR_PATIENCE,
                "gmm_n_components": GMM_N_COMPONENTS,
                "gmm_covariance_type": GMM_COVARIANCE_TYPE,
                "gmm_max_iter": GMM_MAX_ITER,
                "gmm_reg_covar": GMM_REG_COVAR,
                "knn_k": KNN_K,
                "knn_use_pca": KNN_USE_PCA,
                "knn_pca_dim": KNN_PCA_DIM,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)
    print("RESUMEN FINAL ABLATION")
    print("=" * 80)
    print(results_df)

    print("\nGanador por val_acc:")
    print(winner)

    print(f"\nArchivos guardados en: {OUT_DIR}")


if __name__ == "__main__":
    main()