from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, silhouette_score
from scipy.spatial.distance import cdist


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
WORKING_DIR = PROJECT_ROOT / "data" / "working"

EMB_DIR = WORKING_DIR / "router" / "embeddings" / "vit_tiny_patch16_224_balanced"
ABL_DIR = WORKING_DIR / "router" / "ablation" / "vit_tiny_patch16_224_balanced"
OUT_DIR = WORKING_DIR / "router" / "diagnostics" / "vit_tiny_patch16_224_balanced"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("EMB_DIR:", EMB_DIR)
print("ABL_DIR:", ABL_DIR)
print("OUT_DIR:", OUT_DIR)


# =========================================================
# 1) Carga
# =========================================================
Z_train = np.load(EMB_DIR / "Z_train.npy")
y_train = np.load(EMB_DIR / "y_train_expert.npy")
Z_val = np.load(EMB_DIR / "Z_val.npy")
y_val = np.load(EMB_DIR / "y_val_expert.npy")
Z_test = np.load(EMB_DIR / "Z_test.npy")
y_test = np.load(EMB_DIR / "y_test_expert.npy")

meta_train = pd.read_csv(EMB_DIR / "meta_train.csv")
meta_val = pd.read_csv(EMB_DIR / "meta_val.csv")
meta_test = pd.read_csv(EMB_DIR / "meta_test.csv")

linear_test_pred = pd.read_csv(ABL_DIR / "linear_test_predictions.csv")
linear_val_pred = pd.read_csv(ABL_DIR / "linear_val_predictions.csv")

expert_map = pd.read_csv(EMB_DIR / "expert_id_map.csv")
expert_map = expert_map.sort_values("expert_id").reset_index(drop=True)

class_ids = expert_map["expert_id"].tolist()
class_names = expert_map["expert_name"].tolist()
id_to_name = dict(zip(class_ids, class_names))

print("\nClases:")
print(expert_map)


# =========================================================
# 2) Matriz de confusión del linear
# =========================================================
y_true_test = linear_test_pred["y_true"].to_numpy()
y_pred_test = linear_test_pred["y_pred"].to_numpy()

cm = confusion_matrix(y_true_test, y_pred_test, labels=class_ids)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv(OUT_DIR / "linear_test_confusion_matrix_reexport.csv")

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("Linear Router - Test Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)
plt.xlabel("Predicted")
plt.ylabel("True")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(OUT_DIR / "linear_test_confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.close()

print("\nMatriz de confusión test guardada.")


# =========================================================
# 3) Preparar embeddings combinados
# =========================================================
Z_all = np.concatenate([Z_train, Z_val, Z_test], axis=0)
y_all = np.concatenate([y_train, y_val, y_test], axis=0)

meta_all = pd.concat([
    meta_train.assign(split="train"),
    meta_val.assign(split="val"),
    meta_test.assign(split="test"),
], ignore_index=True)

scaler = StandardScaler()
Z_all_s = scaler.fit_transform(Z_all)

# Guardar una versión tabular simple
emb_df = meta_all.copy()
for i in range(Z_all_s.shape[1]):
    emb_df[f"z_{i:03d}"] = Z_all_s[:, i]
emb_df.to_csv(OUT_DIR / "all_embeddings_scaled.csv", index=False)

print("\nEmbeddings combinados:", Z_all_s.shape)


# =========================================================
# 4) Métricas de separabilidad
# =========================================================
sil_score = silhouette_score(Z_all_s, y_all, metric="euclidean")
print(f"\nSilhouette score global por experto: {sil_score:.4f}")

# centroides por experto
centroids = []
for expert_id in class_ids:
    cent = Z_all_s[y_all == expert_id].mean(axis=0)
    centroids.append(cent)
centroids = np.stack(centroids, axis=0)

centroid_dist = cdist(centroids, centroids, metric="euclidean")
centroid_df = pd.DataFrame(centroid_dist, index=class_names, columns=class_names)
centroid_df.to_csv(OUT_DIR / "expert_centroid_distance_matrix.csv")

summary = {
    "silhouette_score": float(sil_score),
}
with open(OUT_DIR / "embedding_diagnostics_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\nMatriz de distancias entre centroides guardada.")


# =========================================================
# 5) PCA 2D
# =========================================================
pca = PCA(n_components=2, random_state=42)
Z_pca = pca.fit_transform(Z_all_s)

pca_df = meta_all.copy()
pca_df["pca_1"] = Z_pca[:, 0]
pca_df["pca_2"] = Z_pca[:, 1]
pca_df["expert_name"] = pca_df["expert_id"].map(id_to_name)
pca_df.to_csv(OUT_DIR / "pca_2d_points.csv", index=False)

plt.figure(figsize=(9, 7))
for expert_id, expert_name in zip(class_ids, class_names):
    mask = pca_df["expert_id"] == expert_id
    plt.scatter(
        pca_df.loc[mask, "pca_1"],
        pca_df.loc[mask, "pca_2"],
        s=10,
        alpha=0.65,
        label=expert_name,
    )

plt.title("Router Embeddings - PCA 2D")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% var)")
plt.legend(markerscale=2, fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "router_embeddings_pca_2d.png", dpi=180, bbox_inches="tight")
plt.close()

print("\nPCA 2D guardado.")


# =========================================================
# 6) t-SNE 2D
# =========================================================
# Reducimos primero a 30 dims para acelerar y suavizar ruido
pca_tsne = PCA(n_components=min(30, Z_all_s.shape[1]), random_state=42)
Z_pre_tsne = pca_tsne.fit_transform(Z_all_s)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42,
)
Z_tsne = tsne.fit_transform(Z_pre_tsne)

tsne_df = meta_all.copy()
tsne_df["tsne_1"] = Z_tsne[:, 0]
tsne_df["tsne_2"] = Z_tsne[:, 1]
tsne_df["expert_name"] = tsne_df["expert_id"].map(id_to_name)
tsne_df.to_csv(OUT_DIR / "tsne_2d_points.csv", index=False)

plt.figure(figsize=(9, 7))
for expert_id, expert_name in zip(class_ids, class_names):
    mask = tsne_df["expert_id"] == expert_id
    plt.scatter(
        tsne_df.loc[mask, "tsne_1"],
        tsne_df.loc[mask, "tsne_2"],
        s=10,
        alpha=0.65,
        label=expert_name,
    )

plt.title("Router Embeddings - t-SNE 2D")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(markerscale=2, fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "router_embeddings_tsne_2d.png", dpi=180, bbox_inches="tight")
plt.close()

print("\nt-SNE 2D guardado.")


# =========================================================
# 7) Tabla de errores del linear
# =========================================================
errors_test = linear_test_pred.copy()
errors_test["true_name"] = errors_test["y_true"].map(id_to_name)
errors_test["pred_name"] = errors_test["y_pred"].map(id_to_name)
errors_only = errors_test[errors_test["y_true"] != errors_test["y_pred"]].copy()
errors_only.to_csv(OUT_DIR / "linear_test_errors_only.csv", index=False)

print(f"\nErrores test del linear: {len(errors_only)} / {len(errors_test)}")


# =========================================================
# 8) Resumen final
# =========================================================
print("\n" + "=" * 80)
print("DIAGNÓSTICO TERMINADO")
print("=" * 80)
print(f"Silhouette global: {sil_score:.4f}")
print(f"Errores test linear: {len(errors_only)} / {len(errors_test)}")
print(f"Resultados guardados en: {OUT_DIR}")