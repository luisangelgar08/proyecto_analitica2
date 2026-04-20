from pathlib import Path
import json
import math
import importlib.util

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


# =========================================================
# Config
# =========================================================
PROJECT_ROOT = Path("/home/uaodeepia/luis/proyecto_analitica2")
MOE_SCRIPT = PROJECT_ROOT / "notebooks" / "moe" / "01_moe_inference_pipeline.py"

ID_MANIFEST = PROJECT_ROOT / "data" / "working" / "router" / "manifests" / "router_manifest_balanced_test.csv"

# Crea esta carpeta y mete ahí imágenes NO médicas
OOD_DIR = PROJECT_ROOT / "data" / "source" / "ood_nonmedical"

OUT_DIR = PROJECT_ROOT / "data" / "working" / "moe_eval" / "router_ood_entropy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_ID = 200
N_OOD = 200

OOD_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


# =========================================================
# Cargar RouterInference
# =========================================================
spec = importlib.util.spec_from_file_location("moe_pipeline_module", MOE_SCRIPT)
moe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moe_module)

router = moe_module.RouterInference()


# =========================================================
# Helpers
# =========================================================
def entropy_from_probs(prob_dict):
    probs = np.array(list(prob_dict.values()), dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-(probs * np.log(probs)).sum())


def sample_id_paths():
    if not ID_MANIFEST.exists():
        raise FileNotFoundError(f"No existe ID_MANIFEST: {ID_MANIFEST}")

    id_df = pd.read_csv(ID_MANIFEST)
    id_df = id_df.sample(n=min(N_ID, len(id_df)), random_state=SEED, replace=False).reset_index(drop=True)

    if not OOD_DIR.exists():
        raise FileNotFoundError(
            f"No existe OOD_DIR: {OOD_DIR}\n"
            "Crea esa carpeta y mete imágenes no médicas (.png/.jpg/.jpeg/.webp)."
        )

    ood_files = []
    for p in OOD_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in OOD_EXTS:
            ood_files.append(p)

    if len(ood_files) == 0:
        raise FileNotFoundError(
            f"No encontré imágenes OOD en: {OOD_DIR}\n"
            "Mete imágenes no médicas ahí."
        )

    rng = np.random.default_rng(SEED)
    if len(ood_files) > N_OOD:
        idx = rng.choice(len(ood_files), size=N_OOD, replace=False)
        ood_files = [ood_files[i] for i in idx]

    return id_df, ood_files


def run_router_on_path(path: str | Path):
    result = router.predict(Path(path))
    probs = result["router_probs"]
    pred = result["router_pred_name"]
    ent = entropy_from_probs(probs)
    conf = float(max(probs.values()))
    return pred, conf, ent, probs


def threshold_from_id_entropy(id_entropy, q=0.95):
    return float(np.quantile(np.asarray(id_entropy, dtype=np.float64), q))


# =========================================================
# Main
# =========================================================
def main():
    id_df, ood_files = sample_id_paths()

    print("=" * 80)
    print("OOD EVALUATION DEL ROUTER")
    print("=" * 80)
    print(f"ID samples : {len(id_df)}")
    print(f"OOD samples: {len(ood_files)}")
    print(f"OOD dir    : {OOD_DIR}")

    rows = []

    # ---------------------------
    # ID
    # ---------------------------
    for i, row in id_df.iterrows():
        path = row["input_path"]
        true_expert = row["expert_name"]

        pred, conf, ent, probs = run_router_on_path(path)

        rows.append({
            "split_type": "ID",
            "input_path": path,
            "true_expert": true_expert,
            "router_pred": pred,
            "router_conf": conf,
            "router_entropy": ent,
            "ood_label": 0,  # ID = 0
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(id_df):
            print(f"[ID {i+1}/{len(id_df)}]")

    # ---------------------------
    # OOD
    # ---------------------------
    for i, path in enumerate(ood_files):
        pred, conf, ent, probs = run_router_on_path(path)

        rows.append({
            "split_type": "OOD",
            "input_path": str(path),
            "true_expert": None,
            "router_pred": pred,
            "router_conf": conf,
            "router_entropy": ent,
            "ood_label": 1,  # OOD = 1
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(ood_files):
            print(f"[OOD {i+1}/{len(ood_files)}]")

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(OUT_DIR / "router_ood_entropy_predictions.csv", index=False)

    # AUROC: más entropía => más OOD
    y_true = pred_df["ood_label"].to_numpy()
    y_score = pred_df["router_entropy"].to_numpy()
    auroc = float(roc_auc_score(y_true, y_score))

    id_entropy = pred_df.loc[pred_df["split_type"] == "ID", "router_entropy"].to_numpy()
    ood_entropy = pred_df.loc[pred_df["split_type"] == "OOD", "router_entropy"].to_numpy()

    alert_threshold = threshold_from_id_entropy(id_entropy, q=0.95)

    pred_df["ood_alert"] = (pred_df["router_entropy"] >= alert_threshold).astype(int)
    pred_df.to_csv(OUT_DIR / "router_ood_entropy_predictions_with_alert.csv", index=False)

    id_alert_rate = float(pred_df.loc[pred_df["split_type"] == "ID", "ood_alert"].mean())
    ood_alert_rate = float(pred_df.loc[pred_df["split_type"] == "OOD", "ood_alert"].mean())

    summary = {
        "n_id": int((pred_df["split_type"] == "ID").sum()),
        "n_ood": int((pred_df["split_type"] == "OOD").sum()),
        "ood_auroc_entropy": auroc,
        "entropy_threshold_p95_id": alert_threshold,
        "id_entropy_mean": float(id_entropy.mean()),
        "ood_entropy_mean": float(ood_entropy.mean()),
        "id_alert_rate": id_alert_rate,
        "ood_alert_rate": ood_alert_rate,
        "outputs": {
            "predictions_csv": str(OUT_DIR / "router_ood_entropy_predictions.csv"),
            "predictions_with_alert_csv": str(OUT_DIR / "router_ood_entropy_predictions_with_alert.csv"),
        }
    }

    with open(OUT_DIR / "router_ood_entropy_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("RESUMEN FINAL OOD ROUTER")
    print("=" * 80)
    print("OOD AUROC (entropy) =", summary["ood_auroc_entropy"])
    print("Entropy threshold p95(ID) =", summary["entropy_threshold_p95_id"])
    print("ID entropy mean  =", summary["id_entropy_mean"])
    print("OOD entropy mean =", summary["ood_entropy_mean"])
    print("ID alert rate    =", summary["id_alert_rate"])
    print("OOD alert rate   =", summary["ood_alert_rate"])
    print("\nGuardado en:", OUT_DIR)


if __name__ == "__main__":
    main()