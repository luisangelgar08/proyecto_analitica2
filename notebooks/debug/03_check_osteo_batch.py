from pathlib import Path
import json
import importlib.util

import numpy as np
import pandas as pd


# =========================================================
# Config
# =========================================================
PROJECT_ROOT = Path("/home/uaodeepia/luis/proyecto_analitica2")
MOE_SCRIPT = PROJECT_ROOT / "notebooks" / "moe" / "01_moe_inference_pipeline.py"

MANIFEST_PATH = PROJECT_ROOT / "data" / "working" / "router" / "manifests" / "router_manifest_osteo.csv"
OUT_DIR = PROJECT_ROOT / "data" / "working" / "moe_checks" / "osteo_batch_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_PER_CLASS = 8   # prueba rápida; si quieres más, súbelo a 10 o 15

OSTEO_CLASSES = ["0", "1", "2", "3", "4"]


# =========================================================
# Cargar el pipeline MoE desde el script actual
# =========================================================
spec = importlib.util.spec_from_file_location("moe_pipeline_module", MOE_SCRIPT)
moe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moe_module)

pipeline = moe_module.MoEInferencePipeline()


# =========================================================
# Helpers
# =========================================================
def softmax_top_prob(prob_dict):
    if not isinstance(prob_dict, dict) or len(prob_dict) == 0:
        return np.nan
    return float(max(prob_dict.values()))


def build_sample_df():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"No existe manifest: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)

    # normalizar labels a string
    df["task_label"] = df["task_label"].astype(str)

    parts = []
    for cls in OSTEO_CLASSES:
        part = df[df["task_label"] == cls].copy()
        if len(part) == 0:
            print(f"[WARN] No encontré muestras para clase {cls}")
            continue

        n = min(N_PER_CLASS, len(part))
        part = part.sample(n=n, random_state=SEED, replace=False)
        parts.append(part)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["task_label", "sample_id"]).reset_index(drop=True)
    return out


def confusion_table(df, y_true_col, y_pred_col, labels):
    cm = pd.crosstab(
        pd.Categorical(df[y_true_col], categories=labels),
        pd.Categorical(df[y_pred_col], categories=labels),
        dropna=False
    )
    cm.index.name = "true"
    cm.columns.name = "pred"
    return cm


# =========================================================
# Main
# =========================================================
def main():
    sample_df = build_sample_df()

    print("=" * 80)
    print("MUESTRAS SELECCIONADAS")
    print("=" * 80)
    print(sample_df["task_label"].value_counts().sort_index())
    print(f"Total: {len(sample_df)}")

    rows = []

    for i, row in sample_df.iterrows():
        input_path = row["input_path"]
        true_label = str(row["task_label"])
        sample_id = row["sample_id"]

        result = pipeline.predict(input_path)

        router_pred = result["router"]["router_pred_name"]
        router_conf = softmax_top_prob(result["router"]["router_probs"])

        expert_status = result["expert_status"]

        if expert_status == "ok":
            expert_pred = str(result["expert_output"].get("pred_label", "NA"))
            expert_conf = softmax_top_prob(result["expert_output"].get("probs", {}))
        else:
            expert_pred = "ERROR"
            expert_conf = np.nan

        rows.append({
            "sample_id": sample_id,
            "input_path": input_path,
            "true_label": true_label,
            "router_pred": router_pred,
            "router_conf": router_conf,
            "expert_status": expert_status,
            "expert_pred": expert_pred,
            "expert_conf": expert_conf,
            "router_ok": int(router_pred == "osteo"),
            "expert_ok": int(expert_pred == true_label),
        })

        print(f"[{i+1:02d}/{len(sample_df)}] true={true_label} | router={router_pred} ({router_conf:.4f}) | expert={expert_pred} ({expert_conf if pd.notna(expert_conf) else 'NA'})")

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(OUT_DIR / "osteo_batch_predictions.csv", index=False)

    # métricas
    router_acc = float(pred_df["router_ok"].mean()) if len(pred_df) else np.nan
    expert_acc = float(pred_df["expert_ok"].mean()) if len(pred_df) else np.nan

    class_summary = (
        pred_df.groupby("true_label")
        .agg(
            n=("true_label", "size"),
            router_acc=("router_ok", "mean"),
            expert_acc=("expert_ok", "mean"),
            mean_router_conf=("router_conf", "mean"),
            mean_expert_conf=("expert_conf", "mean"),
        )
        .reset_index()
        .sort_values("true_label")
    )
    class_summary.to_csv(OUT_DIR / "osteo_batch_class_summary.csv", index=False)

    cm = confusion_table(pred_df, "true_label", "expert_pred", OSTEO_CLASSES)
    cm.to_csv(OUT_DIR / "osteo_batch_confusion_matrix.csv")

    suspicious = pred_df[
        (pred_df["expert_ok"] == 0) & (pred_df["expert_conf"] >= 0.95)
    ].copy()
    suspicious.to_csv(OUT_DIR / "osteo_batch_high_confidence_errors.csv", index=False)

    summary = {
        "n_total": int(len(pred_df)),
        "n_per_class_requested": int(N_PER_CLASS),
        "router_acc_on_osteo_subset": router_acc,
        "expert_acc_on_osteo_subset": expert_acc,
        "n_high_confidence_errors": int(len(suspicious)),
        "outputs": {
            "predictions_csv": str(OUT_DIR / "osteo_batch_predictions.csv"),
            "class_summary_csv": str(OUT_DIR / "osteo_batch_class_summary.csv"),
            "confusion_matrix_csv": str(OUT_DIR / "osteo_batch_confusion_matrix.csv"),
            "high_confidence_errors_csv": str(OUT_DIR / "osteo_batch_high_confidence_errors.csv"),
        }
    }

    with open(OUT_DIR / "osteo_batch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("RESUMEN FINAL OSTEO")
    print("=" * 80)
    print("router_acc_on_osteo_subset =", router_acc)
    print("expert_acc_on_osteo_subset =", expert_acc)
    print("\nResumen por clase:")
    print(class_summary)
    print("\nMatriz de confusión:")
    print(cm)
    print("\nHigh-confidence errors:", len(suspicious))
    print("\nGuardado en:", OUT_DIR)


if __name__ == "__main__":
    main()