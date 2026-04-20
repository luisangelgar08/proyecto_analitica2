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
TEST_MANIFEST = PROJECT_ROOT / "data" / "working" / "router" / "manifests" / "router_manifest_balanced_test.csv"

OUT_DIR = PROJECT_ROOT / "data" / "working" / "moe_eval" / "balanced_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERTS = ["nih", "isic", "osteo", "luna16", "pancreas"]


# =========================================================
# Cargar pipeline
# =========================================================
spec = importlib.util.spec_from_file_location("moe_pipeline_module", MOE_SCRIPT)
moe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moe_module)

pipeline = moe_module.MoEInferencePipeline()


# =========================================================
# Helpers
# =========================================================
def max_prob(prob_dict):
    if not isinstance(prob_dict, dict) or len(prob_dict) == 0:
        return np.nan
    return float(max(prob_dict.values()))


def normalize_task_label(expert_name: str, task_label):
    """
    Convierte el label del manifest a la forma comparable con pred_label.
    """
    x = str(task_label)

    if expert_name == "isic":
        return x

    if expert_name == "osteo":
        return x

    if expert_name == "luna16":
        mapping = {
            "0": "negative",
            "1": "positive",
            0: "negative",
            1: "positive",
        }
        return mapping.get(task_label, mapping.get(x, x))

    if expert_name == "pancreas":
        mapping = {
            "0": "non-PDAC",
            "1": "PDAC",
            0: "non-PDAC",
            1: "PDAC",
        }
        return mapping.get(task_label, mapping.get(x, x))

    if expert_name == "nih":
        # NIH es multilabel, se procesa aparte
        return x

    return x


def evaluate_nih_output(expert_output: dict, true_task_label: str):
    """
    Para NIH, el manifest puede traer varias etiquetas separadas por '|'.
    Consideramos acierto si alguna etiqueta predicha aparece en las verdaderas.
    """
    true_labels = {s.strip() for s in str(true_task_label).split("|") if str(s).strip() != ""}
    preds = expert_output.get("predictions", [])

    pred_labels = [p.get("label") for p in preds if isinstance(p, dict) and "label" in p]

    if len(true_labels) == 0:
        return np.nan, None, np.nan

    hit_any = int(any(lbl in true_labels for lbl in pred_labels))
    top_label = pred_labels[0] if len(pred_labels) > 0 else None
    top_conf = preds[0].get("prob", np.nan) if len(preds) > 0 and isinstance(preds[0], dict) else np.nan
    return hit_any, top_label, top_conf


def evaluate_singlelabel_output(expert_name: str, expert_output: dict, true_task_label):
    true_label = normalize_task_label(expert_name, true_task_label)
    pred_label = expert_output.get("pred_label")
    pred_conf = max_prob(expert_output.get("probs", {}))
    hit = int(str(pred_label) == str(true_label))
    return hit, pred_label, pred_conf, true_label


# =========================================================
# Main
# =========================================================
def main():
    if not TEST_MANIFEST.exists():
        raise FileNotFoundError(f"No existe manifest de test: {TEST_MANIFEST}")

    df = pd.read_csv(TEST_MANIFEST)
    df = df[df["expert_name"].isin(EXPERTS)].copy()

    print("=" * 80)
    print("MANIFEST TEST BALANCEADO")
    print("=" * 80)
    print(df["expert_name"].value_counts().sort_index())
    print(f"Total: {len(df)}")

    rows = []

    for i, row in df.iterrows():
        sample_id = row["sample_id"]
        input_path = row["input_path"]
        true_expert = row["expert_name"]
        true_task_label = row["task_label"]

        result = pipeline.predict(input_path)

        router_pred = result["router"]["router_pred_name"]
        router_conf = max_prob(result["router"]["router_probs"])
        router_ok = int(router_pred == true_expert)

        expert_status = result["expert_status"]
        expert_output = result.get("expert_output", {})

        expert_metric = np.nan
        expert_pred_label = None
        expert_conf = np.nan
        true_label_normalized = None

        if expert_status == "ok":
            if true_expert == "nih":
                expert_metric, expert_pred_label, expert_conf = evaluate_nih_output(
                    expert_output, true_task_label
                )
                true_label_normalized = str(true_task_label)
            else:
                expert_metric, expert_pred_label, expert_conf, true_label_normalized = evaluate_singlelabel_output(
                    true_expert, expert_output, true_task_label
                )

        rows.append({
            "sample_id": sample_id,
            "input_path": input_path,
            "true_expert": true_expert,
            "router_pred": router_pred,
            "router_conf": router_conf,
            "router_ok": router_ok,
            "expert_status": expert_status,
            "true_task_label_raw": str(true_task_label),
            "true_task_label_normalized": true_label_normalized,
            "expert_pred_label": expert_pred_label,
            "expert_conf": expert_conf,
            "expert_metric": expert_metric,
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            print(f"[{i+1}/{len(df)}] procesadas")

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(OUT_DIR / "moe_balanced_test_predictions.csv", index=False)

    # -----------------------------------------------------
    # Routing summary
    # -----------------------------------------------------
    routing_confusion = pd.crosstab(
        pred_df["true_expert"],
        pred_df["router_pred"],
        dropna=False
    )
    routing_confusion.to_csv(OUT_DIR / "routing_confusion_matrix.csv")

    routing_summary = (
        pred_df.groupby("true_expert")
        .agg(
            n=("true_expert", "size"),
            router_acc=("router_ok", "mean"),
            mean_router_conf=("router_conf", "mean"),
        )
        .reset_index()
        .sort_values("true_expert")
    )
    routing_summary.to_csv(OUT_DIR / "routing_summary_by_expert.csv", index=False)

    # -----------------------------------------------------
    # Expert summary
    # -----------------------------------------------------
    expert_summary = (
        pred_df.groupby("true_expert")
        .agg(
            n=("true_expert", "size"),
            ok_count=("expert_status", lambda s: int((s == "ok").sum())),
            error_count=("expert_status", lambda s: int((s == "error").sum())),
            expert_metric_mean=("expert_metric", "mean"),
            mean_expert_conf=("expert_conf", "mean"),
        )
        .reset_index()
        .sort_values("true_expert")
    )
    expert_summary.to_csv(OUT_DIR / "expert_summary_by_expert.csv", index=False)

    # -----------------------------------------------------
    # Overall summary
    # -----------------------------------------------------
    overall = {
        "n_total": int(len(pred_df)),
        "router_acc_overall": float(pred_df["router_ok"].mean()),
        "n_expert_status_ok": int((pred_df["expert_status"] == "ok").sum()),
        "n_expert_status_error": int((pred_df["expert_status"] == "error").sum()),
        "outputs": {
            "predictions_csv": str(OUT_DIR / "moe_balanced_test_predictions.csv"),
            "routing_confusion_matrix_csv": str(OUT_DIR / "routing_confusion_matrix.csv"),
            "routing_summary_by_expert_csv": str(OUT_DIR / "routing_summary_by_expert.csv"),
            "expert_summary_by_expert_csv": str(OUT_DIR / "expert_summary_by_expert.csv"),
        }
    }

    with open(OUT_DIR / "moe_balanced_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("RESUMEN FINAL EVALUACIÓN MOE")
    print("=" * 80)
    print("router_acc_overall =", overall["router_acc_overall"])
    print("n_expert_status_ok =", overall["n_expert_status_ok"])
    print("n_expert_status_error =", overall["n_expert_status_error"])
    print("\nRouting summary by expert:")
    print(routing_summary)
    print("\nExpert summary by expert:")
    print(expert_summary)
    print("\nRouting confusion matrix:")
    print(routing_confusion)
    print("\nGuardado en:", OUT_DIR)


if __name__ == "__main__":
    main()