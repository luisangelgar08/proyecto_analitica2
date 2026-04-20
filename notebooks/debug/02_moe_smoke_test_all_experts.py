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
MANIFEST_ALL = PROJECT_ROOT / "data" / "working" / "router" / "manifests" / "router_manifest_all.csv"

OUT_DIR = PROJECT_ROOT / "data" / "working" / "moe_checks" / "smoke_test_all_experts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_PER_EXPERT = 5

EXPERTS = ["nih", "isic", "osteo", "luna16", "pancreas"]


# =========================================================
# Cargar el pipeline actual
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


def sample_manifest():
    if not MANIFEST_ALL.exists():
        raise FileNotFoundError(f"No existe manifest: {MANIFEST_ALL}")

    df = pd.read_csv(MANIFEST_ALL)

    parts = []
    for expert in EXPERTS:
        part = df[df["expert_name"] == expert].copy()
        if len(part) == 0:
            print(f"[WARN] No encontré muestras para experto {expert}")
            continue

        n = min(N_PER_EXPERT, len(part))
        part = part.sample(n=n, random_state=SEED, replace=False)
        parts.append(part)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["expert_name", "sample_id"]).reset_index(drop=True)
    return out


def summarize_expert_output(expert_output):
    if not isinstance(expert_output, dict):
        return None, None

    if "pred_label" in expert_output:
        return expert_output.get("pred_label"), max_prob(expert_output.get("probs", {}))

    if "predictions" in expert_output:
        preds = expert_output.get("predictions", [])
        if len(preds) == 0:
            return None, np.nan
        top = max(preds, key=lambda x: x.get("prob", -1))
        return top.get("label"), float(top.get("prob", np.nan))

    return None, np.nan


# =========================================================
# Main
# =========================================================
def main():
    df = sample_manifest()

    print("=" * 80)
    print("MUESTRAS SELECCIONADAS")
    print("=" * 80)
    print(df["expert_name"].value_counts().sort_index())
    print(f"Total: {len(df)}")

    rows = []

    for i, row in df.iterrows():
        input_path = row["input_path"]
        true_expert = row["expert_name"]
        sample_id = row["sample_id"]
        split = row["split"]

        result = pipeline.predict(input_path)

        router_pred = result["router"]["router_pred_name"]
        router_conf = max_prob(result["router"]["router_probs"])
        expert_status = result["expert_status"]

        expert_pred_label, expert_conf = summarize_expert_output(result.get("expert_output", {}))

        rows.append({
            "sample_id": sample_id,
            "split": split,
            "input_path": input_path,
            "true_expert": true_expert,
            "router_pred": router_pred,
            "router_conf": router_conf,
            "router_ok": int(router_pred == true_expert),
            "expert_status": expert_status,
            "expert_pred_label": expert_pred_label,
            "expert_conf": expert_conf,
        })

        print(
            f"[{i+1:02d}/{len(df)}] "
            f"true_expert={true_expert} | "
            f"router={router_pred} ({router_conf:.4f}) | "
            f"expert_status={expert_status} | "
            f"expert_pred={expert_pred_label} ({expert_conf if pd.notna(expert_conf) else 'NA'})"
        )

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(OUT_DIR / "smoke_test_predictions.csv", index=False)

    # resumen por experto
    summary_expert = (
        pred_df.groupby("true_expert")
        .agg(
            n=("true_expert", "size"),
            router_acc=("router_ok", "mean"),
            mean_router_conf=("router_conf", "mean"),
            ok_count=("expert_status", lambda s: int((s == "ok").sum())),
            error_count=("expert_status", lambda s: int((s == "error").sum())),
            mean_expert_conf=("expert_conf", "mean"),
        )
        .reset_index()
        .sort_values("true_expert")
    )
    summary_expert.to_csv(OUT_DIR / "smoke_test_summary_by_expert.csv", index=False)

    # matriz de routing
    routing_cm = pd.crosstab(
        pred_df["true_expert"],
        pred_df["router_pred"],
        dropna=False
    )
    routing_cm.to_csv(OUT_DIR / "routing_confusion_matrix.csv")

    overall = {
        "n_total": int(len(pred_df)),
        "n_per_expert_requested": int(N_PER_EXPERT),
        "router_acc_overall": float(pred_df["router_ok"].mean()) if len(pred_df) else None,
        "n_expert_status_ok": int((pred_df["expert_status"] == "ok").sum()),
        "n_expert_status_error": int((pred_df["expert_status"] == "error").sum()),
        "outputs": {
            "predictions_csv": str(OUT_DIR / "smoke_test_predictions.csv"),
            "summary_by_expert_csv": str(OUT_DIR / "smoke_test_summary_by_expert.csv"),
            "routing_confusion_matrix_csv": str(OUT_DIR / "routing_confusion_matrix.csv"),
        }
    }

    with open(OUT_DIR / "smoke_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("RESUMEN FINAL SMOKE TEST")
    print("=" * 80)
    print("router_acc_overall =", overall["router_acc_overall"])
    print("n_expert_status_ok =", overall["n_expert_status_ok"])
    print("n_expert_status_error =", overall["n_expert_status_error"])
    print("\nResumen por experto:")
    print(summary_expert)
    print("\nMatriz de routing:")
    print(routing_cm)
    print("\nGuardado en:", OUT_DIR)


if __name__ == "__main__":
    main()