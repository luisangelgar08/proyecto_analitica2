from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Config
# =========================================================
PROJECT_ROOT = Path("/home/uaodeepia/luis/proyecto_analitica2")

ROUTER_FINAL_DIR = PROJECT_ROOT / "data" / "working" / "router" / "final" / "vit_tiny_patch16_224_balanced"
ROUTER_AUX_DIR = PROJECT_ROOT / "data" / "working" / "router" / "aux_training" / "vit_tiny_patch16_224_balanced"

MOE_FINAL_EVAL_DIR = PROJECT_ROOT / "data" / "working" / "moe" / "final" / "eval"
OOD_DIR = PROJECT_ROOT / "data" / "working" / "moe_eval" / "router_ood_entropy"

OUT_DIR = PROJECT_ROOT / "data" / "working" / "final_results_bundle"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Helpers
# =========================================================
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_bar_plot(x, y, title, xlabel, ylabel, out_path, rotation=0, ylim=None):
    plt.figure(figsize=(8, 5))
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if rotation:
        plt.xticks(rotation=rotation, ha="right")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_line_plot(x, y, title, xlabel, ylabel, out_path, ylim=None):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=1.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_multi_line_plot(df, x_col, y_cols, title, xlabel, ylabel, out_path):
    plt.figure(figsize=(9, 5))
    for col in y_cols:
        plt.plot(df[x_col], df[col], marker="o", linewidth=1.2, label=col)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_heatmap_from_crosstab(df, title, out_path):
    arr = df.to_numpy()

    plt.figure(figsize=(7, 6))
    plt.imshow(arr, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    plt.xticks(np.arange(len(df.columns)), df.columns, rotation=45, ha="right")
    plt.yticks(np.arange(len(df.index)), df.index)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            plt.text(j, i, str(arr[i, j]), ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


# =========================================================
# Load inputs
# =========================================================
router_summary = load_json(ROUTER_FINAL_DIR / "router_aux_summary_alpha_0p02.json")
if router_summary is None:
    router_summary = load_json(ROUTER_AUX_DIR / "router_aux_summary_alpha_0p02.json")

router_history_path = ROUTER_FINAL_DIR / "router_aux_history_alpha_0p02.csv"
if not router_history_path.exists():
    router_history_path = ROUTER_AUX_DIR / "router_aux_history_alpha_0p02.csv"

router_history = pd.read_csv(router_history_path)

moe_summary = load_json(MOE_FINAL_EVAL_DIR / "moe_balanced_test_summary.json")
routing_summary = pd.read_csv(MOE_FINAL_EVAL_DIR / "routing_summary_by_expert.csv")
expert_summary = pd.read_csv(MOE_FINAL_EVAL_DIR / "expert_summary_by_expert.csv")
routing_cm = pd.read_csv(MOE_FINAL_EVAL_DIR / "routing_confusion_matrix.csv", index_col=0)

ood_summary = load_json(OOD_DIR / "router_ood_entropy_summary.json")


# =========================================================
# 1) Router balance figures
# =========================================================
save_line_plot(
    router_history["epoch"],
    router_history["train_ratio"],
    "Router Balance Ratio (Train)",
    "Epoch",
    "max(fi)/min(fi)",
    OUT_DIR / "figure_router_train_ratio.png",
)

save_line_plot(
    router_history["epoch"],
    router_history["val_ratio"],
    "Router Balance Ratio (Validation)",
    "Epoch",
    "max(fi)/min(fi)",
    OUT_DIR / "figure_router_val_ratio.png",
)

train_f_cols = [c for c in router_history.columns if c.startswith("train_f_")]
val_f_cols = [c for c in router_history.columns if c.startswith("val_f_")]

save_multi_line_plot(
    router_history,
    "epoch",
    train_f_cols,
    "Router fi Evolution (Train)",
    "Epoch",
    "fi",
    OUT_DIR / "figure_router_fi_train.png",
)

save_multi_line_plot(
    router_history,
    "epoch",
    val_f_cols,
    "Router fi Evolution (Validation)",
    "Epoch",
    "fi",
    OUT_DIR / "figure_router_fi_val.png",
)


# =========================================================
# 2) MoE figures
# =========================================================
save_heatmap_from_crosstab(
    routing_cm,
    "Routing Confusion Matrix",
    OUT_DIR / "figure_routing_confusion_matrix.png",
)

save_bar_plot(
    routing_summary["true_expert"],
    routing_summary["router_acc"],
    "Routing Accuracy by Expert",
    "Expert",
    "Routing Accuracy",
    OUT_DIR / "figure_routing_accuracy_by_expert.png",
    rotation=30,
    ylim=(0, 1.05),
)

save_bar_plot(
    expert_summary["true_expert"],
    expert_summary["expert_metric_mean"],
    "Expert Metric by Expert",
    "Expert",
    "Expert Metric Mean",
    OUT_DIR / "figure_expert_metric_by_expert.png",
    rotation=30,
    ylim=(0, 1.05),
)

save_bar_plot(
    routing_summary["true_expert"],
    routing_summary["mean_router_conf"],
    "Mean Router Confidence by Expert",
    "Expert",
    "Mean Router Confidence",
    OUT_DIR / "figure_router_conf_by_expert.png",
    rotation=30,
    ylim=(0, 1.05),
)


# =========================================================
# 3) OOD figure
# =========================================================
if ood_summary is not None:
    ood_bar = pd.DataFrame({
        "metric": ["ID entropy mean", "OOD entropy mean", "ID alert rate", "OOD alert rate", "OOD AUROC"],
        "value": [
            ood_summary["id_entropy_mean"],
            ood_summary["ood_entropy_mean"],
            ood_summary["id_alert_rate"],
            ood_summary["ood_alert_rate"],
            ood_summary["ood_auroc_entropy"],
        ]
    })
    ood_bar.to_csv(OUT_DIR / "ood_metrics_table.csv", index=False)

    save_bar_plot(
        ood_bar["metric"],
        ood_bar["value"],
        "Router OOD Metrics",
        "Metric",
        "Value",
        OUT_DIR / "figure_router_ood_metrics.png",
        rotation=30,
    )


# =========================================================
# 4) Final metrics table
# =========================================================
final_rows = []

# Router final
final_rows.append({
    "component": "router_final",
    "metric_name": "routing_accuracy_test",
    "metric_value": router_summary["test_acc"],
})
final_rows.append({
    "component": "router_final",
    "metric_name": "load_balance_ratio_test",
    "metric_value": router_summary["test_ratio"],
})
final_rows.append({
    "component": "router_final",
    "metric_name": "alpha",
    "metric_value": router_summary["alpha"],
})

# Router fi final
for k, v in router_summary["test_fi"].items():
    final_rows.append({
        "component": "router_final",
        "metric_name": f"test_fi_{k}",
        "metric_value": v,
    })

# OOD
if ood_summary is not None:
    final_rows.append({
        "component": "router_final",
        "metric_name": "ood_auroc_entropy",
        "metric_value": ood_summary["ood_auroc_entropy"],
    })
    final_rows.append({
        "component": "router_final",
        "metric_name": "ood_threshold_p95_id",
        "metric_value": ood_summary["entropy_threshold_p95_id"],
    })

# MoE
final_rows.append({
    "component": "moe_final",
    "metric_name": "router_acc_overall",
    "metric_value": moe_summary["router_acc_overall"],
})
final_rows.append({
    "component": "moe_final",
    "metric_name": "n_expert_status_ok",
    "metric_value": moe_summary["n_expert_status_ok"],
})
final_rows.append({
    "component": "moe_final",
    "metric_name": "n_expert_status_error",
    "metric_value": moe_summary["n_expert_status_error"],
})

final_metrics_df = pd.DataFrame(final_rows)
final_metrics_df.to_csv(OUT_DIR / "final_metrics_table.csv", index=False)

routing_summary.to_csv(OUT_DIR / "final_routing_summary_by_expert.csv", index=False)
expert_summary.to_csv(OUT_DIR / "final_expert_summary_by_expert.csv", index=False)
routing_cm.to_csv(OUT_DIR / "final_routing_confusion_matrix.csv")


# =========================================================
# 5) Final bundle JSON
# =========================================================
bundle_summary = {
    "router_final_summary": router_summary,
    "moe_final_summary": moe_summary,
    "ood_summary": ood_summary,
    "outputs": {
        "final_metrics_table_csv": str(OUT_DIR / "final_metrics_table.csv"),
        "routing_summary_csv": str(OUT_DIR / "final_routing_summary_by_expert.csv"),
        "expert_summary_csv": str(OUT_DIR / "final_expert_summary_by_expert.csv"),
        "routing_confusion_matrix_csv": str(OUT_DIR / "final_routing_confusion_matrix.csv"),
    }
}

with open(OUT_DIR / "final_results_bundle_summary.json", "w", encoding="utf-8") as f:
    json.dump(bundle_summary, f, indent=2, ensure_ascii=False)


print("=" * 80)
print("FINAL RESULTS BUNDLE GENERADO")
print("=" * 80)
print("Salida:", OUT_DIR)
print("\nArchivos creados:")
for p in sorted(OUT_DIR.glob("*")):
    print("-", p.name)