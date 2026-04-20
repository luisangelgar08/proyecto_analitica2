from pathlib import Path
import json
import time
import importlib.util

import numpy as np
import pandas as pd
import torch


# =========================================================
# Config
# =========================================================
PROJECT_ROOT = Path("/home/uaodeepia/luis/proyecto_analitica2")
MOE_SCRIPT = PROJECT_ROOT / "notebooks" / "moe" / "01_moe_inference_pipeline.py"
MANIFEST_PATH = PROJECT_ROOT / "data" / "working" / "router" / "manifests" / "router_manifest_balanced_test.csv"

OUT_DIR = PROJECT_ROOT / "data" / "working" / "moe_eval" / "runtime_vram"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_PER_EXPERT = 83   # 83 usa todo el balanced_test
EXPERTS = ["nih", "isic", "osteo", "luna16", "pancreas"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GB = 1024 ** 3


# =========================================================
# Load pipeline
# =========================================================
spec = importlib.util.spec_from_file_location("moe_pipeline_module", MOE_SCRIPT)
moe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moe_module)

pipeline = moe_module.MoEInferencePipeline()


# =========================================================
# Helpers
# =========================================================
def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mem_gb(x: int) -> float:
    return float(x / GB)


def build_eval_df():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"No existe manifest: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)
    parts = []

    for expert in EXPERTS:
        part = df[df["expert_name"] == expert].copy()
        if len(part) == 0:
            print(f"[WARN] No encontré muestras para experto: {expert}")
            continue

        n = min(N_PER_EXPERT, len(part))
        part = part.sample(n=n, random_state=SEED, replace=False)
        parts.append(part)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["expert_name", "sample_id"]).reset_index(drop=True)
    return out


def preload_experts(df: pd.DataFrame):
    """
    Warm-up para evitar medir el costo de primera carga de checkpoints.
    """
    print("=" * 80)
    print("WARM-UP / PRELOAD")
    print("=" * 80)

    used = set()
    for _, row in df.iterrows():
        expert = row["expert_name"]
        if expert in used:
            continue

        path = row["input_path"]

        try:
            _ = pipeline.predict(path)
            used.add(expert)
            print(f"[PRELOAD OK] {expert}")
        except Exception as e:
            print(f"[PRELOAD ERROR] {expert}: {e}")

    if torch.cuda.is_available():
        sync_cuda()
        torch.cuda.empty_cache()


def measure_router_only(path: str):
    sync_cuda()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        alloc_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
    else:
        alloc_before = 0
        reserved_before = 0

    t0 = time.perf_counter()
    result = pipeline.router.predict(Path(path))
    sync_cuda()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if torch.cuda.is_available():
        alloc_after = torch.cuda.memory_allocated()
        reserved_after = torch.cuda.memory_reserved()
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
    else:
        alloc_after = reserved_after = peak_alloc = peak_reserved = 0

    return {
        "result": result,
        "elapsed_ms": elapsed_ms,
        "alloc_before_gb": mem_gb(alloc_before),
        "reserved_before_gb": mem_gb(reserved_before),
        "alloc_after_gb": mem_gb(alloc_after),
        "reserved_after_gb": mem_gb(reserved_after),
        "peak_alloc_gb": mem_gb(peak_alloc),
        "peak_reserved_gb": mem_gb(peak_reserved),
    }


def measure_moe_full(path: str):
    sync_cuda()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        alloc_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
    else:
        alloc_before = 0
        reserved_before = 0

    t0 = time.perf_counter()
    result = pipeline.predict(path)
    sync_cuda()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if torch.cuda.is_available():
        alloc_after = torch.cuda.memory_allocated()
        reserved_after = torch.cuda.memory_reserved()
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
    else:
        alloc_after = reserved_after = peak_alloc = peak_reserved = 0

    return {
        "result": result,
        "elapsed_ms": elapsed_ms,
        "alloc_before_gb": mem_gb(alloc_before),
        "reserved_before_gb": mem_gb(reserved_before),
        "alloc_after_gb": mem_gb(alloc_after),
        "reserved_after_gb": mem_gb(reserved_after),
        "peak_alloc_gb": mem_gb(peak_alloc),
        "peak_reserved_gb": mem_gb(peak_reserved),
    }


def summarize_stage(df: pd.DataFrame, stage_name: str):
    stage_df = df[df["stage"] == stage_name].copy()

    overall = {
        "n": int(len(stage_df)),
        "latency_mean_ms": float(stage_df["elapsed_ms"].mean()),
        "latency_std_ms": float(stage_df["elapsed_ms"].std(ddof=0)),
        "latency_max_ms": float(stage_df["elapsed_ms"].max()),
        "peak_alloc_mean_gb": float(stage_df["peak_alloc_gb"].mean()),
        "peak_alloc_max_gb": float(stage_df["peak_alloc_gb"].max()),
        "peak_reserved_mean_gb": float(stage_df["peak_reserved_gb"].mean()),
        "peak_reserved_max_gb": float(stage_df["peak_reserved_gb"].max()),
    }

    by_expert = (
        stage_df.groupby("true_expert")
        .agg(
            n=("true_expert", "size"),
            latency_mean_ms=("elapsed_ms", "mean"),
            latency_max_ms=("elapsed_ms", "max"),
            peak_alloc_mean_gb=("peak_alloc_gb", "mean"),
            peak_alloc_max_gb=("peak_alloc_gb", "max"),
            peak_reserved_mean_gb=("peak_reserved_gb", "mean"),
            peak_reserved_max_gb=("peak_reserved_gb", "max"),
        )
        .reset_index()
        .sort_values("true_expert")
    )

    return overall, by_expert


# =========================================================
# Main
# =========================================================
def main():
    df = build_eval_df()

    print("=" * 80)
    print("CONFIG")
    print("=" * 80)
    print("DEVICE:", DEVICE)
    print("Muestras por experto:", N_PER_EXPERT)
    print(df["expert_name"].value_counts().sort_index())
    print("Total:", len(df))

    preload_experts(df)

    rows = []

    print("\n" + "=" * 80)
    print("MEDICIÓN ROUTER ONLY")
    print("=" * 80)
    for i, row in df.iterrows():
        path = row["input_path"]
        true_expert = row["expert_name"]
        sample_id = row["sample_id"]

        info = measure_router_only(path)
        router_pred = info["result"]["router_pred_name"]

        rows.append({
            "stage": "router_only",
            "sample_id": sample_id,
            "true_expert": true_expert,
            "router_pred": router_pred,
            "selected_expert": None,
            "expert_status": None,
            "elapsed_ms": info["elapsed_ms"],
            "alloc_before_gb": info["alloc_before_gb"],
            "reserved_before_gb": info["reserved_before_gb"],
            "alloc_after_gb": info["alloc_after_gb"],
            "reserved_after_gb": info["reserved_after_gb"],
            "peak_alloc_gb": info["peak_alloc_gb"],
            "peak_reserved_gb": info["peak_reserved_gb"],
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            print(f"[router_only {i+1}/{len(df)}]")

    print("\n" + "=" * 80)
    print("MEDICIÓN MOE FULL")
    print("=" * 80)
    for i, row in df.iterrows():
        path = row["input_path"]
        true_expert = row["expert_name"]
        sample_id = row["sample_id"]

        info = measure_moe_full(path)
        router_pred = info["result"]["router"]["router_pred_name"]
        selected_expert = info["result"]["selected_expert"]
        expert_status = info["result"]["expert_status"]

        rows.append({
            "stage": "moe_full",
            "sample_id": sample_id,
            "true_expert": true_expert,
            "router_pred": router_pred,
            "selected_expert": selected_expert,
            "expert_status": expert_status,
            "elapsed_ms": info["elapsed_ms"],
            "alloc_before_gb": info["alloc_before_gb"],
            "reserved_before_gb": info["reserved_before_gb"],
            "alloc_after_gb": info["alloc_after_gb"],
            "reserved_after_gb": info["reserved_after_gb"],
            "peak_alloc_gb": info["peak_alloc_gb"],
            "peak_reserved_gb": info["peak_reserved_gb"],
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            print(f"[moe_full {i+1}/{len(df)}]")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT_DIR / "runtime_vram_raw_measurements.csv", index=False)

    router_overall, router_by_expert = summarize_stage(result_df, "router_only")
    moe_overall, moe_by_expert = summarize_stage(result_df, "moe_full")

    router_by_expert.to_csv(OUT_DIR / "router_only_runtime_vram_by_expert.csv", index=False)
    moe_by_expert.to_csv(OUT_DIR / "moe_full_runtime_vram_by_expert.csv", index=False)

    summary = {
        "device": DEVICE,
        "n_per_expert": int(N_PER_EXPERT),
        "router_only": router_overall,
        "moe_full": moe_overall,
        "notes": {
            "router_only": "Incluye preprocesado + backbone ViT + scaler + cabeza linear del router final.",
            "moe_full": "Incluye router + selección de experto + inferencia del experto.",
            "peak_reserved_recommendation": "Usa peak_reserved_max_gb como referencia conservadora de VRAM por tarjeta.",
        }
    }

    with open(OUT_DIR / "runtime_vram_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("RESUMEN FINAL VRAM / LATENCY")
    print("=" * 80)
    print("ROUTER ONLY:")
    print(router_overall)
    print("\nMOE FULL:")
    print(moe_overall)
    print("\nGuardado en:", OUT_DIR)


if __name__ == "__main__":
    main()