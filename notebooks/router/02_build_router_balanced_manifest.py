from pathlib import Path
import pandas as pd


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
ROUTER_MANIFESTS_DIR = WORKING_DIR / "router" / "manifests"
ROUTER_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("ROUTER_MANIFESTS_DIR:", ROUTER_MANIFESTS_DIR)


# =========================================================
# 1) Configuración
# =========================================================
SEED = 42

INPUT_MANIFEST = ROUTER_MANIFESTS_DIR / "router_manifest_all.csv"

# Si quieres forzar tamaños manuales, cambia esto.
# Si lo dejas en None, usa el mínimo automático por split.
TARGETS_BY_SPLIT = {
    "train": None,
    "val": None,
    "test": None,
}

EXPECTED_SPLITS = {"train", "val", "test"}
EXPECTED_EXPERTS = {"nih", "isic", "osteo", "luna16", "pancreas"}


# =========================================================
# 2) Helpers
# =========================================================
def sample_group(df_group: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df_group) < n:
        raise ValueError(
            f"El grupo tiene {len(df_group)} filas y no alcanza para muestrear {n}."
        )
    return df_group.sample(n=n, replace=False, random_state=seed)


def build_balanced_manifest(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    # Validaciones básicas
    missing_cols = {
        "sample_id", "expert_id", "expert_name", "dataset_name", "split",
        "input_path", "file_exists"
    } - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing_cols)}")

    if not df["file_exists"].all():
        bad = (~df["file_exists"]).sum()
        raise FileNotFoundError(f"Hay {bad} archivos con file_exists=False.")

    splits_found = set(df["split"].dropna().unique())
    if not EXPECTED_SPLITS.issubset(splits_found):
        raise ValueError(
            f"Splits encontrados: {sorted(splits_found)} | "
            f"Se esperaban al menos: {sorted(EXPECTED_SPLITS)}"
        )

    experts_found = set(df["expert_name"].dropna().unique())
    if not EXPECTED_EXPERTS.issubset(experts_found):
        raise ValueError(
            f"Expertos encontrados: {sorted(experts_found)} | "
            f"Se esperaban al menos: {sorted(EXPECTED_EXPERTS)}"
        )

    # Conteos base
    counts = (
        df.groupby(["expert_name", "split"])
        .size()
        .reset_index(name="n_samples")
        .sort_values(["split", "expert_name"])
        .reset_index(drop=True)
    )

    print("\nConteos originales por experto x split:")
    print(counts)

    # Objetivos por split
    target_rows = []
    for split in ["train", "val", "test"]:
        split_counts = counts.loc[counts["split"] == split, "n_samples"]
        auto_min = int(split_counts.min())

        target_n = TARGETS_BY_SPLIT[split]
        if target_n is None:
            target_n = auto_min

        if target_n > auto_min:
            raise ValueError(
                f"Para split='{split}', target_n={target_n} supera el mínimo disponible ({auto_min})."
            )

        target_rows.append({
            "split": split,
            "target_n_per_expert": target_n,
            "auto_min_available": auto_min,
        })

    target_df = pd.DataFrame(target_rows)
    print("\nObjetivo balanceado por split:")
    print(target_df)

    # Muestreo balanceado
    sampled_parts = []
    sampled_within_split_rank = []

    for split in ["train", "val", "test"]:
        target_n = int(target_df.loc[target_df["split"] == split, "target_n_per_expert"].iloc[0])

        for expert_name in sorted(EXPECTED_EXPERTS):
            part = df[(df["split"] == split) & (df["expert_name"] == expert_name)].copy()

            # seed desplazado por split y experto para que sea estable y distinto
            local_seed = seed + hash((split, expert_name)) % 100000

            sampled = sample_group(part, n=target_n, seed=local_seed).copy()
            sampled["balanced_target_n"] = target_n
            sampled_parts.append(sampled)

    balanced_df = pd.concat(sampled_parts, ignore_index=True)

    # Orden bonito
    balanced_df = balanced_df.sort_values(
        ["split", "expert_id", "sample_id"]
    ).reset_index(drop=True)

    # Duplicados dentro del balanced
    dup_mask = balanced_df.duplicated(subset=["expert_name", "split", "sample_id"], keep=False)
    n_dups = int(dup_mask.sum())
    if n_dups > 0:
        raise ValueError(f"Se detectaron {n_dups} filas duplicadas en el balanced manifest.")

    return balanced_df, target_df


# =========================================================
# 3) Main
# =========================================================
def main():
    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(f"No existe el archivo: {INPUT_MANIFEST}")

    df = pd.read_csv(INPUT_MANIFEST)
    print(f"\nLeído: {INPUT_MANIFEST}")
    print(f"Shape original: {df.shape}")

    balanced_df, target_df = build_balanced_manifest(df, seed=SEED)

    # Guardados principales
    out_all = ROUTER_MANIFESTS_DIR / "router_manifest_balanced_all.csv"
    out_train = ROUTER_MANIFESTS_DIR / "router_manifest_balanced_train.csv"
    out_val = ROUTER_MANIFESTS_DIR / "router_manifest_balanced_val.csv"
    out_test = ROUTER_MANIFESTS_DIR / "router_manifest_balanced_test.csv"
    out_targets = ROUTER_MANIFESTS_DIR / "router_balanced_targets.csv"
    out_summary = ROUTER_MANIFESTS_DIR / "router_manifest_balanced_summary.csv"

    balanced_df.to_csv(out_all, index=False)
    balanced_df[balanced_df["split"] == "train"].to_csv(out_train, index=False)
    balanced_df[balanced_df["split"] == "val"].to_csv(out_val, index=False)
    balanced_df[balanced_df["split"] == "test"].to_csv(out_test, index=False)
    target_df.to_csv(out_targets, index=False)

    summary = (
        balanced_df.groupby(["expert_name", "split"])
        .size()
        .reset_index(name="n_samples")
        .sort_values(["split", "expert_name"])
        .reset_index(drop=True)
    )
    summary.to_csv(out_summary, index=False)

    # Resumen en consola
    print("\n" + "=" * 80)
    print("RESUMEN BALANCEADO")
    print("=" * 80)

    print("\nConteos balanceados por experto x split:")
    print(summary)

    print("\nTabla experto x split:")
    print(pd.crosstab(balanced_df["expert_name"], balanced_df["split"], dropna=False))

    print("\nTotal final por split:")
    print(balanced_df["split"].value_counts().sort_index())

    print("\nTotal final por experto:")
    print(balanced_df["expert_name"].value_counts().sort_index())

    print(f"\nGuardado: {out_all}")
    print(f"Guardado: {out_train}")
    print(f"Guardado: {out_val}")
    print(f"Guardado: {out_test}")
    print(f"Guardado: {out_targets}")
    print(f"Guardado: {out_summary}")



if __name__ == "__main__":
    main()