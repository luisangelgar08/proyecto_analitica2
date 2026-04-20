from pathlib import Path
import pandas as pd
import numpy as np


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
MANIFESTS_DIR = DATA_DIR / "manifests"

ROUTER_WORKING_DIR = WORKING_DIR / "router"
ROUTER_MANIFESTS_DIR = ROUTER_WORKING_DIR / "manifests"
ROUTER_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_DIR:", DATA_DIR)
print("WORKING_DIR:", WORKING_DIR)
print("MANIFESTS_DIR:", MANIFESTS_DIR)
print("ROUTER_MANIFESTS_DIR:", ROUTER_MANIFESTS_DIR)


# =========================================================
# 1) Configuración general
# =========================================================
OLD_PROJECT_ROOTS = [
    Path("/mnt/d/Universidad/analitica/proyecto_analitica2"),  # PC/WSL
]

EXPERT_MAP = {
    "nih": 0,
    "isic": 1,
    "osteo": 2,
    "luna16": 3,
    "pancreas": 4,
}


# =========================================================
# 2) Helpers
# =========================================================
def remap_project_path(value, project_root: Path) -> str | pd._libs.missing.NAType:
    """
    Corrige rutas absolutas viejas del PC para que apunten al clúster.
    """
    if pd.isna(value):
        return pd.NA

    s = str(value).strip()
    if not s:
        return pd.NA

    # Caso 1: reemplazar prefijo viejo del proyecto
    for old_root in OLD_PROJECT_ROOTS:
        old_root_str = str(old_root)
        if s.startswith(old_root_str):
            rel = Path(s).relative_to(old_root_str)
            return str(project_root / rel)

    # Caso 2: si trae algún tail reconocible desde /data/source/
    marker = "/data/source/"
    if marker in s:
        tail = s.split(marker, 1)[1]
        return str(project_root / "data" / "source" / tail)

    # Caso 3: si ya es ruta válida en clúster o relativa rara, la dejamos igual
    return s


def to_rel_from_project(value, project_root: Path):
    if pd.isna(value):
        return pd.NA
    try:
        p = Path(str(value))
        return str(p.relative_to(project_root))
    except Exception:
        return pd.NA


def ensure_str_col(df: pd.DataFrame, col: str, default="") -> pd.Series:
    if col in df.columns:
        return df[col].astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype="object")


def normalize_split(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
    }
    return s.map(lambda x: mapping.get(x, x))


def finalize_router_manifest(
    df: pd.DataFrame,
    *,
    expert_name: str,
    source_manifest: str,
) -> pd.DataFrame:
    """
    Orden final y columnas estándar del router.
    """
    df = df.copy()

    df["expert_name"] = expert_name
    df["expert_id"] = EXPERT_MAP[expert_name]
    df["source_manifest"] = source_manifest

    # Remap de rutas
    df["input_path"] = df["input_path"].apply(lambda x: remap_project_path(x, PROJECT_ROOT))
    df["input_relpath"] = df["input_path"].apply(lambda x: to_rel_from_project(x, PROJECT_ROOT))
    df["file_exists"] = df["input_path"].apply(lambda x: Path(x).exists() if pd.notna(x) else False)

    # Split limpio
    df["split"] = normalize_split(df["split"])

    # Columnas mínimas estándar
    standard_cols = [
        "sample_id",
        "expert_id",
        "expert_name",
        "dataset_name",
        "split",
        "input_path",
        "input_relpath",
        "task_label",
        "dimension",
        "modality",
        "source_manifest",
        "file_exists",
    ]

    # Agregar cualquier extra útil al final
    extra_cols = [c for c in df.columns if c not in standard_cols]
    df = df[standard_cols + extra_cols]

    return df


# =========================================================
# 3) Builders por dataset
# =========================================================
def build_isic() -> pd.DataFrame:
    path = MANIFESTS_DIR / "manifest_isic_master.csv"
    df = pd.read_csv(path)

    split_col = "split_final" if "split_final" in df.columns else "split"

    out = pd.DataFrame({
        "sample_id": ensure_str_col(df, "image_name"),
        "dataset_name": "isic",
        "split": ensure_str_col(df, split_col),
        "input_path": ensure_str_col(df, "file_path"),
        "task_label": ensure_str_col(df, "target_label"),
        "dimension": ensure_str_col(df, "dimension", "2D"),
        "modality": ensure_str_col(df, "modality", "dermoscopy"),
    })

    return finalize_router_manifest(
        out,
        expert_name="isic",
        source_manifest=path.name,
    )


def build_nih() -> pd.DataFrame:
    path = MANIFESTS_DIR / "manifest_nih_final_with_paths.csv"
    df = pd.read_csv(path)

    split_col = "split_final" if "split_final" in df.columns else "split"

    out = pd.DataFrame({
        "sample_id": ensure_str_col(df, "image_name"),
        "dataset_name": "nih",
        "split": ensure_str_col(df, split_col),
        "input_path": ensure_str_col(df, "file_path"),
        "task_label": ensure_str_col(df, "finding_labels"),
        "dimension": ensure_str_col(df, "dimension", "2D"),
        "modality": ensure_str_col(df, "modality", "xray"),
    })

    return finalize_router_manifest(
        out,
        expert_name="nih",
        source_manifest=path.name,
    )


def build_osteo() -> pd.DataFrame:
    path = MANIFESTS_DIR / "osteo_master_manifest.csv"
    df = pd.read_csv(path)

    split_col = "split_final" if "split_final" in df.columns else "split"

    out = pd.DataFrame({
        "sample_id": ensure_str_col(df, "image_id"),
        "dataset_name": "osteo",
        "split": ensure_str_col(df, split_col),
        "input_path": ensure_str_col(df, "file_path"),
        "task_label": ensure_str_col(df, "target_label"),
        "dimension": "2D",
        "modality": "xray",
    })

    return finalize_router_manifest(
        out,
        expert_name="osteo",
        source_manifest=path.name,
    )


def build_luna16() -> pd.DataFrame:
    path = MANIFESTS_DIR / "split_final_luna16.csv"
    df = pd.read_csv(path)

    out = pd.DataFrame({
        "sample_id": ensure_str_col(df, "seriesuid"),
        "dataset_name": "luna16",
        "split": ensure_str_col(df, "split"),
        "input_path": ensure_str_col(df, "mhd_path"),
        "task_label": ensure_str_col(df, "label_volume_binary"),
        "dimension": "3D",
        "modality": "ct",
    })

    return finalize_router_manifest(
        out,
        expert_name="luna16",
        source_manifest=path.name,
    )


def build_pancreas() -> pd.DataFrame:
    path = MANIFESTS_DIR / "split_final_pancreas.csv"
    df = pd.read_csv(path)

    sample_col = "case_id" if "case_id" in df.columns else "image_name"

    out = pd.DataFrame({
        "sample_id": ensure_str_col(df, sample_col),
        "dataset_name": "pancreas",
        "split": ensure_str_col(df, "split"),
        "input_path": ensure_str_col(df, "image_path"),
        "task_label": ensure_str_col(df, "label_binary"),
        "dimension": "3D",
        "modality": "ct",
    })

    return finalize_router_manifest(
        out,
        expert_name="pancreas",
        source_manifest=path.name,
    )


# =========================================================
# 4) Construcción completa
# =========================================================
def main():
    manifests = {
        "nih": build_nih(),
        "isic": build_isic(),
        "osteo": build_osteo(),
        "luna16": build_luna16(),
        "pancreas": build_pancreas(),
    }

    # Guardar por experto
    for name, df in manifests.items():
        out_path = ROUTER_MANIFESTS_DIR / f"router_manifest_{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"\nGuardado: {out_path}")
        print(df.head(2))

    # Unificado
    all_df = pd.concat(manifests.values(), ignore_index=True)

    # Orden opcional para inspección
    all_df = all_df.sort_values(["expert_id", "split", "sample_id"]).reset_index(drop=True)

    out_all = ROUTER_MANIFESTS_DIR / "router_manifest_all.csv"
    all_df.to_csv(out_all, index=False)

    print("\n" + "=" * 80)
    print("RESUMEN GENERAL")
    print("=" * 80)

    print("\nTotal muestras por experto:")
    print(all_df["expert_name"].value_counts(dropna=False).sort_index())

    print("\nTotal muestras por split:")
    print(all_df["split"].value_counts(dropna=False).sort_index())

    print("\nTabla experto x split:")
    print(pd.crosstab(all_df["expert_name"], all_df["split"], dropna=False))

    print("\nArchivos faltantes por experto:")
    print(all_df.groupby("expert_name")["file_exists"].apply(lambda s: (~s).sum()))

    missing = all_df.loc[~all_df["file_exists"]].copy()
    missing_out = ROUTER_MANIFESTS_DIR / "router_manifest_missing_paths.csv"
    missing.to_csv(missing_out, index=False)

    print(f"\nManifest combinado guardado en: {out_all}")
    print(f"Paths faltantes guardados en: {missing_out}")

    # Explota si faltan archivos
    n_missing = (~all_df["file_exists"]).sum()
    if n_missing > 0:
        raise FileNotFoundError(
            f"Se encontraron {n_missing} rutas inexistentes. "
            f"Revisa router_manifest_missing_paths.csv antes de seguir."
        )

    print("\nTodo bien: no hay rutas faltantes. Ya puedes pasar a extracción de embeddings.")


if __name__ == "__main__":
    main()