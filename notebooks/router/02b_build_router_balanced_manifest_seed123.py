from pathlib import Path
import pandas as pd


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

SEED = 123
INPUT_MANIFEST = ROUTER_MANIFESTS_DIR / "router_manifest_all.csv"

EXPECTED_SPLITS = {"train", "val", "test"}
EXPECTED_EXPERTS = {"nih", "isic", "osteo", "luna16", "pancreas"}


def sample_group(df_group: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df_group.sample(n=n, replace=False, random_state=seed)


def main():
    df = pd.read_csv(INPUT_MANIFEST)

    counts = (
        df.groupby(["expert_name", "split"])
        .size()
        .reset_index(name="n_samples")
        .sort_values(["split", "expert_name"])
        .reset_index(drop=True)
    )

    print("\nConteos originales por experto x split:")
    print(counts)

    target_rows = []
    for split in ["train", "val", "test"]:
        auto_min = int(counts.loc[counts["split"] == split, "n_samples"].min())
        target_rows.append({
            "split": split,
            "target_n_per_expert": auto_min,
        })
    target_df = pd.DataFrame(target_rows)

    sampled_parts = []
    for split in ["train", "val", "test"]:
        target_n = int(target_df.loc[target_df["split"] == split, "target_n_per_expert"].iloc[0])

        for expert_name in sorted(EXPECTED_EXPERTS):
            part = df[(df["split"] == split) & (df["expert_name"] == expert_name)].copy()
            local_seed = SEED + hash((split, expert_name)) % 100000
            sampled = sample_group(part, n=target_n, seed=local_seed).copy()
            sampled["balanced_target_n"] = target_n
            sampled["balanced_seed"] = SEED
            sampled_parts.append(sampled)

    balanced_df = pd.concat(sampled_parts, ignore_index=True)
    balanced_df = balanced_df.sort_values(["split", "expert_id", "sample_id"]).reset_index(drop=True)

    out_all = ROUTER_MANIFESTS_DIR / "router_manifest_balanced_all_seed123.csv"
    out_train = ROUTER_MANIFESTS_DIR / "router_manifest_balanced_train_seed123.csv"
    out_val = ROUTER_MANIFESTS_DIR / "router_manifest_balanced_val_seed123.csv"
    out_test = ROUTER_MANIFESTS_DIR / "router_manifest_balanced_test_seed123.csv"
    out_summary = ROUTER_MANIFESTS_DIR / "router_manifest_balanced_summary_seed123.csv"

    balanced_df.to_csv(out_all, index=False)
    balanced_df[balanced_df["split"] == "train"].to_csv(out_train, index=False)
    balanced_df[balanced_df["split"] == "val"].to_csv(out_val, index=False)
    balanced_df[balanced_df["split"] == "test"].to_csv(out_test, index=False)

    summary = (
        balanced_df.groupby(["expert_name", "split"])
        .size()
        .reset_index(name="n_samples")
        .sort_values(["split", "expert_name"])
        .reset_index(drop=True)
    )
    summary.to_csv(out_summary, index=False)

    print("\nRESUMEN BALANCEADO seed=123")
    print(summary)
    print("\nTabla experto x split:")
    print(pd.crosstab(balanced_df["expert_name"], balanced_df["split"], dropna=False))
    print(f"\nGuardado: {out_all}")


if __name__ == "__main__":
    main()