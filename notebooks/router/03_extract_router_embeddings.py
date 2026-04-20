from pathlib import Path
import json
import math
import random

import numpy as np
import pandas as pd
from PIL import Image
import SimpleITK as sitk

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm


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
ROUTER_DIR = WORKING_DIR / "router"
ROUTER_MANIFESTS_DIR = ROUTER_DIR / "manifests"

INPUT_MANIFEST = ROUTER_MANIFESTS_DIR / "router_manifest_balanced_all.csv"

OUTPUT_DIR = ROUTER_DIR / "embeddings" / "vit_tiny_patch16_224_balanced"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("INPUT_MANIFEST:", INPUT_MANIFEST)
print("OUTPUT_DIR:", OUTPUT_DIR)


# =========================================================
# 1) Configuración
# =========================================================
SEED = 42
MODEL_NAME = "vit_tiny_patch16_224"
IMAGE_SIZE = 224
VOLUME_SIZE = 64

BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


# =========================================================
# 2) Semillas
# =========================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# =========================================================
# 3) Helpers de preprocesado
# =========================================================
def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def load_2d_image(path: str) -> torch.Tensor:
    """
    Devuelve tensor [3, 224, 224] float32 normalizado.
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC, [0,1]
    x = torch.from_numpy(arr).permute(2, 0, 1)        # [3,H,W]
    x = F.interpolate(
        x.unsqueeze(0),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)
    x = normalize_imagenet(x)
    return x


def robust_minmax_normalize(volume: np.ndarray) -> np.ndarray:
    """
    Normalización robusta y agnóstica a modalidad:
    usa percentiles del propio volumen.
    """
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # saneamiento básico de valores absurdos
    volume = np.where(volume < -32768, -32768, volume)
    volume = np.where(volume > 32768, 32768, volume)

    p1 = np.percentile(volume, 1.0)
    p99 = np.percentile(volume, 99.0)

    if not np.isfinite(p1):
        p1 = float(volume.min())
    if not np.isfinite(p99):
        p99 = float(volume.max())

    if p99 <= p1:
        p1 = float(volume.min())
        p99 = float(volume.max())

    if p99 <= p1:
        return np.zeros_like(volume, dtype=np.float32)

    volume = np.clip(volume, p1, p99)
    volume = (volume - p1) / (p99 - p1 + 1e-8)
    return volume.astype(np.float32)


def volume_to_pseudo_rgb(path: str) -> torch.Tensor:
    """
    Lee volumen 3D y lo convierte a pseudo-imagen RGB [3,224,224]
    usando cortes centrales axial/coronal/sagital.
    """
    img = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(img).astype(np.float32)  # [D,H,W]

    vol = robust_minmax_normalize(vol)

    x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)   # [1,1,D,H,W]
    x = F.interpolate(
        x,
        size=(VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE),
        mode="trilinear",
        align_corners=False
    ).squeeze(0).squeeze(0)  # [D,H,W]

    d, h, w = x.shape
    axial = x[d // 2, :, :]      # [H,W]
    coronal = x[:, h // 2, :]    # [D,W]
    sagittal = x[:, :, w // 2]   # [D,H]

    # Cada corte se vuelve canal
    pseudo = torch.stack([axial, coronal, sagittal], dim=0)  # [3,*,*]
    pseudo = F.interpolate(
        pseudo.unsqueeze(0),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    pseudo = normalize_imagenet(pseudo)
    return pseudo


# =========================================================
# 4) Dataset
# =========================================================
class RouterEmbeddingDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        input_path = str(row["input_path"])
        dimension = str(row["dimension"]).strip().upper()

        if dimension == "2D":
            image = load_2d_image(input_path)
        elif dimension == "3D":
            image = volume_to_pseudo_rgb(input_path)
        else:
            raise ValueError(f"Dimension no soportada: {dimension}")

        sample = {
            "image": image,  # [3,224,224]
            "expert_id": int(row["expert_id"]),
            "sample_id": str(row["sample_id"]),
            "expert_name": str(row["expert_name"]),
            "dataset_name": str(row["dataset_name"]),
            "split": str(row["split"]),
            "dimension": str(row["dimension"]),
            "modality": str(row["modality"]),
            "task_label": str(row["task_label"]),
            "input_path": str(row["input_path"]),
            "input_relpath": str(row["input_relpath"]) if "input_relpath" in row else "",
        }
        return sample


def collate_router(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    expert_ids = torch.tensor([b["expert_id"] for b in batch], dtype=torch.long)

    meta = {
        "sample_id": [b["sample_id"] for b in batch],
        "expert_name": [b["expert_name"] for b in batch],
        "dataset_name": [b["dataset_name"] for b in batch],
        "split": [b["split"] for b in batch],
        "dimension": [b["dimension"] for b in batch],
        "modality": [b["modality"] for b in batch],
        "task_label": [b["task_label"] for b in batch],
        "input_path": [b["input_path"] for b in batch],
        "input_relpath": [b["input_relpath"] for b in batch],
    }
    return images, expert_ids, meta


# =========================================================
# 5) Modelo
# =========================================================
def build_backbone(model_name: str):
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_features(model, x: torch.Tensor) -> torch.Tensor:
    """
    Para ViT:
    - si forward_features devuelve [B,N,D], tomamos CLS [:,0,:]
    - si devuelve [B,D], usamos eso directamente
    """
    x = x.to(DEVICE, non_blocking=True)

    if DEVICE == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            feats = model.forward_features(x)
    else:
        feats = model.forward_features(x)

    if isinstance(feats, (list, tuple)):
        feats = feats[0]

    if feats.ndim == 3:
        feats = feats[:, 0, :]  # CLS token
    elif feats.ndim == 2:
        pass
    else:
        raise ValueError(f"Forma inesperada de features: {tuple(feats.shape)}")

    return feats.float().cpu()


# =========================================================
# 6) Extracción por split
# =========================================================
def run_split(model, df_split: pd.DataFrame, split_name: str):
    print("\n" + "=" * 80)
    print(f"EXTRAYENDO SPLIT: {split_name}")
    print("=" * 80)
    print(f"Muestras: {len(df_split)}")

    dataset = RouterEmbeddingDataset(df_split)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_router,
    )

    all_z = []
    all_y = []
    all_meta_rows = []

    for batch_idx, (images, expert_ids, meta) in enumerate(loader):
        z = extract_features(model, images)

        all_z.append(z.numpy())
        all_y.append(expert_ids.numpy())

        batch_size_actual = len(meta["sample_id"])
        for i in range(batch_size_actual):
            all_meta_rows.append({
                "sample_id": meta["sample_id"][i],
                "expert_id": int(expert_ids[i].item()),
                "expert_name": meta["expert_name"][i],
                "dataset_name": meta["dataset_name"][i],
                "split": meta["split"][i],
                "dimension": meta["dimension"][i],
                "modality": meta["modality"][i],
                "task_label": meta["task_label"][i],
                "input_path": meta["input_path"][i],
                "input_relpath": meta["input_relpath"][i],
            })

        if batch_idx == 0:
            print(f"Primer batch imágenes: {tuple(images.shape)}")
            print(f"Primer batch embeddings: {tuple(z.shape)}")

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(f"Batch {batch_idx + 1}/{len(loader)} completado")

    Z = np.concatenate(all_z, axis=0)
    y = np.concatenate(all_y, axis=0)
    meta_df = pd.DataFrame(all_meta_rows)

    print(f"Embeddings finales {split_name}: {Z.shape}")
    print(f"Labels finales {split_name}: {y.shape}")
    print(meta_df["expert_name"].value_counts().sort_index())

    np.save(OUTPUT_DIR / f"Z_{split_name}.npy", Z)
    np.save(OUTPUT_DIR / f"y_{split_name}_expert.npy", y)
    meta_df.to_csv(OUTPUT_DIR / f"meta_{split_name}.csv", index=False)

    return Z, y, meta_df


# =========================================================
# 7) Main
# =========================================================
def main():
    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(f"No existe el manifest: {INPUT_MANIFEST}")

    df = pd.read_csv(INPUT_MANIFEST)
    print(f"\nManifest leído: {INPUT_MANIFEST}")
    print(f"Shape: {df.shape}")

    print("\nDistribución por split:")
    print(df["split"].value_counts().sort_index())

    print("\nDistribución por experto:")
    print(df["expert_name"].value_counts().sort_index())

    model = build_backbone(MODEL_NAME)
    print(f"\nModelo cargado: {MODEL_NAME}")
    print(f"DEVICE: {DEVICE}")

    split_dfs = {
        "train": df[df["split"] == "train"].reset_index(drop=True),
        "val": df[df["split"] == "val"].reset_index(drop=True),
        "test": df[df["split"] == "test"].reset_index(drop=True),
    }

    results = {}
    for split_name, df_split in split_dfs.items():
        Z, y, meta_df = run_split(model, df_split, split_name)
        results[split_name] = {
            "n_samples": int(len(df_split)),
            "embedding_dim": int(Z.shape[1]),
        }

    summary = {
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "image_size": IMAGE_SIZE,
        "volume_size": VOLUME_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "results": results,
    }

    with open(OUTPUT_DIR / "embedding_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    expert_map_df = (
        df[["expert_id", "expert_name"]]
        .drop_duplicates()
        .sort_values("expert_id")
        .reset_index(drop=True)
    )
    expert_map_df.to_csv(OUTPUT_DIR / "expert_id_map.csv", index=False)

    print("\n" + "=" * 80)
    print("EXTRACCIÓN TERMINADA")
    print("=" * 80)
    print(f"Archivos guardados en: {OUTPUT_DIR}")
    print("Se generaron:")
    print("- Z_train.npy / y_train_expert.npy / meta_train.csv")
    print("- Z_val.npy / y_val_expert.npy / meta_val.csv")
    print("- Z_test.npy / y_test_expert.npy / meta_test.csv")
    print("- embedding_summary.json")
    print("- expert_id_map.csv")


if __name__ == "__main__":
    main()