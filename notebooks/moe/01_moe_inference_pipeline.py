from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from torchvision.models import efficientnet_b3, densenet121, resnet34
from torchvision.models.video import r3d_18


# =========================================================
# 0) Proyecto y configuración base
# =========================================================
def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / "data").exists() and (candidate / "notebooks").exists():
            return candidate
    raise FileNotFoundError("No pude encontrar la raíz del proyecto.")


PROJECT_ROOT = find_project_root(Path.cwd())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Router
ROUTER_MODEL_NAME = "vit_tiny_patch16_224"
ROUTER_IMAGE_SIZE = 224
ROUTER_VOLUME_SIZE = 64

# Checkpoints reales en el clúster
CHECKPOINTS = {
    "router_linear": PROJECT_ROOT / "data" / "working" / "router" / "aux_training" / "vit_tiny_patch16_224_balanced" / "linear_router_aux_alpha_0p02.pt",
    "nih": PROJECT_ROOT / "data" / "working" / "experts" / "nih" / "nih_baseline_densenet121_subset_large.pth",
    "isic": PROJECT_ROOT / "data" / "working" / "experts" / "isic" / "isic_baseline_efficientnet_b3.pt",
    "osteo": PROJECT_ROOT / "data" / "working" / "experts" / "osteo" / "best_model.pt",
    "luna16": PROJECT_ROOT / "data" / "working" / "experts" / "luna16" / "best_luna16_r3d18_v2.pt",
    "pancreas": PROJECT_ROOT / "data" / "working" / "experts" / "pancreas" / "best_pancreas_r3d18.pt",
}

# Router mapping
EXPERT_ID_TO_NAME = {
    0: "nih",
    1: "isic",
    2: "osteo",
    3: "luna16",
    4: "pancreas",
}
EXPERT_NAME_TO_ID = {v: k for k, v in EXPERT_ID_TO_NAME.items()}

# Labels de salida
NIH_CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

ISIC_CLASS_NAMES = [
    "MEL",
    "NV",
    "BCC",
    "AK",
    "BKL",
    "DF",
    "VASC",
    "SCC",
]

OSTEO_CLASS_NAMES = ["0", "1", "2", "3", "4"]
LUNA16_CLASS_NAMES = ["negative", "positive"]
PANCREAS_CLASS_NAMES = ["non-PDAC", "PDAC"]

NIH_THRESHOLD = 0.50
LUNA16_POS_THRESHOLD = 0.56
PANCREAS_POS_THRESHOLD = 0.50

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


# =========================================================
# 1) Utilidades generales
# =========================================================
def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(x.device)
    std = IMAGENET_STD.to(x.device)
    return (x - mean) / std


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


def remove_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def extract_state_dict_from_checkpoint(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj:
            return clean_state_dict(ckpt_obj["model_state_dict"])
        if "state_dict" in ckpt_obj:
            return clean_state_dict(ckpt_obj["state_dict"])
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return clean_state_dict(ckpt_obj)

    raise ValueError("No pude extraer state_dict del checkpoint.")


def load_checkpoint_object(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"No existe checkpoint: {path}")
    return torch.load(path, map_location="cpu")


def strict_load(model: nn.Module, state_dict: Dict[str, torch.Tensor], ckpt_name: str) -> nn.Module:
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    print(f"[LOAD OK] {ckpt_name}")
    return model


def detect_input_dimension(path: Path) -> str:
    """
    Detecta 2D/3D desde el archivo, sin usar metadatos externos.
    """
    try:
        img = Image.open(path)
        arr = np.array(img)
        if arr.ndim in (2, 3):
            return "2D"
    except Exception:
        pass

    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim == 3:
        return "3D"
    if arr.ndim == 2:
        return "2D"

    raise ValueError(f"No pude detectar dimensión válida para: {path}")


# =========================================================
# 2) Preprocesado del router
# =========================================================
def load_router_2d_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)
    x = F.interpolate(
        x.unsqueeze(0),
        size=(ROUTER_IMAGE_SIZE, ROUTER_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    x = normalize_imagenet(x)
    return x


def robust_minmax_normalize(volume: np.ndarray) -> np.ndarray:
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    volume = np.clip(volume, -32768, 32768)

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


def load_router_3d_pseudo_rgb(path: Path) -> torch.Tensor:
    img = sitk.ReadImage(str(path))
    vol = sitk.GetArrayFromImage(img).astype(np.float32)  # [D,H,W]

    vol = robust_minmax_normalize(vol)

    x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
    x = F.interpolate(
        x,
        size=(ROUTER_VOLUME_SIZE, ROUTER_VOLUME_SIZE, ROUTER_VOLUME_SIZE),
        mode="trilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    d, h, w = x.shape
    axial = x[d // 2, :, :]
    coronal = x[:, h // 2, :]
    sagittal = x[:, :, w // 2]

    pseudo = torch.stack([axial, coronal, sagittal], dim=0)
    pseudo = F.interpolate(
        pseudo.unsqueeze(0),
        size=(ROUTER_IMAGE_SIZE, ROUTER_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    pseudo = normalize_imagenet(pseudo)
    return pseudo


# =========================================================
# 3) Preprocesado específico por experto
# =========================================================
def gray_to_3ch_tensor(gray_np: np.ndarray, out_size: int) -> torch.Tensor:
    x = torch.from_numpy(gray_np.astype(np.float32) / 255.0).unsqueeze(0)
    x = F.interpolate(
        x.unsqueeze(0),
        size=(out_size, out_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    x = x.repeat(3, 1, 1)
    x = normalize_imagenet(x)
    return x


def apply_clahe_or_fallback(gray_np: np.ndarray) -> np.ndarray:
    gray_np = gray_np.astype(np.uint8)
    if HAS_CV2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_np)
    pil_img = Image.fromarray(gray_np)
    pil_img = ImageOps.autocontrast(pil_img)
    return np.array(pil_img)


def preprocess_nih(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    gray = np.array(img)
    gray = apply_clahe_or_fallback(gray)
    x = gray_to_3ch_tensor(gray, out_size=224)
    return x.unsqueeze(0)


def preprocess_osteo(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    gray = np.array(img)
    gray = apply_clahe_or_fallback(gray)
    x = gray_to_3ch_tensor(gray, out_size=224)
    return x.unsqueeze(0)


def preprocess_isic(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)
    x = F.interpolate(
        x.unsqueeze(0),
        size=(300, 300),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    x = normalize_imagenet(x)
    return x.unsqueeze(0)


def resize_volume_tensor(vol: np.ndarray, size: Tuple[int, int, int]) -> torch.Tensor:
    x = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    x = F.interpolate(
        x,
        size=size,
        mode="trilinear",
        align_corners=False,
    )
    return x.squeeze(0)  # [1,D,H,W]


def read_ct_volume(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    vol[vol < -1024] = -1024
    return vol


def window_and_normalize(vol: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    vol = np.clip(vol, hu_min, hu_max)
    vol = (vol - hu_min) / (hu_max - hu_min + 1e-8)
    return vol.astype(np.float32)


def preprocess_luna16(path: Path) -> torch.Tensor:
    vol = read_ct_volume(path)
    vol = window_and_normalize(vol, -1000.0, 400.0)
    x = resize_volume_tensor(vol, (64, 64, 64))
    return x.unsqueeze(0)  # [1,1,64,64,64]


def preprocess_pancreas(path: Path) -> torch.Tensor:
    vol = read_ct_volume(path)
    vol = window_and_normalize(vol, -150.0, 250.0)
    x = resize_volume_tensor(vol, (64, 64, 64))
    return x.unsqueeze(0)  # [1,1,64,64,64]


# =========================================================
# 4) Router
# =========================================================
class LinearRouter(nn.Module):
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return self.fc(x)


class RouterInference:
    def __init__(self):
        if not CHECKPOINTS["router_linear"].exists():
            raise FileNotFoundError(f"Falta checkpoint del router: {CHECKPOINTS['router_linear']}")

        # backbone ViT congelado
        self.backbone = timm.create_model(ROUTER_MODEL_NAME, pretrained=True)
        self.backbone.eval().to(DEVICE)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # checkpoint del router entrenado con aux loss
        ckpt = torch.load(CHECKPOINTS["router_linear"], map_location="cpu")

        input_dim = int(ckpt["input_dim"])
        n_classes = int(ckpt["n_classes"])

        self.head = LinearRouter(input_dim, n_classes)
        self.head.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.head.eval().to(DEVICE)

        # scaler guardado durante el entrenamiento del router
        scaler_mean_path = Path(ckpt["scaler_mean_path"])
        scaler_scale_path = Path(ckpt["scaler_scale_path"])

        if not scaler_mean_path.exists():
            raise FileNotFoundError(f"No existe scaler_mean.npy: {scaler_mean_path}")
        if not scaler_scale_path.exists():
            raise FileNotFoundError(f"No existe scaler_scale.npy: {scaler_scale_path}")

        scaler_mean = np.load(scaler_mean_path).astype(np.float32)
        scaler_scale = np.load(scaler_scale_path).astype(np.float32)

        self.scaler_mean = torch.tensor(scaler_mean, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        self.scaler_scale = torch.tensor(scaler_scale, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        print(f"[ROUTER LOAD OK] {CHECKPOINTS['router_linear'].name}")
        print(f"[ROUTER SCALER] mean={scaler_mean_path.name} | scale={scaler_scale_path.name}")

    @torch.no_grad()
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(DEVICE)
        feats = self.backbone.forward_features(x)

        if isinstance(feats, (list, tuple)):
            feats = feats[0]

        if feats.ndim == 3:
            feats = feats[:, 0, :]
        elif feats.ndim == 2:
            pass
        else:
            raise ValueError(f"Forma inesperada de features: {tuple(feats.shape)}")

        return feats.float()

    @torch.no_grad()
    def predict(self, input_path: Path) -> Dict[str, Any]:
        dimension = detect_input_dimension(input_path)

        if dimension == "2D":
            x = load_router_2d_image(input_path).unsqueeze(0)
        else:
            x = load_router_3d_pseudo_rgb(input_path).unsqueeze(0)

        # embedding del ViT
        z = self.extract_embedding(x)

        # estandarización EXACTA usada en el entrenamiento del router aux
        z_std = (z - self.scaler_mean) / (self.scaler_scale + 1e-8)

        logits = self.head(z_std)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_id = int(np.argmax(probs))
        pred_name = EXPERT_ID_TO_NAME[pred_id]

        return {
            "router_dimension_detected": dimension,
            "router_embedding_dim": int(z.shape[1]),
            "router_pred_id": pred_id,
            "router_pred_name": pred_name,
            "router_probs": {
                EXPERT_ID_TO_NAME[i]: float(probs[i]) for i in range(len(probs))
            },
        }

# =========================================================
# 5) Builders correctos por checkpoint
# =========================================================
def build_nih_model() -> nn.Module:
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(NIH_CLASS_NAMES))
    return model


def build_isic_model() -> nn.Module:
    model = efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(ISIC_CLASS_NAMES))
    return model


def build_osteo_model() -> nn.Module:
    model = resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(OSTEO_CLASS_NAMES))
    return model


def build_luna16_model() -> nn.Module:
    model = r3d_18(weights=None)
    model.stem[0] = nn.Conv3d(
        1,
        64,
        kernel_size=(3, 7, 7),
        stride=(1, 2, 2),
        padding=(1, 3, 3),
        bias=False,
    )
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def build_pancreas_model() -> nn.Module:
    model = r3d_18(weights=None)
    model.stem[0] = nn.Conv3d(
        1,
        64,
        kernel_size=(3, 7, 7),
        stride=(1, 2, 2),
        padding=(1, 3, 3),
        bias=False,
    )
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


# =========================================================
# 6) Carga específica de checkpoints por experto
# =========================================================
def load_nih_model() -> nn.Module:
    model = build_nih_model()
    ckpt_obj = load_checkpoint_object(CHECKPOINTS["nih"])
    state_dict = extract_state_dict_from_checkpoint(ckpt_obj)
    return strict_load(model, state_dict, CHECKPOINTS["nih"].name)


def load_isic_model() -> nn.Module:
    model = build_isic_model()
    ckpt_obj = load_checkpoint_object(CHECKPOINTS["isic"])
    state_dict = extract_state_dict_from_checkpoint(ckpt_obj)
    return strict_load(model, state_dict, CHECKPOINTS["isic"].name)


def load_osteo_model() -> nn.Module:
    model = build_osteo_model()
    ckpt_obj = load_checkpoint_object(CHECKPOINTS["osteo"])
    state_dict = extract_state_dict_from_checkpoint(ckpt_obj)
    return strict_load(model, state_dict, CHECKPOINTS["osteo"].name)


def load_luna16_model() -> nn.Module:
    model = build_luna16_model()
    ckpt_obj = load_checkpoint_object(CHECKPOINTS["luna16"])
    state_dict = extract_state_dict_from_checkpoint(ckpt_obj)
    state_dict = remove_prefix_if_present(state_dict, "base.")
    return strict_load(model, state_dict, CHECKPOINTS["luna16"].name)


def load_pancreas_model() -> nn.Module:
    model = build_pancreas_model()
    ckpt_obj = load_checkpoint_object(CHECKPOINTS["pancreas"])
    state_dict = extract_state_dict_from_checkpoint(ckpt_obj)
    state_dict = remove_prefix_if_present(state_dict, "base.")
    return strict_load(model, state_dict, CHECKPOINTS["pancreas"].name)


class ExpertRegistry:
    def __init__(self):
        self.cache: Dict[str, nn.Module] = {}

    def get(self, expert_name: str) -> nn.Module:
        if expert_name in self.cache:
            return self.cache[expert_name]

        if expert_name == "nih":
            model = load_nih_model()
        elif expert_name == "isic":
            model = load_isic_model()
        elif expert_name == "osteo":
            model = load_osteo_model()
        elif expert_name == "luna16":
            model = load_luna16_model()
        elif expert_name == "pancreas":
            model = load_pancreas_model()
        else:
            raise ValueError(f"Experto no soportado: {expert_name}")

        self.cache[expert_name] = model
        return model


# =========================================================
# 7) Predicción por experto
# =========================================================
@torch.no_grad()
def predict_nih(model: nn.Module, input_path: Path) -> Dict[str, Any]:
    x = preprocess_nih(input_path).to(DEVICE)
    logits = model(x)
    probs = torch.sigmoid(logits).cpu().numpy()[0]

    positive = []
    for label, p in zip(NIH_CLASS_NAMES, probs):
        if p >= NIH_THRESHOLD:
            positive.append({"label": label, "prob": float(p)})

    if not positive:
        top_idx = np.argsort(-probs)[:3]
        positive = [{"label": NIH_CLASS_NAMES[i], "prob": float(probs[i])} for i in top_idx]

    return {
        "task_type": "multilabel",
        "threshold": NIH_THRESHOLD,
        "predictions": positive,
    }


@torch.no_grad()
def predict_multiclass(model: nn.Module, x: torch.Tensor, class_names: List[str]) -> Dict[str, Any]:
    x = x.to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return {
        "task_type": "multiclass",
        "pred_idx": pred_idx,
        "pred_label": class_names[pred_idx],
        "probs": {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    }


@torch.no_grad()
def predict_luna16(model: nn.Module, input_path: Path) -> Dict[str, Any]:
    x = preprocess_luna16(input_path).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pos_prob = float(probs[1])
    pred_idx = 1 if pos_prob >= LUNA16_POS_THRESHOLD else 0

    return {
        "task_type": "binary_softmax_with_threshold",
        "positive_class": LUNA16_CLASS_NAMES[1],
        "threshold": LUNA16_POS_THRESHOLD,
        "pred_idx": pred_idx,
        "pred_label": LUNA16_CLASS_NAMES[pred_idx],
        "probs": {
            LUNA16_CLASS_NAMES[0]: float(probs[0]),
            LUNA16_CLASS_NAMES[1]: float(probs[1]),
        }
    }


@torch.no_grad()
def predict_pancreas(model: nn.Module, input_path: Path) -> Dict[str, Any]:
    x = preprocess_pancreas(input_path).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pos_prob = float(probs[1])
    pred_idx = 1 if pos_prob >= PANCREAS_POS_THRESHOLD else 0

    return {
        "task_type": "binary_softmax_with_threshold",
        "positive_class": PANCREAS_CLASS_NAMES[1],
        "threshold": PANCREAS_POS_THRESHOLD,
        "pred_idx": pred_idx,
        "pred_label": PANCREAS_CLASS_NAMES[pred_idx],
        "probs": {
            PANCREAS_CLASS_NAMES[0]: float(probs[0]),
            PANCREAS_CLASS_NAMES[1]: float(probs[1]),
        }
    }


def run_expert_prediction(registry: ExpertRegistry, expert_name: str, input_path: Path) -> Dict[str, Any]:
    model = registry.get(expert_name)

    if expert_name == "nih":
        return predict_nih(model, input_path)

    if expert_name == "isic":
        x = preprocess_isic(input_path)
        return predict_multiclass(model, x, ISIC_CLASS_NAMES)

    if expert_name == "osteo":
        x = preprocess_osteo(input_path)
        return predict_multiclass(model, x, OSTEO_CLASS_NAMES)

    if expert_name == "luna16":
        return predict_luna16(model, input_path)

    if expert_name == "pancreas":
        return predict_pancreas(model, input_path)

    raise ValueError(f"Experto no soportado: {expert_name}")


# =========================================================
# 8) Pipeline MoE
# =========================================================
class MoEInferencePipeline:
    def __init__(self):
        self.router = RouterInference()
        self.experts = ExpertRegistry()

    def predict(self, input_path: str) -> Dict[str, Any]:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"No existe input_path: {input_path}")

        router_out = self.router.predict(input_path)
        selected_expert = router_out["router_pred_name"]

        try:
            expert_out = run_expert_prediction(self.experts, selected_expert, input_path)
            expert_status = "ok"
        except Exception as e:
            expert_out = {
                "error": str(e),
            }
            expert_status = "error"

        return {
            "input_path": str(input_path),
            "router": router_out,
            "selected_expert": selected_expert,
            "expert_status": expert_status,
            "expert_output": expert_out,
        }


# =========================================================
# 9) CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="MoE inference pipeline")
    parser.add_argument("--input-path", type=str, required=True, help="Ruta a la imagen o volumen")
    parser.add_argument("--pretty", action="store_true", help="Imprimir JSON bonito")
    args = parser.parse_args()

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DEVICE:", DEVICE)
    print("Router checkpoint:", CHECKPOINTS["router_linear"])

    pipeline = MoEInferencePipeline()
    result = pipeline.predict(args.input_path)

    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()