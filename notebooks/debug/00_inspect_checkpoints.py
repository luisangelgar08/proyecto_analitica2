from pathlib import Path
from collections import Counter
import json

import torch
import torch.nn as nn

from torchvision.models import densenet121, resnet34, resnet18, efficientnet_b3
from torchvision.models.video import r3d_18

try:
    import timm
    HAS_TIMM = True
except Exception:
    HAS_TIMM = False


PROJECT_ROOT = Path("/home/uaodeepia/luis/proyecto_analitica2")
EXPERTS_ROOT = PROJECT_ROOT / "data" / "working" / "experts"

NIH_N_CLASSES = 14
ISIC_N_CLASSES = 8
OSTEO_N_CLASSES = 5
BIN_N_CLASSES = 2


def strip_module_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        out[k[7:]] = v if k.startswith("module.") else v
        if not k.startswith("module."):
            out[k] = v
    return out


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return strip_module_prefix(ckpt["model_state_dict"]), "model_state_dict"
        if "state_dict" in ckpt:
            return strip_module_prefix(ckpt["state_dict"]), "state_dict"
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return strip_module_prefix(ckpt), "raw_state_dict_dict"
    raise ValueError("No pude extraer state_dict del checkpoint.")


def find_one_ckpt(expert_name: str) -> Path:
    expert_dir = EXPERTS_ROOT / expert_name
    if not expert_dir.exists():
        raise FileNotFoundError(f"No existe carpeta de experto: {expert_dir}")

    files = sorted(
        [p for p in expert_dir.rglob("*") if p.is_file() and p.suffix in {".pt", ".pth"}]
    )
    if not files:
        raise FileNotFoundError(f"No encontré checkpoints en: {expert_dir}")
    if len(files) > 1:
        print(f"\n[WARN] {expert_name}: encontré varios checkpoints, usaré el primero:")
        for f in files:
            print("   ", f)
    return files[0]


def build_tv_efficientnet_b3(n_classes: int):
    model = efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, n_classes)
    return model


def build_timm_efficientnet_b3(n_classes: int):
    if not HAS_TIMM:
        raise RuntimeError("timm no está instalado")
    return timm.create_model("efficientnet_b3", pretrained=False, num_classes=n_classes)


def build_densenet121(n_classes: int):
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    return model


def build_resnet34(n_classes: int):
    model = resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def build_resnet18(n_classes: int):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def build_r3d18_binary_1ch():
    model = r3d_18(weights=None)
    model.stem[0] = nn.Conv3d(
        1, 64,
        kernel_size=(3, 7, 7),
        stride=(1, 2, 2),
        padding=(1, 3, 3),
        bias=False
    )
    model.fc = nn.Linear(model.fc.in_features, BIN_N_CLASSES)
    return model


def summarize_keys(state_dict):
    keys = list(state_dict.keys())
    pref1 = Counter(k.split(".")[0] for k in keys)
    pref2 = Counter(".".join(k.split(".")[:2]) for k in keys if "." in k)

    return {
        "n_keys": len(keys),
        "first_keys": keys[:15],
        "top_prefixes_1": pref1.most_common(10),
        "top_prefixes_2": pref2.most_common(15),
    }


def find_head_shapes(state_dict):
    out = {}
    for k, v in state_dict.items():
        if any(tag in k for tag in ["classifier", "fc", "head"]):
            if isinstance(v, torch.Tensor):
                out[k] = list(v.shape)
    return out


def try_load(model, state_dict):
    result = model.load_state_dict(state_dict, strict=False)
    return {
        "missing": len(result.missing_keys),
        "unexpected": len(result.unexpected_keys),
        "total_mismatch": len(result.missing_keys) + len(result.unexpected_keys),
        "missing_sample": result.missing_keys[:10],
        "unexpected_sample": result.unexpected_keys[:10],
    }


def guess_family(keys):
    if any(k.startswith("features.") for k in keys) and any("classifier.1.weight" in k for k in keys):
        return "torchvision_efficientnet"
    if any(k.startswith("features.denseblock") for k in keys) or any("denselayer" in k for k in keys):
        return "torchvision_densenet"
    if any(k.startswith("layer1.") for k in keys) and "fc.weight" in keys:
        return "torchvision_resnet_family"
    if any(k.startswith("stem.") for k in keys) and "fc.weight" in keys:
        return "torchvision_video_r3d_family"
    if any(k.startswith("conv_stem.") for k in keys) and any(k.startswith("blocks.") for k in keys):
        return "timm_efficientnet"
    return "unknown"


def inspect_one(name: str, path: Path):
    print("\n" + "=" * 100)
    print(f"CHECKPOINT: {name}")
    print(path)
    print("=" * 100)

    ckpt = torch.load(path, map_location="cpu")
    state_dict, source_type = extract_state_dict(ckpt)
    keys = list(state_dict.keys())

    print("source_type:", source_type)
    print("guess_family:", guess_family(keys))

    summary = summarize_keys(state_dict)
    print("n_keys:", summary["n_keys"])
    print("first_keys:")
    for k in summary["first_keys"]:
        print(" ", k)

    print("\ntop_prefixes_1:")
    for k, c in summary["top_prefixes_1"]:
        print(f"  {k}: {c}")

    print("\nhead_shapes:")
    for k, shape in find_head_shapes(state_dict).items():
        print(f"  {k}: {shape}")

    candidates = {}
    if name == "isic":
        candidates["torchvision_efficientnet_b3"] = build_tv_efficientnet_b3(ISIC_N_CLASSES)
        candidates["resnet18_8c"] = build_resnet18(ISIC_N_CLASSES)
        candidates["resnet34_8c"] = build_resnet34(ISIC_N_CLASSES)
        if HAS_TIMM:
            candidates["timm_efficientnet_b3"] = build_timm_efficientnet_b3(ISIC_N_CLASSES)

    elif name == "nih":
        candidates["torchvision_densenet121"] = build_densenet121(NIH_N_CLASSES)
        candidates["resnet18_14c"] = build_resnet18(NIH_N_CLASSES)
        candidates["resnet34_14c"] = build_resnet34(NIH_N_CLASSES)

    elif name == "osteo":
        candidates["torchvision_resnet34"] = build_resnet34(OSTEO_N_CLASSES)
        candidates["torchvision_resnet18"] = build_resnet18(OSTEO_N_CLASSES)
        candidates["torchvision_densenet121_5c"] = build_densenet121(OSTEO_N_CLASSES)

    elif name in {"luna16", "pancreas"}:
        candidates["torchvision_r3d18_binary_1ch"] = build_r3d18_binary_1ch()

    scores = {}
    best_name, best_score = None, None
    print("\ncompatibility_scores:")
    for cand_name, model in candidates.items():
        try:
            info = try_load(model, state_dict)
        except Exception as e:
            info = {
                "missing": None,
                "unexpected": None,
                "total_mismatch": 10**9,
                "error": str(e),
            }
        scores[cand_name] = info
        print(f"  {cand_name}: {info}")
        score = info["total_mismatch"]
        if best_score is None or score < best_score:
            best_score = score
            best_name = cand_name

    print("\nBEST_MATCH:", best_name, "| total_mismatch =", best_score)

    return {
        "path": str(path),
        "source_type": source_type,
        "guess_family": guess_family(keys),
        "best_match": best_name,
        "best_score": best_score,
        "scores": scores,
    }


def main():
    expert_names = ["isic", "nih", "osteo", "luna16", "pancreas"]
    results = {}

    for name in expert_names:
        ckpt_path = find_one_ckpt(name)
        results[name] = inspect_one(name, ckpt_path)

    out_path = PROJECT_ROOT / "data" / "working" / "experts" / "checkpoint_inspection_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("RESUMEN FINAL")
    print("=" * 100)
    for name, info in results.items():
        print(f"{name}: {info['best_match']} | mismatch={info['best_score']}")

    print("\nGuardado en:", out_path)


if __name__ == "__main__":
    main()