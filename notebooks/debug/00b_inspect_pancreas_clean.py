from pathlib import Path
import torch

CKPT_PATH = Path("/home/uaodeepia/luis/proyecto_analitica2/data/working/experts/pancreas/best_pancreas_r3d18.pt")

def clean_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned

def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    print("type(ckpt):", type(ckpt))

    if isinstance(ckpt, dict):
        print("\nTOP-LEVEL KEYS:")
        for k in ckpt.keys():
            print(" ", k)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = clean_state_dict(ckpt["model_state_dict"])
        print("\nUSANDO: model_state_dict")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = clean_state_dict(ckpt["state_dict"])
        print("\nUSANDO: state_dict")
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        sd = clean_state_dict(ckpt)
        print("\nUSANDO: dict tensorial directo")
    else:
        raise ValueError("No pude extraer state_dict correctamente")

    keys = list(sd.keys())
    print("\nN_KEYS:", len(keys))

    print("\nFIRST 40 KEYS:")
    for k in keys[:40]:
        print(" ", k)

    print("\nHEAD CANDIDATES:")
    for k, v in sd.items():
        if any(tag in k for tag in ["fc", "classifier", "head"]):
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {list(v.shape)}")

    print("\nCHECK EXACT KEYS:")
    for target in [
        "fc.weight",
        "fc.bias",
        "base.fc.weight",
        "base.fc.bias",
        "head.weight",
        "head.bias",
        "classifier.weight",
        "classifier.bias",
    ]:
        if target in sd:
            print(f"  FOUND {target}: {list(sd[target].shape)}")

if __name__ == "__main__":
    main()