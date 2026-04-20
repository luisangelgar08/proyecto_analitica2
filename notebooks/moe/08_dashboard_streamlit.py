from __future__ import annotations

import io
import json
import math
import tempfile
import time
import importlib.util
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import SimpleITK as sitk
import torch
import torch.nn.functional as F


# =========================================================
# Config
# =========================================================
PROJECT_ROOT = Path("/home/uaodeepia/luis/proyecto_analitica2")
MOE_SCRIPT = PROJECT_ROOT / "notebooks" / "moe" / "01_moe_inference_pipeline.py"

FINAL_BUNDLE_DIR = PROJECT_ROOT / "data" / "working" / "final_results_bundle"
MOE_FINAL_EVAL_DIR = PROJECT_ROOT / "data" / "working" / "moe" / "final" / "eval"
ROUTER_OOD_DIR = PROJECT_ROOT / "data" / "working" / "moe_eval" / "router_ood_entropy"
ABlation_DIR = PROJECT_ROOT / "data" / "working" / "router" / "ablation" / "vit_tiny_patch16_224_balanced"

TMP_UPLOAD_DIR = PROJECT_ROOT / "data" / "working" / "dashboard_uploads"
TMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="MoE Médico Dashboard",
    page_icon="🩺",
    layout="wide",
)

# =========================================================
# Load module + pipeline
# =========================================================
@st.cache_resource
def load_moe_module():
    spec = importlib.util.spec_from_file_location("moe_pipeline_module", MOE_SCRIPT)
    moe_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(moe_module)
    return moe_module


@st.cache_resource
def load_pipeline():
    moe_module = load_moe_module()
    pipeline = moe_module.MoEInferencePipeline()
    return moe_module, pipeline


moe_module, pipeline = load_pipeline()


# =========================================================
# Session state
# =========================================================
if "routing_counts" not in st.session_state:
    st.session_state.routing_counts = {
        "nih": 0,
        "isic": 0,
        "osteo": 0,
        "luna16": 0,
        "pancreas": 0,
    }

if "history_rows" not in st.session_state:
    st.session_state.history_rows = []


# =========================================================
# Helpers: file + shapes
# =========================================================
def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix
    if uploaded_file.name.endswith(".nii.gz"):
        suffix = ".nii.gz"

    out_path = TMP_UPLOAD_DIR / f"upload_{int(time.time() * 1000)}_{uploaded_file.name}"
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


def detect_original_shape(path: Path) -> Tuple[str, Tuple[int, ...]]:
    try:
        img = Image.open(path)
        arr = np.array(img)
        if arr.ndim in (2, 3):
            return "2D", tuple(arr.shape)
    except Exception:
        pass

    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim == 3:
        return "3D", tuple(arr.shape)
    if arr.ndim == 2:
        return "2D", tuple(arr.shape)

    raise ValueError(f"No pude detectar forma original para: {path}")


def get_router_adapted_shape(dimension: str) -> Tuple[int, ...]:
    if dimension == "2D":
        return (3, moe_module.ROUTER_IMAGE_SIZE, moe_module.ROUTER_IMAGE_SIZE)
    return (3, moe_module.ROUTER_IMAGE_SIZE, moe_module.ROUTER_IMAGE_SIZE)


def get_original_display_image(path: Path) -> Image.Image:
    dim, _ = detect_original_shape(path)

    if dim == "2D":
        img = Image.open(path).convert("RGB")
        return img

    img = sitk.ReadImage(str(path))
    vol = sitk.GetArrayFromImage(img)  # [D,H,W]
    z = vol.shape[0] // 2
    sl = vol[z].astype(np.float32)

    p1, p99 = np.percentile(sl, 1.0), np.percentile(sl, 99.0)
    if p99 <= p1:
        p1, p99 = float(sl.min()), float(sl.max())
    sl = np.clip(sl, p1, p99)
    sl = (sl - p1) / (p99 - p1 + 1e-8)
    sl = (sl * 255).astype(np.uint8)

    return Image.fromarray(sl).convert("RGB")


def get_router_input_tensor(path: Path) -> Tuple[str, torch.Tensor]:
    dimension = moe_module.detect_input_dimension(path)
    if dimension == "2D":
        x = moe_module.load_router_2d_image(path).unsqueeze(0)
    else:
        x = moe_module.load_router_3d_pseudo_rgb(path).unsqueeze(0)
    return dimension, x


# =========================================================
# Helpers: heatmap aprox. ViT
# =========================================================
@torch.no_grad()
def build_router_heatmap(path: Path) -> Image.Image:
    """
    Heatmap aproximado basado en similitud CLS <-> patch tokens.
    Es útil como visual explicativa aunque no sea el mapa bruto de atención.
    """
    router = pipeline.router
    _, x = get_router_input_tensor(path)
    x = x.to(moe_module.DEVICE)

    feats = router.backbone.forward_features(x)
    if isinstance(feats, (list, tuple)):
        feats = feats[0]

    if feats.ndim != 3:
        raise ValueError(f"Se esperaban tokens [B,N,D], recibí: {tuple(feats.shape)}")

    cls_token = feats[:, 0, :]           # [1, D]
    patch_tokens = feats[:, 1:, :]       # [1, Np, D]

    cls_norm = F.normalize(cls_token, dim=-1)
    patch_norm = F.normalize(patch_tokens, dim=-1)

    sim = torch.einsum("bd,bnd->bn", cls_norm, patch_norm)[0]  # [Np]
    sim = sim.float().cpu().numpy()

    n_patches = sim.shape[0]
    grid = int(math.sqrt(n_patches))
    if grid * grid != n_patches:
        raise ValueError(f"No puedo formar grilla cuadrada con {n_patches} patches")

    heat = sim.reshape(grid, grid)
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-8)

    original = get_original_display_image(path)
    orig_w, orig_h = original.size

    heat_img = Image.fromarray((heat * 255).astype(np.uint8))
    heat_img = heat_img.resize((orig_w, orig_h), Image.BILINEAR)

    cmap = plt.get_cmap("jet")
    heat_rgba = cmap(np.array(heat_img) / 255.0)
    heat_rgb = (heat_rgba[:, :, :3] * 255).astype(np.uint8)
    heat_pil = Image.fromarray(heat_rgb).convert("RGBA")

    original_rgba = original.convert("RGBA")
    overlay = Image.blend(original_rgba, heat_pil, alpha=0.35).convert("RGB")
    return overlay


# =========================================================
# Helpers: result loading
# =========================================================
def safe_read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_final_metrics():
    final_metrics = safe_read_csv(FINAL_BUNDLE_DIR / "final_metrics_table.csv")
    routing_summary = safe_read_csv(FINAL_BUNDLE_DIR / "final_routing_summary_by_expert.csv")
    expert_summary = safe_read_csv(FINAL_BUNDLE_DIR / "final_expert_summary_by_expert.csv")
    ood_summary = safe_read_json(ROUTER_OOD_DIR / "router_ood_entropy_summary.json")
    moe_summary = safe_read_json(MOE_FINAL_EVAL_DIR / "moe_balanced_test_summary.json")
    bundle_summary = safe_read_json(FINAL_BUNDLE_DIR / "final_results_bundle_summary.json")
    return final_metrics, routing_summary, expert_summary, ood_summary, moe_summary, bundle_summary


def load_ablation_table() -> Optional[pd.DataFrame]:
    candidates = list(ABlation_DIR.glob("*.csv"))
    for p in candidates:
        try:
            df = pd.read_csv(p)
            if "model_name" in df.columns:
                return df
        except Exception:
            pass
    return None


final_metrics, routing_summary_df, expert_summary_df, ood_summary, moe_summary, bundle_summary = load_final_metrics()
ablation_df = load_ablation_table()


# =========================================================
# Helpers: display
# =========================================================
def probs_to_df(prob_dict: Dict[str, float], value_name="prob"):
    return pd.DataFrame({
        "label": list(prob_dict.keys()),
        value_name: list(prob_dict.values()),
    }).sort_values(value_name, ascending=False).reset_index(drop=True)


def draw_bar_dataframe(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(df[x_col], df[y_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)


def draw_routing_counts():
    counts = st.session_state.routing_counts
    df = pd.DataFrame({
        "expert": list(counts.keys()),
        "count": list(counts.values()),
    })
    draw_bar_dataframe(
        df,
        "expert",
        "count",
        "Conteo manual de inferencias en esta sesión"
    )

##--------------------------
def extract_final_router_fi(final_metrics_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if final_metrics_df is None or len(final_metrics_df) == 0:
        return None

    df = final_metrics_df.copy()
    mask = (
        (df["component"] == "router_final")
        & (df["metric_name"].astype(str).str.startswith("test_fi_"))
    )
    df = df[mask].copy()

    if len(df) == 0:
        return None

    df["expert"] = df["metric_name"].str.replace("test_fi_", "", regex=False)
    df["fi"] = df["metric_value"].astype(float)
    df = df[["expert", "fi"]].sort_values("expert").reset_index(drop=True)
    return df
##-------------------------

def compute_entropy(prob_dict: Dict[str, float]) -> float:
    probs = np.array(list(prob_dict.values()), dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-(probs * np.log(probs)).sum())


def expert_arch_name(expert_name: str) -> str:
    mapping = {
        "nih": "DenseNet121",
        "isic": "EfficientNet-B3",
        "osteo": "ResNet34",
        "luna16": "R3D-18 3D",
        "pancreas": "R3D-18 3D",
    }
    return mapping.get(expert_name, "N/A")


def expert_dataset_name(expert_name: str) -> str:
    mapping = {
        "nih": "NIH ChestX-ray14",
        "isic": "ISIC 2019",
        "osteo": "Osteoarthritis",
        "luna16": "LUNA16",
        "pancreas": "Pancreatic Cancer",
    }
    return mapping.get(expert_name, "N/A")

def mem_gb(x: int) -> float:
    return float(x / (1024 ** 3))

def cuda_stats_now():
    if not torch.cuda.is_available():
        return {
            "alloc_gb": 0.0,
            "reserved_gb": 0.0,
            "peak_alloc_gb": 0.0,
            "peak_reserved_gb": 0.0,
        }

    return {
        "alloc_gb": mem_gb(torch.cuda.memory_allocated()),
        "reserved_gb": mem_gb(torch.cuda.memory_reserved()),
        "peak_alloc_gb": mem_gb(torch.cuda.max_memory_allocated()),
        "peak_reserved_gb": mem_gb(torch.cuda.max_memory_reserved()),
    }

def reset_cuda_peak():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# =========================================================
# Sidebar
# =========================================================
st.sidebar.title("🩺 Dashboard MoE")
st.sidebar.markdown("Router ViT + expertos heterogéneos")
st.sidebar.markdown("---")

source_mode = st.sidebar.radio(
    "Fuente de entrada",
    ["Subir archivo", "Usar ruta existente del clúster"],
    index=0,
)

example_paths = {
    "ISIC ejemplo": str(PROJECT_ROOT / "data" / "source" / "isic" / "MEL" / "ISIC_0000004.jpg"),
    "NIH ejemplo": str(PROJECT_ROOT / "data" / "source" / "nih" / "images_001" / "images" / "00000003_000.png"),
    "Osteo ejemplo": str(PROJECT_ROOT / "data" / "source" / "osteo" / "KLGrade" / "KLGrade" / "0" / "0_111_png.rf.feb50161b2231dc9244620846a45de3e.jpg"),
    "LUNA16 ejemplo": str(PROJECT_ROOT / "data" / "source" / "luna16" / "subset5" / "1.3.6.1.4.1.14519.5.2.1.6279.6001.100332161840553388986847034053.mhd"),
    "Pancreas ejemplo": str(PROJECT_ROOT / "data" / "source" / "pancreas" / "100007_00001_0000.nii.gz"),
}

selected_path: Optional[Path] = None

if source_mode == "Subir archivo":
    uploaded = st.sidebar.file_uploader(
        "Sube PNG / JPG / JPEG / NIfTI (.nii, .nii.gz)",
        type=["png", "jpg", "jpeg", "nii", "gz"],
    )
    if uploaded is not None:
        selected_path = save_uploaded_file(uploaded)
else:
    picked_example = st.sidebar.selectbox("Ejemplo rápido", ["(ninguno)"] + list(example_paths.keys()))
    path_text = st.sidebar.text_input("Ruta del archivo", value=example_paths.get(picked_example, ""))
    if path_text.strip():
        selected_path = Path(path_text.strip())

run_button = st.sidebar.button("Ejecutar inferencia", use_container_width=True)

st.title("🧠 Mixture of Experts Médico")
st.caption("Dashboard final del router ViT + expertos 2D/3D")

tab1, tab2, tab3 = st.tabs(["Inferencia", "Ablation", "Balance + OOD"])


# =========================================================
# TAB 1: Inferencia
# =========================================================
with tab1:
    st.subheader("Inferencia interactiva")

    if selected_path is None:
        st.info("Selecciona una entrada en la barra lateral.")
    else:
        st.write(f"**Archivo actual:** `{selected_path}`")

        if run_button:
            if not selected_path.exists():
                st.error("La ruta no existe.")
            else:
                try:
                    original_dim, original_shape = detect_original_shape(selected_path)
                    adapted_shape = get_router_adapted_shape(original_dim)

                    sync_cuda()
                    reset_cuda_peak()
                    vram_before = cuda_stats_now()

                    t0 = time.perf_counter()
                    result = pipeline.predict(str(selected_path))
                    sync_cuda()
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    vram_after = cuda_stats_now()
                    
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    router_probs = result["router"]["router_probs"]
                    router_pred = result["router"]["router_pred_name"]
                    router_conf = max(router_probs.values())
                    router_entropy = compute_entropy(router_probs)

                    st.session_state.history_rows.append({
                        "path": str(selected_path),
                        "router_pred": router_pred,
                        "router_conf": router_conf,
                        "router_entropy": router_entropy,
                        "elapsed_ms": elapsed_ms,
                        "vram_reserved_before_gb": vram_before["reserved_gb"],
                        "vram_reserved_after_gb": vram_after["reserved_gb"],
                        "vram_peak_reserved_gb": vram_after["peak_reserved_gb"],
                    })

                    c1, c2 = st.columns([1, 1])

                    with c1:
                        st.markdown("### Preprocesado")
                        st.write(f"**Dimensión detectada:** {original_dim}")
                        st.write(f"**Shape original:** {original_shape}")
                        st.write(f"**Shape adaptada al router:** {adapted_shape}")

                        original_img = get_original_display_image(selected_path)
                        st.image(original_img, caption="Imagen / slice original", use_container_width=True)

                    with c2:
                        st.markdown("### Router")

                        ood_flag = False
                        thr = None
                        if ood_summary is not None:
                            thr = ood_summary["entropy_threshold_p95_id"]
                            ood_flag = router_entropy >= thr

                        if ood_flag:
                            st.metric("Estado de entrada", "OOD / no válida")
                        else:
                            st.metric("Experto activado", router_pred)

                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("Confianza router", f"{router_conf:.4f}")
                            st.metric("Tiempo total", f"{elapsed_ms:.2f} ms")
                            st.metric("Entropía router", f"{router_entropy:.4f}")

                        with m2:
                            st.metric("VRAM reservada antes", f"{vram_before['reserved_gb']:.3f} GB")
                            st.metric("VRAM reservada después", f"{vram_after['reserved_gb']:.3f} GB")
                            st.metric("Pico VRAM reservada", f"{vram_after['peak_reserved_gb']:.3f} GB")

                        if ood_summary is not None:
                            if ood_flag:
                                st.error(
                                    f"⚠ Entrada fuera del dominio del sistema.\n\n"
                                    f"La entropía del router ({router_entropy:.4f}) superó el umbral calibrado ({thr:.4f}).\n"
                                    f"Esta imagen no pertenece a los dominios médicos soportados, por lo que la salida no debe interpretarse clínicamente."
                                )
                            else:
                                st.success(f"Sin alerta OOD (umbral {thr:.4f})")

                        router_df = probs_to_df(router_probs)
                        st.dataframe(router_df, use_container_width=True, hide_index=True)

                        st.markdown("### Heatmap del router")
                        st.caption("Mapa aproximado del ViT basado en similitud CLS ↔ parches")
                        try:
                            heatmap_overlay = build_router_heatmap(selected_path)
                            hc1, hc2, hc3 = st.columns([1, 2, 1])
                            with hc2:
                                st.image(
                                    heatmap_overlay,
                                    caption="Heatmap del router sobre la imagen/slice",
                                    use_container_width=True
                            )
                        except Exception as e:
                            st.warning(f"No pude construir el heatmap: {e}")

                    st.markdown("### Experto activado")
                    selected_expert = result["selected_expert"]
                    expert_output = result["expert_output"]
                    expert_status = result["expert_status"]

                    if ood_flag:
                        st.warning(
                            "La entrada fue marcada como OOD. "
                            "Por seguridad, la salida del experto no se presenta como predicción válida."
                        )

                        with st.expander("Ver salida técnica cruda del sistema"):
                            ec1, ec2 = st.columns([1, 1])
                            with ec1:
                                st.write(f"**Experto seleccionado internamente:** {selected_expert}")
                                st.write(f"**Arquitectura:** {expert_arch_name(selected_expert)}")
                                st.write(f"**Dataset de origen:** {expert_dataset_name(selected_expert)}")
                                st.write(f"**Gating score:** {router_conf:.4f}")
                                st.write(f"**Estado experto:** {expert_status}")
                            with ec2:
                                st.json(expert_output)

                            if isinstance(expert_output, dict) and "probs" in expert_output:
                                exp_df = probs_to_df(expert_output["probs"], value_name="prob")
                                st.dataframe(exp_df, use_container_width=True, hide_index=True)

                    else:
                        ec1, ec2 = st.columns([1, 1])
                        with ec1:
                            st.write(f"**Nombre:** {selected_expert}")
                            st.write(f"**Arquitectura:** {expert_arch_name(selected_expert)}")
                            st.write(f"**Dataset de origen:** {expert_dataset_name(selected_expert)}")
                            st.write(f"**Gating score:** {router_conf:.4f}")
                            st.write(f"**Estado experto:** {expert_status}")

                        with ec2:
                            st.json(expert_output)

                        if isinstance(expert_output, dict) and "probs" in expert_output:
                            exp_df = probs_to_df(expert_output["probs"], value_name="prob")
                            st.dataframe(exp_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.exception(e)


# =========================================================
# TAB 2: Ablation
# =========================================================
with tab2:
    st.subheader("Ablation histórico del router")
    st.caption("Comparación entre candidatos del router. El router oficial final del sistema se resume más abajo en 'Resultados oficiales finales del router'.")
# =========================================================
# TAB 2: Ablation
# =========================================================
with tab2:
    st.subheader("Ablation histórico del router")
    st.caption("Comparación entre candidatos del router. El router oficial final del sistema se resume más abajo en 'Resultados finales del router'.")
    if ablation_df is not None:
        st.markdown("### Tabla comparativa")
        st.dataframe(ablation_df, use_container_width=True, hide_index=True)

        numeric_acc_col = None
        for col in ["test_acc", "val_acc"]:
            if col in ablation_df.columns:
                numeric_acc_col = col
                break

        if numeric_acc_col is not None:
            draw_bar_dataframe(
                ablation_df[["model_name", numeric_acc_col]].copy(),
                "model_name",
                numeric_acc_col,
                f"Ablation - {numeric_acc_col}",
            )
    else:
        st.warning("No encontré la tabla del ablation automáticamente. Si quieres, luego la conectamos a una ruta exacta.")

    st.markdown("### Resultados oficiales finales del router")
    if final_metrics is not None:
        st.dataframe(final_metrics, use_container_width=True, hide_index=True)
    else:
        st.info("No encontré final_metrics_table.csv")

    fig1 = FINAL_BUNDLE_DIR / "figure_router_train_ratio.png"
    fig2 = FINAL_BUNDLE_DIR / "figure_router_val_ratio.png"
    fig3 = FINAL_BUNDLE_DIR / "figure_router_fi_train.png"
    fig4 = FINAL_BUNDLE_DIR / "figure_router_fi_val.png"

    for p, title in [
        (fig1, "Balance ratio train"),
        (fig2, "Balance ratio val"),
        (fig3, "fi train"),
        (fig4, "fi val"),
    ]:
        if p.exists():
            st.image(str(p), caption=title, use_container_width=True)


# =========================================================
# TAB 3: Balance + OOD
# =========================================================
with tab3:
    st.subheader("Balance de carga y OOD")

    # -------------------------------------------------
    # Conteo manual de la sesión
    # -------------------------------------------------
    st.markdown("### Conteo manual de inferencias en esta sesión")
    st.caption("Este gráfico refleja únicamente las inferencias que tú has ejecutado manualmente dentro del dashboard actual.")
    draw_routing_counts()

    # -------------------------------------------------
    # Balance oficial final del router
    # -------------------------------------------------
    st.markdown("### Balance oficial final del router")
    st.caption("Este sí corresponde al balance formal del router final sobre el conjunto de evaluación.")

    final_fi_df = extract_final_router_fi(final_metrics)

    if final_fi_df is not None:
        draw_bar_dataframe(
            final_fi_df,
            "expert",
            "fi",
            "Distribución final fi por experto"
        )

        fi_max = float(final_fi_df["fi"].max())
        fi_min = float(final_fi_df["fi"].min())
        fi_ratio = fi_max / fi_min if fi_min > 0 else np.inf

        c1, c2, c3 = st.columns(3)
        c1.metric("max(fi)", f"{fi_max:.4f}")
        c2.metric("min(fi)", f"{fi_min:.4f}")
        c3.metric("max(fi)/min(fi)", f"{fi_ratio:.4f}")

        if fi_ratio < 1.30:
            st.success("Balance final correcto: max(fi)/min(fi) < 1.30")
        else:
            st.error("Balance final fuera de rango: max(fi)/min(fi) ≥ 1.30")
    else:
        st.info("No encontré los valores test_fi_* dentro de final_metrics_table.csv")

    # -------------------------------------------------
    # Resúmenes finales
    # -------------------------------------------------
    if routing_summary_df is not None:
        st.markdown("### Routing summary final")
        st.dataframe(routing_summary_df, use_container_width=True, hide_index=True)

    if expert_summary_df is not None:
        st.markdown("### Expert summary final")
        st.dataframe(expert_summary_df, use_container_width=True, hide_index=True)

    routing_cm_img = FINAL_BUNDLE_DIR / "figure_routing_confusion_matrix.png"
    if routing_cm_img.exists():
        st.image(str(routing_cm_img), caption="Matriz de confusión del routing", use_container_width=True)

    # -------------------------------------------------
    # OOD
    # -------------------------------------------------
    st.markdown("### OOD del router")