import csv
import io
import os
import random
import time

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from torchvision import transforms

from model import CatDogCNN, CatDogTransfer

st.set_page_config(
    page_title="Cat vs Dog Vision Studio",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")

# Custom CSS for clean black-and-white monochrome cards, badges, and controls
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        color: #000000;
        border-right: 1px solid #000000;
    }
    [data-testid="stSidebar"] * {
        color: #000000;
    }
    .stButton > button[kind="primary"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #000000 !important;
    }
    .stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
    }
    [data-testid="stProgress"] > div > div > div > div {
        background-color: #000000 !important;
    }
    .hero-header {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        background: #000000;
        color: #FFFFFF;
        border: 1px solid #000000;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0;
        color: #FFFFFF;
    }
    .hero-subtitle {
        margin: 0.25rem 0 0 0;
        font-size: 0.95rem;
        color: #FFFFFF;
    }
    .result-card {
        padding: 1rem 1.15rem;
        border-radius: 12px;
        border: 1.5px solid #000000;
        background: #FFFFFF;
        color: #000000;
        margin-bottom: 0.75rem;
    }
    .mono-notice {
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: 1.5px solid #000000;
        background: #FFFFFF;
        color: #000000;
        margin-bottom: 0.75rem;
        font-size: 0.92rem;
    }
    .mono-notice-inverted {
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: 1.5px solid #000000;
        background: #000000;
        color: #FFFFFF;
        margin-bottom: 0.75rem;
        font-size: 0.92rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-solid {
        background: #000000;
        color: #FFFFFF;
        border: 1px solid #000000;
    }
    .badge-outline {
        background: #FFFFFF;
        color: #000000;
        border: 1.5px dashed #000000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def discover_checkpoints():
    """Find available trained model checkpoints in the project directory."""
    candidates = [
        ("transfer", "MobileNetV3 (Fine-tuned)", os.path.join(BASE_DIR, "best_model.pth")),
        ("scratch", "Custom CNN (From Scratch)", os.path.join(BASE_DIR, "best_model_scratch.pth")),
    ]
    found = {}
    for key, label, path in candidates:
        if os.path.exists(path):
            found[key] = {"label": label, "path": path}
    return found


@st.cache_resource
def load_model(model_path):
    """Load a PyTorch checkpoint and return the model along with metadata."""
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint.get("classes", ["cats", "dogs"])
    state_dict = checkpoint["model_state_dict"]

    is_transfer = checkpoint.get("model_type") == "transfer" or "model.classifier.3.weight" in state_dict

    if is_transfer:
        model = CatDogTransfer(num_classes=len(classes), pretrained=False).to(device)
        img_size = (224, 224)
        model_name = "MobileNetV3 (Fine-tuned)"
        arch_key = "transfer"
    else:
        model = CatDogCNN(num_classes=len(classes)).to(device)
        img_size = (150, 150)
        model_name = "Custom CNN (From Scratch)"
        arch_key = "scratch"

    model.load_state_dict(state_dict)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    return {
        "model": model,
        "device": device,
        "img_size": img_size,
        "model_name": model_name,
        "arch_key": arch_key,
        "classes": classes,
        "val_acc": checkpoint.get("val_acc"),
        "epoch": checkpoint.get("epoch"),
        "param_count": param_count,
    }


def build_transform(img_size):
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def apply_image_adjustments(image, rotation=0, brightness=1.0, contrast=1.0, flip_h=False, zoom=1.0):
    """Apply interactive user transformations before running inference."""
    img = image.copy()
    if flip_h:
        img = ImageOps.mirror(img)
    if zoom > 1.0:
        w, h = img.size
        crop_w, crop_h = w / zoom, h / zoom
        left = (w - crop_w) / 2
        top = (h - crop_h) / 2
        img = img.crop((left, top, left + crop_w, top + crop_h)).resize((w, h), Image.Resampling.BICUBIC)
    if rotation != 0:
        img = img.rotate(-rotation, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(20, 20, 24))
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def jet_colormap(cam_norm):
    """Map a normalized [0, 1] 2D array to an RGB heatmap without matplotlib overhead."""
    r = np.clip(1.5 - np.abs(4.0 * cam_norm - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * cam_norm - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * cam_norm - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def compute_gradcam(model_bundle, image, target_class_idx, alpha=0.45):
    """Generate a Grad-CAM heatmap overlay highlighting regions driving the prediction."""
    model = model_bundle["model"]
    device = model_bundle["device"]
    img_size = model_bundle["img_size"]
    arch_key = model_bundle["arch_key"]

    target_layer = model.model.features[-1] if arch_key == "transfer" else model.features[-1]
    activations = []
    gradients = []

    def fwd_hook(_mod, _inp, out):
        activations.append(out.detach())

    def bwd_hook(_mod, _grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    try:
        tensor = build_transform(img_size)(image).unsqueeze(0).to(device)
        tensor.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        score = logits[0, target_class_idx]
        score.backward()

        act = activations[0][0]  # (C, H, W)
        grad = gradients[0][0]   # (C, H, W)
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=0)).cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        heatmap_rgb = Image.fromarray(jet_colormap(cam)).resize(image.size, Image.Resampling.BILINEAR)
        overlay = Image.blend(image.convert("RGB"), heatmap_rgb, alpha=alpha)
        return overlay
    finally:
        h_fwd.remove()
        h_bwd.remove()


def run_inference(model_bundle, image, with_gradcam=False, cam_alpha=0.45):
    """Run forward pass and return probabilities, timing, and optional Grad-CAM."""
    model = model_bundle["model"]
    device = model_bundle["device"]
    img_size = model_bundle["img_size"]

    tensor = build_transform(img_size)(image).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    cat_prob = float(probs[0].item())
    dog_prob = float(probs[1].item())
    target_idx = 0 if cat_prob >= dog_prob else 1

    cam_img = None
    if with_gradcam:
        cam_img = compute_gradcam(model_bundle, image, target_idx, alpha=cam_alpha)

    return {
        "cat_prob": cat_prob,
        "dog_prob": dog_prob,
        "predicted": "Cat" if target_idx == 0 else "Dog",
        "confidence": max(cat_prob, dog_prob),
        "latency_ms": latency_ms,
        "gradcam": cam_img,
    }


def list_validation_images():
    """Collect validation image paths with their ground-truth labels if available."""
    val_root = os.path.join(BASE_DIR, "data", "cats_and_dogs_filtered", "validation")
    items = []
    for folder, label in [("cats", "Cat"), ("dogs", "Dog")]:
        folder_path = os.path.join(val_root, folder)
        if os.path.isdir(folder_path):
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(VALID_EXTENSIONS) and not fname.startswith("."):
                    items.append((os.path.join(folder_path, fname), label, fname))
    return items


def record_history(entry):
    """Append or update prediction history in session state without duplicate consecutive spam."""
    history = st.session_state.setdefault("history", [])
    if history and history[-1]["signature"] == entry["signature"]:
        return
    history.append(entry)
    if len(history) > 100:
        history.pop(0)


def history_to_csv(rows):
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["timestamp", "source", "model", "prediction", "confidence_pct", "cat_pct", "dog_pct", "status", "feedback"],
    )
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
    return buf.getvalue()


def render_notice(text, inverted=False):
    cls = "mono-notice-inverted" if inverted else "mono-notice"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def render_prediction_card(title, model_bundle, result, threshold, ground_truth=None):
    """Render an interactive prediction summary card for a single model."""
    conf_pct = result["confidence"] * 100.0
    is_uncertain = conf_pct < threshold
    is_cat = result["predicted"] == "Cat"
    pred_emoji = "🐱" if is_cat else "🐶"
    badge_cls = "badge-outline" if is_uncertain else "badge-solid"
    status_text = (
        f"⚠️ Low Confidence / Possible Out-of-Distribution (< {threshold}%)"
        if is_uncertain
        else f"✅ High Confidence (≥ {threshold}%)"
    )

    gt_html = ""
    if ground_truth:
        correct = (not is_uncertain) and (result["predicted"] == ground_truth)
        gt_badge = "badge-solid" if correct else "badge-outline"
        gt_label = f"Ground Truth: {ground_truth} ({'Match' if correct else 'Mismatch'})"
        gt_html = f'<span class="badge {gt_badge}">{gt_label}</span>'

    st.markdown(
        f"""
        <div class="result-card">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                <span style="font-weight:700; font-size:1.05rem;">{title}</span>
                <span style="font-size:0.82rem;">{model_bundle['device'].type.upper()} • {result['latency_ms']:.1f} ms</span>
            </div>
            <div style="margin-top:0.55rem;">
                <span class="badge {badge_cls}">{pred_emoji} {result['predicted']} ({conf_pct:.1f}%)</span>
                {gt_html}
            </div>
            <div style="margin-top:0.35rem; font-size:0.85rem;">{status_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("🐱 Cat", f"{result['cat_prob'] * 100:.1f}%")
        st.progress(result["cat_prob"])
    with c2:
        st.metric("🐶 Dog", f"{result['dog_prob'] * 100:.1f}%")
        st.progress(result["dog_prob"])


def reset_playground():
    st.session_state["pg_rotation"] = 0
    st.session_state["pg_brightness"] = 1.0
    st.session_state["pg_contrast"] = 1.0
    st.session_state["pg_zoom"] = 1.0
    st.session_state["pg_flip"] = False


# --- App Initialization ---
st.markdown(
    """
    <div class="hero-header">
        <p class="hero-title">🐾 Cat vs Dog Vision Studio</p>
        <p class="hero-subtitle">Interactive image classification, live augmentation playground, Grad-CAM explainability, and side-by-side model comparison.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

checkpoints = discover_checkpoints()
if not checkpoints:
    st.warning("⚠️ No trained model weights (`best_model.pth` or `best_model_scratch.pth`) were found. Run training first:")
    st.code("python train_transfer.py", language="bash")
    st.stop()

# Initialize session defaults
st.session_state.setdefault("history", [])
st.session_state.setdefault("pg_rotation", 0)
st.session_state.setdefault("pg_brightness", 1.0)
st.session_state.setdefault("pg_contrast", 1.0)
st.session_state.setdefault("pg_zoom", 1.0)
st.session_state.setdefault("pg_flip", False)

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Studio Controls")

    mode_options = []
    if len(checkpoints) >= 2:
        mode_options.append("Compare Both Models")
    for k, info in checkpoints.items():
        mode_options.append(info["label"])

    selected_mode = st.radio(
        "Inference Mode",
        options=mode_options,
        help="Run a single model or pit MobileNetV3 against the Custom Scratch CNN side-by-side.",
    )

    confidence_threshold = st.slider(
        "Confidence Threshold (%)",
        min_value=50,
        max_value=99,
        value=75,
        step=1,
        help="Predictions below this confidence are flagged as Uncertain / Out-of-Distribution (e.g., non-cat/dog images).",
    )

    st.divider()
    st.subheader("🔥 Explainability (Grad-CAM)")
    show_gradcam = st.toggle("Show Grad-CAM Heatmap", value=True, help="Highlight image regions influencing the model's decision.")
    cam_alpha = st.slider("Heatmap Opacity", min_value=0.15, max_value=0.80, value=0.45, step=0.05, disabled=not show_gradcam)

    st.divider()
    st.subheader("🧾 Session Summary")
    sidebar_history_slot = st.empty()

# Load active models based on selection
active_models = {}
if selected_mode == "Compare Both Models":
    for k, info in checkpoints.items():
        active_models[k] = load_model(info["path"])
else:
    for k, info in checkpoints.items():
        if info["label"] == selected_mode:
            active_models[k] = load_model(info["path"])

val_items = list_validation_images()

# --- Main Interactive Tabs ---
tab_single, tab_batch, tab_insights = st.tabs(
    ["🎯 Interactive Classifier", "📦 Batch Explorer", "📊 Model Insights & History"]
)

with tab_single:
    left_col, right_col = st.columns([1.05, 1.25], gap="large")

    with left_col:
        st.subheader("1. Choose or Capture an Image")
        input_method = st.segmented_control(
            "Input Source",
            options=["📁 Upload", "🖼️ Samples", "🎲 Validation Explorer", "📷 Webcam"],
            default="🖼️ Samples",
            label_visibility="collapsed",
        )
        if input_method is None:
            input_method = "🖼️ Samples"

        raw_image = None
        image_label = None
        ground_truth = None

        if input_method == "📁 Upload":
            uploaded = st.file_uploader(
                "Drop a Cat, Dog, or any test image here",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
            )
            if uploaded is not None:
                raw_image = Image.open(uploaded).convert("RGB")
                image_label = uploaded.name

        elif input_method == "🖼️ Samples":
            built_in_samples = [
                ("🐱 Kitten", os.path.join(BASE_DIR, "test_data", "Unknown.webp"), "Cat"),
                ("🐶 Dog", os.path.join(BASE_DIR, "test_data", "dog.webp"), "Dog"),
                ("🐍 Snake (OOD)", os.path.join(BASE_DIR, "test_data", "snake.jpeg"), None),
            ]
            available_samples = [s for s in built_in_samples if os.path.exists(s[1])]
            if available_samples:
                sample_cols = st.columns(len(available_samples))
                st.session_state.setdefault("selected_sample_idx", 0)
                for idx, (s_label, s_path, _gt) in enumerate(available_samples):
                    with sample_cols[idx]:
                        st.image(Image.open(s_path).convert("RGB"), width="stretch")
                        btn_type = "primary" if st.session_state["selected_sample_idx"] == idx else "secondary"
                        if st.button(s_label, key=f"sample_btn_{idx}", type=btn_type, width="stretch"):
                            st.session_state["selected_sample_idx"] = idx
                            st.rerun()

                chosen = available_samples[st.session_state["selected_sample_idx"]]
                raw_image = Image.open(chosen[1]).convert("RGB")
                image_label = os.path.basename(chosen[1])
                ground_truth = chosen[2]
            else:
                render_notice("No files found in <code>test_data/</code>. Please upload an image.")

        elif input_method == "🎲 Validation Explorer":
            if not val_items:
                render_notice("Validation dataset not found locally (excluded on cloud deployments). Use Upload, Samples, or Webcam instead.")
            else:
                if "val_sample_idx" not in st.session_state:
                    st.session_state["val_sample_idx"] = random.randrange(len(val_items))

                c_btn1, c_btn2 = st.columns(2)
                with c_btn1:
                    if st.button("🎲 Random Image", type="primary", width="stretch"):
                        st.session_state["val_sample_idx"] = random.randrange(len(val_items))
                with c_btn2:
                    if st.button("🔀 Pick Challenging (Next)", width="stretch"):
                        st.session_state["val_sample_idx"] = (st.session_state["val_sample_idx"] + 17) % len(val_items)

                v_path, v_gt, v_name = val_items[st.session_state["val_sample_idx"]]
                raw_image = Image.open(v_path).convert("RGB")
                image_label = f"val/{v_name}"
                ground_truth = v_gt
                st.caption(f"Inspecting `{v_name}` (Ground Truth hidden in badge after prediction)")

        elif input_method == "📷 Webcam":
            cam_photo = st.camera_input("Take a snapshot to classify live")
            if cam_photo is not None:
                raw_image = Image.open(cam_photo).convert("RGB")
                image_label = "webcam_capture.jpg"

        if raw_image is not None:
            with st.expander("🎛️ Live Image Playground (Test Robustness)", expanded=False):
                st.caption("Adjust rotation, zoom, lighting, or flip to see how predictions & Grad-CAM react in real time.")
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    rotation = st.slider("Rotation (°)", -180, 180, key="pg_rotation")
                    brightness = st.slider("Brightness", 0.4, 1.8, step=0.05, key="pg_brightness")
                with p_col2:
                    zoom = st.slider("Center Zoom", 1.0, 2.0, step=0.05, key="pg_zoom")
                    contrast = st.slider("Contrast", 0.4, 1.8, step=0.05, key="pg_contrast")
                f_col1, f_col2 = st.columns([1, 1])
                with f_col1:
                    flip_h = st.checkbox("Mirror Horizontal", key="pg_flip")
                with f_col2:
                    st.button("↺ Reset Adjustments", on_click=reset_playground, width="stretch")

            processed_image = apply_image_adjustments(
                raw_image,
                rotation=st.session_state["pg_rotation"],
                brightness=st.session_state["pg_brightness"],
                contrast=st.session_state["pg_contrast"],
                flip_h=st.session_state["pg_flip"],
                zoom=st.session_state["pg_zoom"],
            )

            st.image(processed_image, caption=f"Model Input: {image_label} ({processed_image.size[0]}×{processed_image.size[1]}px)", width="stretch")

    with right_col:
        st.subheader("2. Live Classification & Explainability")
        if raw_image is None:
            render_notice("👈 Select a sample, upload an image, or snap a webcam photo on the left to start classifying.")
        else:
            results = {}
            for m_key, bundle in active_models.items():
                res = run_inference(bundle, processed_image, with_gradcam=show_gradcam, cam_alpha=cam_alpha)
                results[m_key] = res

                conf_pct = res["confidence"] * 100.0
                status = "Uncertain/OOD" if conf_pct < confidence_threshold else "Confident"
                sig = f"{image_label}|{bundle['model_name']}|{st.session_state['pg_rotation']}|{st.session_state['pg_brightness']:.2f}|{st.session_state['pg_contrast']:.2f}|{st.session_state['pg_zoom']:.2f}|{st.session_state['pg_flip']}"
                record_history(
                    {
                        "timestamp": time.strftime("%H:%M:%S"),
                        "source": image_label,
                        "model": bundle["model_name"],
                        "prediction": res["predicted"],
                        "confidence_pct": f"{conf_pct:.1f}%",
                        "cat_pct": f"{res['cat_prob'] * 100:.1f}%",
                        "dog_pct": f"{res['dog_prob'] * 100:.1f}%",
                        "status": status,
                        "feedback": "",
                        "signature": sig,
                    }
                )

            if len(results) == 2 and "transfer" in results and "scratch" in results:
                t_res = results["transfer"]
                s_res = results["scratch"]
                if t_res["predicted"] != s_res["predicted"]:
                    render_notice(
                        f"⚡ <strong>Model Disagreement:</strong> MobileNetV3 predicts <strong>{t_res['predicted']}</strong> ({t_res['confidence']*100:.1f}%), "
                        f"while Custom CNN predicts <strong>{s_res['predicted']}</strong> ({s_res['confidence']*100:.1f}%).",
                        inverted=False,
                    )
                else:
                    render_notice(
                        f"🤝 <strong>Models Agree:</strong> Both models predict <strong>{t_res['predicted']}</strong>.",
                        inverted=True,
                    )

            for m_key, bundle in active_models.items():
                render_prediction_card(bundle["model_name"], bundle, results[m_key], confidence_threshold, ground_truth=ground_truth)

            if show_gradcam:
                st.markdown("#### 🔥 Grad-CAM Attention Maps")
                st.caption("Warmer regions (red/yellow) had the strongest influence on the predicted class.")
                cam_cols = st.columns(len(active_models))
                for idx, (m_key, bundle) in enumerate(active_models.items()):
                    with cam_cols[idx]:
                        st.image(
                            results[m_key]["gradcam"],
                            caption=f"{bundle['model_name']} → {results[m_key]['predicted']}",
                            width="stretch",
                        )

            # Interactive feedback buttons for latest logged entry
            st.markdown("#### 💬 Was this prediction helpful?")
            fb1, fb2, fb3 = st.columns([1, 1, 2])
            with fb1:
                if st.button("👍 Accurate", width="stretch"):
                    if st.session_state["history"]:
                        st.session_state["history"][-1]["feedback"] = "👍 Accurate"
                    st.toast("Thanks! Feedback saved to session history.", icon="✅")
            with fb2:
                if st.button("👎 Inaccurate", width="stretch"):
                    if st.session_state["history"]:
                        st.session_state["history"][-1]["feedback"] = "👎 Inaccurate"
                    st.toast("Noted! Marked as inaccurate in history.", icon="📝")

with tab_batch:
    st.subheader("📦 Batch Image Classifier")
    st.caption("Classify multiple uploaded images at once, scan `test_data/`, or sample random validation images.")

    batch_ctrl1, batch_ctrl2, batch_ctrl3 = st.columns([2, 1, 1])
    with batch_ctrl1:
        batch_files = st.file_uploader(
            "Upload multiple images",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
            key="batch_uploader",
        )
    with batch_ctrl2:
        st.write("")
        st.write("")
        scan_test_dir = st.button("📂 Run on `test_data/`", width="stretch")
    with batch_ctrl3:
        st.write("")
        st.write("")
        run_val_sample = st.button(
            "🎲 Sample 8 Validation Images",
            disabled=not val_items,
            help="Requires local data/cats_and_dogs_filtered/validation folder" if not val_items else None,
            width="stretch",
        )

    # Select primary model for batch mode
    batch_model_key = "transfer" if "transfer" in active_models else next(iter(active_models))
    batch_bundle = active_models[batch_model_key]
    st.caption(f"Batch Inference Model: **{batch_bundle['model_name']}**")

    if scan_test_dir:
        test_dir = os.path.join(BASE_DIR, "test_data")
        items = []
        if os.path.isdir(test_dir):
            for fname in sorted(os.listdir(test_dir)):
                if fname.lower().endswith(VALID_EXTENSIONS) and not fname.startswith("."):
                    items.append((fname, Image.open(os.path.join(test_dir, fname)).convert("RGB"), None))
        st.session_state["batch_items"] = items
    elif run_val_sample and val_items:
        picked = random.sample(val_items, min(8, len(val_items)))
        st.session_state["batch_items"] = [
            (fname, Image.open(fpath).convert("RGB"), gt) for fpath, gt, fname in picked
        ]
    elif batch_files:
        st.session_state["batch_items"] = [
            (f.name, Image.open(f).convert("RGB"), None) for f in batch_files
        ]

    batch_items = st.session_state.get("batch_items", [])
    if batch_items:
        batch_results = []
        progress = st.progress(0, text="Running batch classification...")
        for i, (fname, img, gt) in enumerate(batch_items):
            res = run_inference(batch_bundle, img, with_gradcam=False)
            conf_pct = res["confidence"] * 100.0
            status = "Uncertain" if conf_pct < confidence_threshold else res["predicted"]
            batch_results.append(
                {
                    "name": fname,
                    "image": img,
                    "gt": gt,
                    "res": res,
                    "status": status,
                }
            )
            progress.progress((i + 1) / len(batch_items), text=f"Classified {i + 1}/{len(batch_items)}: {fname}")
        progress.empty()

        # Summary metrics
        n_total = len(batch_results)
        n_cats = sum(1 for r in batch_results if r["status"] == "Cat")
        n_dogs = sum(1 for r in batch_results if r["status"] == "Dog")
        n_unc = sum(1 for r in batch_results if r["status"] == "Uncertain")
        avg_conf = sum(r["res"]["confidence"] * 100.0 for r in batch_results) / n_total

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Images", n_total)
        m2.metric("🐱 Cats", n_cats)
        m3.metric("🐶 Dogs", n_dogs)
        m4.metric("⚠️ Uncertain", n_unc)
        m5.metric("Avg Confidence", f"{avg_conf:.1f}%")

        filter_choice = st.segmented_control(
            "Filter Results",
            options=["All", "Cat", "Dog", "Uncertain"],
            default="All",
        )
        filtered = [r for r in batch_results if filter_choice in (None, "All") or r["status"] == filter_choice]

        # Download CSV button
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["Filename", "Prediction", "Status", "Confidence %", "Cat %", "Dog %", "Ground Truth"])
        for r in batch_results:
            writer.writerow(
                [
                    r["name"],
                    r["res"]["predicted"],
                    r["status"],
                    f"{r['res']['confidence']*100:.2f}",
                    f"{r['res']['cat_prob']*100:.2f}",
                    f"{r['res']['dog_prob']*100:.2f}",
                    r["gt"] or "-",
                ]
            )
        st.download_button(
            "⬇️ Download Batch Results (CSV)",
            data=csv_buf.getvalue(),
            file_name="batch_predictions.csv",
            mime="text/csv",
        )

        # Render responsive 4-column gallery
        grid_cols = st.columns(4)
        for idx, item in enumerate(filtered):
            with grid_cols[idx % 4]:
                st.image(item["image"], width="stretch")
                conf = item["res"]["confidence"] * 100.0
                pred = item["res"]["predicted"]
                emoji = "⚠️" if item["status"] == "Uncertain" else ("🐱" if pred == "Cat" else "🐶")
                gt_txt = f" | GT: {item['gt']}" if item["gt"] else ""
                st.markdown(f"**{item['name']}**\n\n{emoji} **{item['status']}** ({conf:.1f}%){gt_txt}")
    else:
        render_notice("Upload multiple images above or click <strong>Run on test_data/</strong> or <strong>Sample 8 Validation Images</strong> to classify a batch.")

with tab_insights:
    st.subheader("📊 Model Architecture & Checkpoint Comparison")

    comp_rows = []
    for k, info in checkpoints.items():
        b = load_model(info["path"])
        comp_rows.append(
            {
                "Model": b["model_name"],
                "Checkpoint": os.path.basename(info["path"]),
                "Input Size": f"{b['img_size'][0]}×{b['img_size'][1]}",
                "Parameters": f"{b['param_count']:,}",
                "Best Epoch": b["epoch"] if b["epoch"] is not None else "-",
                "Validation Accuracy": f"{b['val_acc'] * 100:.2f}%" if b["val_acc"] is not None else "-",
                "Device": b["device"].type.upper(),
            }
        )
    st.dataframe(comp_rows, width="stretch", hide_index=True)

    history_img = os.path.join(BASE_DIR, "training_history.png")
    if os.path.exists(history_img):
        with st.expander("📈 View Scratch CNN Training History Plot", expanded=False):
            st.image(history_img, caption="Training & Validation Accuracy/Loss Curve", width="stretch")

    st.divider()
    st.subheader("🧾 Session Prediction History")
    hist_rows = st.session_state.get("history", [])
    if not hist_rows:
        render_notice("No predictions logged yet in this session. Classify an image in the first tab to populate history.")
    else:
        display_rows = [
            {
                "Time": r["timestamp"],
                "Image": r["source"],
                "Model": r["model"],
                "Prediction": r["prediction"],
                "Confidence": r["confidence_pct"],
                "Cat %": r["cat_pct"],
                "Dog %": r["dog_pct"],
                "Status": r["status"],
                "Feedback": r["feedback"] or "-",
            }
            for r in reversed(hist_rows)
        ]
        st.dataframe(display_rows, width="stretch", hide_index=True)
        st.download_button(
            "⬇️ Export Session History (CSV)",
            data=history_to_csv(hist_rows),
            file_name="prediction_history.csv",
            mime="text/csv",
        )

# Populate sidebar summary after predictions have been logged for this run
with sidebar_history_slot.container():
    hist = st.session_state["history"]
    st.metric("Predictions Logged", len(hist))
    if hist:
        if st.button("Clear History", width="stretch"):
            st.session_state["history"] = []
            st.rerun()
