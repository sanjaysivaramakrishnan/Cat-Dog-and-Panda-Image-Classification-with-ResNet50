import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import io
import json
import time
import zipfile
import tempfile
from typing import Dict, List, Tuple, Optional

# Optional deps for metrics/plots
try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_curve,
        auc,
        precision_recall_curve,
        roc_auc_score,
        average_precision_score,
        accuracy_score,
    )
    from sklearn.preprocessing import label_binarize
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Animal Classifier",
    page_icon="🐾",
    layout="centered"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-size: 1.1rem;
        border-radius: 8px;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Device configuration
@st.cache_resource
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
@st.cache_resource
def load_model():
    device = get_device()
    
    # Load ResNet50 architecture
    model = models.resnet50(weights=None)
    
    # Modify the final layer to match your trained model BEFORE loading weights
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.6),  # Updated to match your model
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 3)  # 3 classes: cat, dog, panda
    )
    
    # Move model to device first
    model = model.to(device)
    
    # Load the saved weights
    # Replace 'model.pth' with your actual model file path
    try:
        # Use strict=False to ignore missing/unexpected keys if needed
        state_dict = torch.load('model.pth', map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please ensure 'model.pth' is in the same directory.")
        return None, device
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None, device

# ---------- Utility & visualization helpers ----------

def get_model_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_weights_version(weights_path: str = "model.pth") -> str:
    try:
        stat = os.stat(weights_path)
        return f"{int(stat.st_mtime)}-{stat.st_size}"
    except Exception:
        return "unknown"


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], normalize: bool = True):
    if not MATPLOTLIB_AVAILABLE:
        return None
    if normalize:
        with np.errstate(all="ignore"):
            cm_sum = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, cm_sum, where=cm_sum != 0)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))

    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            ax.text(j, i, f"{value:.2f}" if normalize else f"{int(value)}",
                    ha="center", va="center",
                    color="white" if value > thresh else "black")
    fig.tight_layout()
    return fig


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]):
    if not (SKLEARN_AVAILABLE and MATPLOTLIB_AVAILABLE):
        return None
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(6, 5))
    # Per-class
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC={roc_auc:.3f})")

    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, color="black", lw=2, linestyle="--",
            label=f"micro-average (AUC={auc_micro:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]):
    if not (SKLEARN_AVAILABLE and MATPLOTLIB_AVAILABLE):
        return None
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        ax.plot(recall, precision, lw=2, label=f"{class_name} (AP={ap:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def _read_image(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _collect_dataset_paths_from_dir(root_dir: str, class_names: List[str]) -> List[Tuple[str, int]]:
    # Expect directory structure: root_dir/<class_name>/**/image.jpg
    items: List[Tuple[str, int]] = []
    class_name_to_idx = {name.lower(): idx for idx, name in enumerate(class_names)}
    if not os.path.isdir(root_dir):
        return items
    for class_dir_name in os.listdir(root_dir):
        cls_idx = class_name_to_idx.get(class_dir_name.lower())
        if cls_idx is None:
            # Skip unknown class folder
            continue
        class_dir = os.path.join(root_dir, class_dir_name)
        if not os.path.isdir(class_dir):
            continue
        for base, _dirs, files in os.walk(class_dir):
            for file_name in files:
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    items.append((os.path.join(base, file_name), cls_idx))
    return items


def _extract_zip_to_temp(uploaded_zip_file) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="st_ds_")
    with zipfile.ZipFile(uploaded_zip_file) as zf:
        zf.extractall(tmp_dir)
    return tmp_dir


@st.cache_data(show_spinner=False)
def evaluate_model_on_dataset(
    model_state_version: str,
    dataset_root: str,
    class_names: List[str],
    device: torch.device,
) -> Dict[str, object]:
    """Run evaluation over a folder dataset and return metrics and artifacts."""
    # Collect image paths
    items = _collect_dataset_paths_from_dir(dataset_root, class_names)
    if not items:
        raise ValueError("No images found under dataset directory for the known classes.")

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []
    file_paths: List[str] = []

    model, _ = load_model()
    if model is None:
        raise RuntimeError("Model not loaded.")

    total_items = len(items)
    progress_text = st.empty()
    progress_bar = st.progress(0.0)

    for idx, (img_path, true_label) in enumerate(items, start=1):
        img = _read_image(img_path)
        if img is None:
            continue
        pred_label, _conf, prob_vec = predict(model, img, device)
        y_true.append(true_label)
        y_pred.append(pred_label)
        y_prob.append(prob_vec.tolist())
        file_paths.append(img_path)

        if idx % 5 == 0 or idx == total_items:
            progress = idx / total_items
            progress_bar.progress(progress)
            progress_text.markdown(f"Evaluating images... {idx}/{total_items}")

    progress_text.empty()
    progress_bar.empty()

    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn not available for computing metrics.")

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_prob_arr = np.array(y_prob)

    # Basic metrics
    overall_accuracy = accuracy_score(y_true_arr, y_pred_arr)
    report_dict = classification_report(y_true_arr, y_pred_arr, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(class_names))))

    # AUC/AP
    try:
        roc_auc_macro = roc_auc_score(label_binarize(y_true_arr, classes=list(range(len(class_names)))), y_prob_arr, average="macro", multi_class="ovr")
    except Exception:
        roc_auc_macro = None
    try:
        ap_macro = average_precision_score(label_binarize(y_true_arr, classes=list(range(len(class_names)))), y_prob_arr, average="macro")
    except Exception:
        ap_macro = None

    # Identify difficult/misclassified samples
    mis_idx = np.where(y_true_arr != y_pred_arr)[0].tolist()
    # Sort misclassified by low max prob (least confident)
    mis_idx_sorted = sorted(mis_idx, key=lambda i: float(np.max(y_prob_arr[i])), reverse=False)

    return {
        "accuracy": float(overall_accuracy),
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "roc_auc_macro": None if roc_auc_macro is None else float(roc_auc_macro),
        "ap_macro": None if ap_macro is None else float(ap_macro),
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
        "y_prob": y_prob_arr,
        "file_paths": file_paths,
        "mis_idx_sorted": mis_idx_sorted,
    }

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize((224, 254)),  # Match your test_transform
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(model, image, device):
    # Preprocess the image
    image_tensor = preprocess_image(image).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item(), probabilities[0].cpu().numpy()

# Main app
def main():
    # Header
    st.title("🐾 Animal Classifier")
    st.markdown("### Classify images of Cats, Dogs, and Pandas")
    st.markdown("---")

    # Class names
    class_names = [
        "Cat",
        "Dog",
        "Panda",
    ]

    # Load model
    model, device = load_model()

    if model is None:
        st.info("💡 **Instructions to use this app:**")
        st.markdown(
            """
            1. Save your trained model using: `torch.save(model.state_dict(), 'model.pth')`
            2. Place the `model.pth` file in the same directory as this script
            3. Run the app again: `streamlit run app.py`
            """
        )
        return

    # Sidebar: system + model info
    with st.sidebar:
        st.markdown("### ℹ️ System & Model")
        st.write(f"**Device:** {device}")
        st.write(f"**Classes:** {', '.join(class_names)}")
        try:
            st.write(f"**Parameters:** {get_model_parameter_count(model):,}")
        except Exception:
            pass
        st.caption(f"Weights version: {get_weights_version('model.pth')}")

        if not SKLEARN_AVAILABLE:
            st.warning("Install scikit-learn to enable metrics: `pip install scikit-learn`.")
        if not MATPLOTLIB_AVAILABLE:
            st.warning("Install matplotlib to see charts: `pip install matplotlib`.\n")

    tab_predict, tab_metrics, tab_about = st.tabs(["🔮 Predict", "📊 Metrics", "❔ About"])

    with tab_predict:
        # File uploader
        st.markdown("### Upload an Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload an image of a cat, dog, or panda",
            key="predict_uploader",
        )

        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("#### Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)

                # Optional: let user specify ground truth for quick feedback
                true_label_name = st.selectbox(
                    "Optional: Select the true label to compare",
                    options=["(none)"] + class_names,
                    index=0,
                )

            if st.button("🔍 Classify Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    pred_label, confidence, prob_vec = predict(model, image, device)

                with col2:
                    st.markdown("#### Prediction Results")
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <h2 style='text-align: center; color: #4CAF50;'>{class_names[pred_label]}</h2>
                            <h4 style='text-align: center;'>Confidence: {confidence*100:.2f}%</h4>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Display probabilities as a bar chart
                    st.markdown("#### Confidence Scores")
                    prob_display = {name: float(p) for name, p in zip(class_names, prob_vec)}
                    st.bar_chart(prob_display, height=220)

                    # If user selected true label, show quick correctness
                    if true_label_name and true_label_name != "(none)":
                        is_correct = class_names[pred_label] == true_label_name
                        st.info(
                            f"Prediction {'✅ correct' if is_correct else '❌ incorrect'} vs. true: {true_label_name}"
                        )
        else:
            st.info("👆 Upload an image to get started!")
            with st.expander("📖 How to use"):
                st.markdown(
                    """
                    1. **Upload** an image using the file uploader above
                    2. Click the **Classify Image** button
                    3. View the **prediction** with confidence scores

                    **Supported formats:** JPG, JPEG, PNG
                    
                    **Note:** The model works best with clear images of cats, dogs, or pandas.
                    """
                )

    with tab_metrics:
        st.markdown("### Evaluate Model on a Dataset")
        st.caption(
            "Upload a .zip with subfolders per class (e.g., cat/, dog/, panda/) or enter a directory path."
        )

        colA, colB = st.columns([1, 1])
        with colA:
            dataset_dir_input = st.text_input(
                "Local dataset directory (optional)", value="", placeholder="e.g. data/val"
            )
        with colB:
            uploaded_zip = st.file_uploader(
                "Or upload a zipped dataset (.zip)", type=["zip"], key="metrics_zip_uploader"
            )

        run_eval = st.button("▶️ Run Evaluation", type="primary", disabled=not SKLEARN_AVAILABLE)

        if run_eval:
            if not SKLEARN_AVAILABLE:
                st.error("scikit-learn is required for metrics. Install with `pip install scikit-learn`.")
                st.stop()

            # Determine dataset root
            extract_dir = None
            dataset_root = None
            try:
                if uploaded_zip is not None:
                    # Extract uploaded zip to temp dir
                    with st.spinner("Extracting dataset..."):
                        # Need a real file-like for ZipFile
                        as_bytes = uploaded_zip.read()
                        tmp_zip_path = os.path.join(tempfile.gettempdir(), f"ds_{int(time.time())}.zip")
                        with open(tmp_zip_path, "wb") as f:
                            f.write(as_bytes)
                        extract_dir = _extract_zip_to_temp(tmp_zip_path)
                        dataset_root = extract_dir
                elif dataset_dir_input:
                    dataset_root = dataset_dir_input
                else:
                    st.warning("Please provide a dataset directory or upload a .zip file.")
                    st.stop()

                if not os.path.isdir(dataset_root):
                    st.error("Dataset directory not found or invalid.")
                    st.stop()

                with st.spinner("Running evaluation..."):
                    results = evaluate_model_on_dataset(
                        model_state_version=get_weights_version("model.pth"),
                        dataset_root=dataset_root,
                        class_names=class_names,
                        device=device,
                    )

                # Summary metrics
                st.subheader("Overall Metrics")
                metric_cols = st.columns(3)
                metric_cols[0].metric("Accuracy", f"{results['accuracy']*100:.2f}%")
                if results.get("roc_auc_macro") is not None:
                    metric_cols[1].metric("ROC AUC (macro)", f"{results['roc_auc_macro']:.3f}")
                if results.get("ap_macro") is not None:
                    metric_cols[2].metric("Avg Precision (macro)", f"{results['ap_macro']:.3f}")

                # Classification report (as a table)
                st.markdown("#### Classification Report")
                report = results["classification_report"]
                # Build a table of dict rows without requiring pandas
                table_rows = []
                for key, val in report.items():
                    if isinstance(val, dict) and key in class_names + ["macro avg", "weighted avg", "micro avg", "accuracy"]:
                        if key == "accuracy":
                            # scikit-learn uses a special entry for accuracy
                            table_rows.append({
                                "class": "accuracy",
                                "precision": "-",
                                "recall": "-",
                                "f1-score": f"{val:.3f}",
                                "support": int(sum(report[c]["support"] for c in class_names)),
                            })
                        else:
                            table_rows.append({
                                "class": key,
                                "precision": f"{val.get('precision', 0.0):.3f}",
                                "recall": f"{val.get('recall', 0.0):.3f}",
                                "f1-score": f"{val.get('f1-score', 0.0):.3f}",
                                "support": val.get('support', 0),
                            })
                # Sort rows to show per-class first
                ordered_rows = [r for r in table_rows if r["class"] in class_names] + [r for r in table_rows if r["class"] not in class_names]
                st.table(ordered_rows)

                # Confusion matrix
                st.markdown("#### Confusion Matrix")
                if MATPLOTLIB_AVAILABLE:
                    fig_cm = plot_confusion_matrix(results["confusion_matrix"], class_names, normalize=True)
                    if fig_cm is not None:
                        st.pyplot(fig_cm, use_container_width=True)
                else:
                    st.write(results["confusion_matrix"])

                # ROC and PR curves
                if MATPLOTLIB_AVAILABLE and SKLEARN_AVAILABLE:
                    st.markdown("#### ROC Curves")
                    fig_roc = plot_roc_curves(results["y_true"], results["y_prob"], class_names)
                    if fig_roc is not None:
                        st.pyplot(fig_roc, use_container_width=True)

                    st.markdown("#### Precision–Recall Curves")
                    fig_pr = plot_pr_curves(results["y_true"], results["y_prob"], class_names)
                    if fig_pr is not None:
                        st.pyplot(fig_pr, use_container_width=True)

                # Hardest misclassifications
                st.markdown("#### Hardest Misclassifications")
                mis_indices = results["mis_idx_sorted"][:6]
                if not mis_indices:
                    st.success("No misclassifications found in this dataset.")
                else:
                    cols = st.columns(3)
                    for i, idx in enumerate(mis_indices):
                        col = cols[i % 3]
                        fp = results["file_paths"][idx]
                        img = _read_image(fp)
                        if img is None:
                            continue
                        pred_idx = int(results["y_pred"][idx])
                        true_idx = int(results["y_true"][idx])
                        prob_vec = results["y_prob"][idx]
                        with col:
                            st.image(img, caption=os.path.basename(fp), use_container_width=True)
                            st.caption(
                                f"pred: {class_names[pred_idx]} ({np.max(prob_vec)*100:.1f}%) • true: {class_names[true_idx]}"
                            )

                # Allow download of metrics JSON
                try:
                    serializable_results = {
                        "accuracy": results.get("accuracy"),
                        "classification_report": results.get("classification_report"),
                        "confusion_matrix": (
                            results.get("confusion_matrix").tolist()
                            if hasattr(results.get("confusion_matrix"), "tolist")
                            else results.get("confusion_matrix")
                        ),
                        "roc_auc_macro": results.get("roc_auc_macro"),
                        "ap_macro": results.get("ap_macro"),
                        "y_true": (
                            results.get("y_true").tolist()
                            if hasattr(results.get("y_true"), "tolist")
                            else results.get("y_true")
                        ),
                        "y_pred": (
                            results.get("y_pred").tolist()
                            if hasattr(results.get("y_pred"), "tolist")
                            else results.get("y_pred")
                        ),
                        "y_prob": (
                            results.get("y_prob").tolist()
                            if hasattr(results.get("y_prob"), "tolist")
                            else results.get("y_prob")
                        ),
                        "file_paths": results.get("file_paths"),
                        "mis_idx_sorted": results.get("mis_idx_sorted"),
                    }
                    export_payload = json.dumps(serializable_results, indent=2)
                    st.download_button(
                        label="⬇️ Download metrics (JSON)",
                        data=io.BytesIO(export_payload.encode("utf-8")),
                        file_name="metrics_summary.json",
                        mime="application/json",
                    )
                except Exception:
                    st.download_button(
                        label="⬇️ Download metrics (text)",
                        data=io.BytesIO(bytes(str(results), encoding="utf-8")),
                        file_name="metrics_summary.txt",
                        mime="text/plain",
                    )

            finally:
                # Best-effort cleanup of extracted dataset
                if extract_dir and os.path.isdir(extract_dir):
                    try:
                        # Don't recursively delete aggressively to avoid accidents; temp dir is safe though
                        import shutil

                        shutil.rmtree(extract_dir, ignore_errors=True)
                    except Exception:
                        pass

    with tab_about:
        st.markdown(
            """
            This app classifies images of cats, dogs, and pandas using a ResNet50-based model.
            
            In the Metrics tab, you can evaluate the model on a labeled dataset and view:
            - Confusion matrix (normalized)
            - ROC and Precision–Recall curves (micro/per-class)
            - Classification report (precision, recall, F1, support)
            - Hardest misclassifications for quick error analysis
            """
        )

if __name__ == "__main__":
    main()