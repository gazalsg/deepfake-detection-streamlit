"""
DeepFake Detection System - AICS 2025
Main Streamlit Application with Full Visualizations
"""

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import time

from model import load_model
from preprocessing import preprocess_image
from ui import apply_custom_css, render_header, render_footer, render_sidebar

st.set_page_config(
    page_title="DeepFake Detection - AICS 2025",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─── Attention Heatmap ────────────────────────────────────────────────────────
def generate_attention_heatmap(model, img_array, original_image):
    """Generate a pseudo attention heatmap using gradient-based saliency."""
    import tensorflow as tf

    img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0), dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        pred_class = tf.argmax(preds[0])
        class_score = preds[:, pred_class]

    grads = tape.gradient(class_score, img_tensor)
    grads = grads[0].numpy()

    # Absolute value, max across channels
    saliency = np.max(np.abs(grads), axis=-1)
    saliency = cv2.resize(saliency, (128, 128))
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    # Overlay on original image
    orig = np.array(original_image.resize((128, 128))).astype(np.uint8)
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig, 0.55, heatmap, 0.45, 0)
    return overlay, saliency


# ─── Confidence Bar Chart ─────────────────────────────────────────────────────
def plot_confidence_chart(fake_conf, real_conf):
    fig, ax = plt.subplots(figsize=(6, 2.8))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    bars = ax.barh(
        ['Synthetic (Fake)', 'Authentic (Real)'],
        [fake_conf, real_conf],
        color=['#E85D24' if fake_conf > real_conf else '#555555',
               '#1D9E75' if real_conf > fake_conf else '#555555'],
        height=0.5,
        edgecolor='none'
    )

    for bar, val in zip(bars, [fake_conf, real_conf]):
        ax.text(min(val + 1.5, 96), bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', ha='left',
                color='white', fontsize=13, fontweight='bold')

    ax.set_xlim(0, 105)
    ax.set_xlabel('Confidence (%)', color='#AAAAAA', fontsize=11)
    ax.tick_params(colors='#CCCCCC', labelsize=12)
    ax.spines[:].set_visible(False)
    ax.tick_params(left=False)
    ax.xaxis.set_tick_params(color='#444444')
    ax.set_title('Classification Confidence', color='white', fontsize=13, pad=10)

    for spine in ax.spines.values():
        spine.set_color('#333333')

    plt.tight_layout()
    return fig


# ─── Preprocessing Pipeline Visual ───────────────────────────────────────────
def show_preprocessing_pipeline(original_image, method):
    """Show step-by-step preprocessing pipeline as image grid."""
    steps = []
    titles = []

    # Step 1: Original
    orig = np.array(original_image.convert('RGB'))
    steps.append(orig)
    titles.append("1. Original")

    # Step 2: Resized
    resized = cv2.resize(orig, (128, 128))
    steps.append(resized)
    titles.append("2. Resized 128×128")

    # Step 3: Color conversion
    if method == "training_match":
        converted = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        titles.append("3. RGB → BGR")
    else:
        converted = resized.copy()
        titles.append("3. RGB kept")
    steps.append(converted)

    # Step 4: Normalized view
    if method == "simple_norm":
        normed = (resized / 255.0 * 255).astype(np.uint8)
        titles.append("4. Normalized [0,1]")
    elif method == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        normed_arr = preprocess_input(resized.astype(np.float32))
        normed = ((normed_arr - normed_arr.min()) /
                  (normed_arr.max() - normed_arr.min()) * 255).astype(np.uint8)
        titles.append("4. ImageNet norm")
    else:
        normed = resized.copy()
        titles.append("4. float32 (no norm)")
    steps.append(normed)

    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
    fig.patch.set_facecolor('#0E1117')

    for ax, img, title in zip(axes, steps, titles):
        ax.set_facecolor('#0E1117')
        try:
            ax.imshow(img)
        except Exception:
            ax.imshow(img[:, :, ::-1])
        ax.set_title(title, color='white', fontsize=9, pad=5)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')

    plt.suptitle(f'Preprocessing pipeline — "{method}" method',
                 color='#CCCCCC', fontsize=11, y=1.02)
    plt.tight_layout()
    return fig


# ─── Model Layer Flow ─────────────────────────────────────────────────────────
def plot_model_pipeline(fake_conf, real_conf):
    """Animated-style model pipeline showing activation strength."""
    fig, ax = plt.subplots(figsize=(11, 2.6))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis('off')

    layers = [
        ("Input\n128×128×3", 0.6),
        ("EfficientNetB7\nBackbone", 1.8),
        ("Batch\nNorm", 3.0),
        ("Attention\nBlock", 4.4),
        ("Dropout\n+ Dense 64", 5.8),
        ("Softmax\nOutput", 7.2),
        ("Prediction", 8.8),
    ]

    pred_strength = max(fake_conf, real_conf) / 100
    is_fake = fake_conf > real_conf
    result_color = '#E85D24' if is_fake else '#1D9E75'

    for i, (label, x) in enumerate(layers):
        alpha = 0.4 + 0.6 * (i / len(layers)) * pred_strength
        color = result_color if i == len(layers) - 1 else '#2E75B6'
        if i == 3:
            color = '#7F77DD'

        box = mpatches.FancyBboxPatch(
            (x - 0.55, 0.35), 1.1, 1.3,
            boxstyle="round,pad=0.08",
            linewidth=1.2,
            edgecolor=color,
            facecolor=color + '33',
        )
        ax.add_patch(box)
        ax.text(x, 1.0, label, ha='center', va='center',
                color='white', fontsize=7.5, fontweight='bold',
                multialignment='center')

        if i < len(layers) - 1:
            next_x = layers[i + 1][1]
            ax.annotate('', xy=(next_x - 0.56, 1.0), xytext=(x + 0.56, 1.0),
                        arrowprops=dict(arrowstyle='->', color='#666666', lw=1.2))

    pred_label = f"{'FAKE' if is_fake else 'REAL'}\n{max(fake_conf, real_conf):.1f}%"
    ax.text(8.8, 1.0, pred_label, ha='center', va='center',
            color=result_color, fontsize=9, fontweight='bold',
            multialignment='center')

    ax.set_title('Model inference pipeline', color='#CCCCCC',
                 fontsize=11, pad=6, loc='left')
    plt.tight_layout()
    return fig


# ─── Gauge Chart ─────────────────────────────────────────────────────────────
def plot_gauge(confidence, label, is_fake):
    """Semicircle gauge showing confidence."""
    fig, ax = plt.subplots(figsize=(3.5, 2.2), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    theta = np.linspace(np.pi, 0, 100)
    ax.plot(theta, np.ones(100) * 0.8, color='#333333', lw=12, solid_capstyle='round')

    fill_theta = np.linspace(np.pi, np.pi - (confidence / 100) * np.pi, 100)
    color = '#E85D24' if is_fake else '#1D9E75'
    ax.plot(fill_theta, np.ones(100) * 0.8, color=color, lw=12, solid_capstyle='round')

    ax.set_ylim(0, 1)
    ax.set_theta_zero_location('E')
    ax.axis('off')
    ax.text(0, -0.2, f'{confidence:.1f}%', ha='center', va='center',
            color='white', fontsize=16, fontweight='bold',
            transform=ax.transData)
    ax.text(0, -0.5, label, ha='center', va='center',
            color=color, fontsize=10, fontweight='bold',
            transform=ax.transData)
    plt.tight_layout()
    return fig


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    apply_custom_css()
    render_header()

    model = load_model()
    if model is None:
        st.error("⚠️ Model could not be loaded. Please refresh the page.")
        return

    preprocess_method, show_debug = render_sidebar()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 Image Upload")
        uploaded_file = st.file_uploader(
            "Select an image file...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a JPG, JPEG, or PNG image for deepfake analysis"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if show_debug:
                st.markdown("**🔍 Image Metadata:**")
                st.write(f"- Format: {image.format} | Mode: {image.mode} | Size: {image.size}")

    with col2:
        st.markdown("### 📊 Analysis Results")

        if uploaded_file is not None:
            if st.button("🔍 Analyze Image", use_container_width=True):

                progress = st.progress(0, text="Starting analysis...")
                time.sleep(0.2)

                # ── Step 1: Preprocessing ──────────────────────────────────
                progress.progress(15, text="🔄 Preprocessing image...")
                img = preprocess_image(image, method=preprocess_method)
                if img is None:
                    st.error("❌ Could not process the uploaded image.")
                    return
                time.sleep(0.3)

                # ── Step 2: Inference ──────────────────────────────────────
                progress.progress(45, text="🧠 Running model inference...")
                img_batch = np.expand_dims(img, axis=0)
                prediction = model.predict(img_batch, verbose=0)
                probs = np.array(prediction).squeeze().astype(float)
                time.sleep(0.3)

                # ── Step 3: Results ────────────────────────────────────────
                progress.progress(75, text="📊 Generating visualizations...")
                answer_idx = int(np.argmax(probs))
                class_labels = ['Fake', 'Real']
                pred_label = class_labels[answer_idx]
                fake_conf = float(probs[0]) * 100
                real_conf = float(probs[1]) * 100
                confidence = float(probs[answer_idx]) * 100

                time.sleep(0.3)
                progress.progress(100, text="✅ Analysis complete!")
                time.sleep(0.3)
                progress.empty()

                # ── Verdict Banner ─────────────────────────────────────────
                if pred_label == "Fake":
                    st.error(f"### 🚨 RESULT: FAKE (AI-Generated) — {confidence:.1f}% confidence")
                else:
                    st.success(f"### ✅ RESULT: AUTHENTIC — {confidence:.1f}% confidence")

                st.markdown("---")

                # ── Visualization Tabs ─────────────────────────────────────
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Confidence", "🔥 Attention Map",
                    "🔄 Preprocessing", "🧠 Model Pipeline"
                ])

                with tab1:
                    st.markdown("#### Classification confidence scores")
                    # Gauges side by side
                    g1, g2 = st.columns(2)
                    with g1:
                        fig_g1 = plot_gauge(fake_conf, "FAKE", True)
                        st.pyplot(fig_g1, use_container_width=True)
                        plt.close(fig_g1)
                    with g2:
                        fig_g2 = plot_gauge(real_conf, "REAL", False)
                        st.pyplot(fig_g2, use_container_width=True)
                        plt.close(fig_g2)

                    # Bar chart
                    fig_bar = plot_confidence_chart(fake_conf, real_conf)
                    st.pyplot(fig_bar, use_container_width=True)
                    plt.close(fig_bar)

                    st.info(f"**Decision:** {pred_label} | "
                            f"Fake: {fake_conf:.2f}% | Real: {real_conf:.2f}%")

                with tab2:
                    st.markdown("#### Attention heatmap — where the model focused")
                    with st.spinner("Generating attention heatmap..."):
                        try:
                            overlay, saliency = generate_attention_heatmap(
                                model, img, image)
                            h1, h2 = st.columns(2)
                            with h1:
                                st.image(np.array(image.resize((128, 128))),
                                         caption="Original (128×128)", use_container_width=True)
                            with h2:
                                st.image(overlay,
                                         caption="Attention overlay (red = high focus)",
                                         use_container_width=True)

                            fig_sal, ax_sal = plt.subplots(figsize=(6, 1.2))
                            fig_sal.patch.set_facecolor('#0E1117')
                            ax_sal.set_facecolor('#0E1117')
                            grad = np.linspace(0, 1, 256).reshape(1, -1)
                            ax_sal.imshow(grad, aspect='auto',
                                          cmap='jet', vmin=0, vmax=1)
                            ax_sal.set_xticks([0, 128, 255])
                            ax_sal.set_xticklabels(['Low', 'Medium', 'High'],
                                                    color='white', fontsize=9)
                            ax_sal.set_yticks([])
                            ax_sal.set_title('Attention intensity scale',
                                             color='#CCCCCC', fontsize=10)
                            for sp in ax_sal.spines.values():
                                sp.set_visible(False)
                            st.pyplot(fig_sal, use_container_width=True)
                            plt.close(fig_sal)

                        except Exception as e:
                            st.warning(f"Heatmap generation requires eager execution. "
                                       f"Error: {e}")
                            st.info("Tip: Run with `TF_EAGER=1 streamlit run src/app.py`")

                with tab3:
                    st.markdown("#### Preprocessing pipeline — step by step")
                    fig_pre = show_preprocessing_pipeline(image, preprocess_method)
                    st.pyplot(fig_pre, use_container_width=True)
                    plt.close(fig_pre)

                    st.markdown(f"""
                    | Step | Operation | Output |
                    |------|-----------|--------|
                    | 1 | Load PIL image | RGB array |
                    | 2 | Resize | 128 × 128 px |
                    | 3 | Color conversion | {'BGR' if preprocess_method == 'training_match' else 'RGB'} |
                    | 4 | Normalize | {'float32 raw [0-255]' if preprocess_method == 'training_match' else '[0,1]' if preprocess_method == 'simple_norm' else 'ImageNet norm'} |
                    | 5 | Expand dims | (1, 128, 128, 3) batch |
                    """)

                with tab4:
                    st.markdown("#### Model inference pipeline")
                    fig_pipe = plot_model_pipeline(fake_conf, real_conf)
                    st.pyplot(fig_pipe, use_container_width=True)
                    plt.close(fig_pipe)

                    st.markdown("""
                    | Layer | Type | Output shape |
                    |-------|------|-------------|
                    | Input | Image tensor | (1, 128, 128, 3) |
                    | EfficientNetB7 | CNN backbone | (1, 4, 4, 2560) |
                    | Batch Norm | Normalization | (1, 4, 4, 2560) |
                    | Attention block | Spatial attention | (1, 2560) |
                    | Dropout + Dense 64 | Classification head | (1, 64) |
                    | Dense 2 + Softmax | Output | (1, 2) |
                    """)

                if show_debug:
                    with st.expander("🔍 Debug info"):
                        st.write(f"Tensor shape: {img_batch.shape}")
                        st.write(f"Value range: [{img.min():.4f}, {img.max():.4f}]")
                        st.write(f"Raw probs: {probs}")
        else:
            st.info("👆 Upload an image and click Analyze to see results.")

    render_footer()


if __name__ == "__main__":
    main()