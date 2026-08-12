import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import torch
import shap

from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Fake News Detection | BERT + XAI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================
MODEL_PATH = os.path.join("saved_models", "final_bert")
MAX_LENGTH = 256

CLASS_NAMES = ["Fake News", "Real News"]
FAKE_CLASS = 0
REAL_CLASS = 1

# Final corrected BERT performance from finalcorrected.ipynb
PERFORMANCE = {
    "Accuracy": 99.9227,
    "Precision": 100.0000,
    "Recall": 99.8567,
    "F1-score": 99.9283,
    "ROC-AUC": 99.9992,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# VISUAL DESIGN
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(99,102,241,0.08), transparent 30%),
            radial-gradient(circle at 90% 10%, rgba(14,165,233,0.07), transparent 30%),
            #f7f9fc;
        color: #172033;
    }

    .main .block-container {
        max-width: 1380px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #182235 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #eef2ff !important;
    }

    section[data-testid="stSidebar"] .stMetric {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.7rem;
    }

    .sidebar-card {
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1rem;
        margin: 0.8rem 0;
    }

    .hero {
        background: linear-gradient(135deg, #111827 0%, #1e293b 55%, #263b73 100%);
        border-radius: 24px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 18px 45px rgba(15,23,42,0.14);
        border: 1px solid rgba(255,255,255,0.08);
    }

    .hero-title {
        color: #ffffff;
        font-size: 2.55rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin: 0;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin-top: 0.55rem;
        line-height: 1.65;
    }

    .badge-row {
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
        margin-top: 1.1rem;
    }

    .badge {
        background: rgba(255,255,255,0.10);
        color: #f8fafc;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 999px;
        padding: 0.35rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 750;
        color: #172033;
        margin: 1.8rem 0 0.3rem 0;
    }

    .section-subtitle {
        color: #64748b;
        margin-bottom: 1rem;
    }

    .soft-card {
        background: rgba(255,255,255,0.90);
        border: 1px solid #e5eaf1;
        border-radius: 18px;
        padding: 1.2rem 1.35rem;
        box-shadow: 0 8px 25px rgba(15,23,42,0.045);
    }

    .stat-card {
        background: #ffffff;
        border: 1px solid #e5eaf1;
        border-radius: 16px;
        padding: 1.05rem 1.15rem;
        box-shadow: 0 7px 22px rgba(15,23,42,0.045);
        min-height: 105px;
    }

    .stat-label {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stat-value {
        color: #172033;
        font-size: 1.85rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }

    .result-card {
        border-radius: 20px;
        padding: 1.35rem 1.5rem;
        border: 1px solid;
        box-shadow: 0 10px 30px rgba(15,23,42,0.06);
    }

    .result-fake {
        background: linear-gradient(135deg, #fff1f2, #fff7f7);
        border-color: #fecdd3;
    }

    .result-real {
        background: linear-gradient(135deg, #ecfdf5, #f0fdf9);
        border-color: #a7f3d0;
    }

    .result-label {
        font-size: 0.82rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .result-title {
        font-size: 1.65rem;
        font-weight: 850;
        margin-top: 0.25rem;
    }

    .confidence-number {
        font-size: 2.25rem;
        font-weight: 850;
        color: #172033;
    }

    .xai-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        overflow: hidden;
        border: 1px solid #e5eaf1;
        border-radius: 14px;
        background: #ffffff;
    }

    .xai-table th {
        background: #f8fafc;
        color: #475569;
        text-align: left;
        padding: 0.72rem 0.85rem;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .xai-table td {
        padding: 0.7rem 0.85rem;
        border-top: 1px solid #eef2f7;
        color: #243044;
        font-size: 0.9rem;
    }

    .positive {
        color: #047857;
        font-weight: 700;
    }

    .negative {
        color: #be123c;
        font-weight: 700;
    }

    .stButton > button {
        width: 100%;
        min-height: 52px;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 750;
        border: 0;
        background: linear-gradient(135deg, #4f46e5, #2563eb);
        color: white;
        box-shadow: 0 8px 20px rgba(37,99,235,0.20);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 25px rgba(37,99,235,0.27);
    }

    textarea {
        border-radius: 14px !important;
    }

    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
    }

    .prob-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.35rem;
        color: #334155;
        font-weight: 700;
        font-size: 0.9rem;
    }

    .prob-track {
        height: 11px;
        background: #e8edf4;
        border-radius: 999px;
        overflow: hidden;
    }

    .prob-fake {
        height: 100%;
        background: linear-gradient(90deg, #ef4444, #f97316);
        border-radius: 999px;
    }

    .prob-real {
        height: 100%;
        background: linear-gradient(90deg, #10b981, #06b6d4);
        border-radius: 999px;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.82rem;
        padding-top: 2rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD FINAL SAVED BERT MODEL
# ============================================================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    return tokenizer, model


try:
    with st.spinner("Loading final BERT model..."):
        tokenizer, bert_model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error("BERT model could not be loaded.")
    st.exception(e)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 📰 Fake News Detection")

    st.markdown(
        """
        <div class="sidebar-card">
        <b>Project</b><br>
        Explainable Fake News Detection System
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model_loaded:
        st.success("Final BERT model loaded")
    else:
        st.error("Model unavailable")

    st.markdown("### Explainability")
    st.markdown(
        """
        🔍 **LIME** — local feature explanation  
        📊 **SHAP** — token-level feature attribution
        """
    )

    st.markdown("---")
    st.markdown("### Corrected BERT Performance")

    for metric, value in PERFORMANCE.items():
        st.metric(metric, f"{value:.4f}%")

    st.markdown("---")
    st.markdown("### Runtime")
    st.write(f"**Device:** `{device}`")

    if device.type == "cuda":
        try:
            st.caption(torch.cuda.get_device_name(0))
        except Exception:
            pass

    st.markdown("---")
    st.markdown("### Classification")
    st.write("**0** — Fake News")
    st.write("**1** — Real News")

    st.caption("Research prototype • BERT + LIME + SHAP")


if not model_loaded:
    st.stop()


# ============================================================
# ROBUST TEXT NORMALISATION
# ============================================================
def _stringify(value):
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return " ".join(_stringify(x) for x in value).strip()
    return str(value)


def normalise_batch(texts):
    if isinstance(texts, str):
        return [texts if texts.strip() else " "]

    if isinstance(texts, bytes):
        text = texts.decode("utf-8", errors="ignore")
        return [text if text.strip() else " "]

    if isinstance(texts, np.ndarray):
        texts = texts.tolist()

    if isinstance(texts, tuple):
        texts = list(texts)

    if not isinstance(texts, list):
        texts = [texts]

    return [
        _stringify(x) if _stringify(x).strip() else " "
        for x in texts
    ]


# ============================================================
# MEMORY-SAFE BERT PREDICTION
# ============================================================
def predict_proba(texts, batch_size=4):
    """
    Memory-safe prediction function used by normal inference,
    LIME and SHAP.

    Small batches prevent CUDA OutOfMemoryError when LIME creates
    hundreds of perturbed texts.
    """
    texts = normalise_batch(texts)

    all_probabilities = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoding = {
            key: value.to(device)
            for key, value in encoding.items()
        }

        with torch.no_grad():
            outputs = bert_model(**encoding)
            probabilities = torch.softmax(
                outputs.logits,
                dim=1,
            )

        all_probabilities.append(
            probabilities.detach().cpu().numpy()
        )

        del encoding
        del outputs
        del probabilities

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.vstack(all_probabilities)


def predict_news(text):
    probabilities = predict_proba([text], batch_size=1)[0]

    prediction = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    fake_probability = float(probabilities[FAKE_CLASS])
    real_probability = float(probabilities[REAL_CLASS])

    return (
        prediction,
        confidence,
        fake_probability,
        real_probability,
    )


def lime_predict(texts):
    return predict_proba(texts, batch_size=4)


def shap_predict(texts):
    return predict_proba(texts, batch_size=4)


# ============================================================
# XAI EXPLAINERS
# ============================================================
lime_explainer = LimeTextExplainer(
    class_names=CLASS_NAMES
)


@st.cache_resource
def load_shap_explainer():
    text_masker = shap.maskers.Text(tokenizer)

    return shap.Explainer(
        shap_predict,
        text_masker,
        algorithm="partition",
    )


# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">📰 Explainable Fake News Detection System</div>
        <div class="hero-subtitle">
            A research prototype using a fine-tuned BERT Transformer to classify
            English-language news articles and provide local LIME and SHAP explanations.
        </div>
        <div class="badge-row">
            <span class="badge">🤖 Fine-tuned BERT</span>
            <span class="badge">🔍 LIME</span>
            <span class="badge">📊 SHAP</span>
            <span class="badge">⚡ PyTorch</span>
            <span class="badge">🎓 MSc Research Prototype</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="soft-card">
        <b>Workflow</b><br>
        Article input → validation → BERT tokenisation → BERT prediction →
        confidence and class probabilities → LIME / SHAP explanation.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INPUT
# ============================================================
st.markdown(
    '<div class="section-title">📝 Analyse a News Article</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">'
    'Paste an English-language news article below. '
    'Longer articles are truncated to the first 256 BERT tokens, matching the final model configuration.'
    '</div>',
    unsafe_allow_html=True,
)

examples = {
    "Example 1 — Conventional News Style":
        (
            "Government officials announced on Tuesday that a new public transport "
            "investment programme will begin next year. The department said the plan "
            "will focus on improving rail reliability and reducing delays across major cities."
        ),

    "Example 2 — Sensational Claim":
        (
            "A viral online report claims that scientists have secretly discovered a device "
            "that can produce unlimited free energy without fuel. The article provides no "
            "independent scientific evidence and says the technology is being hidden from the public."
        ),

    "Example 3 — Health Misinformation Claim":
        (
            "A social media article claims that eating large amounts of chocolate immediately "
            "before sleeping causes rapid weight loss overnight. The article says the method is "
            "scientifically proven but provides no clinical study or independent source."
        ),
}

choice = st.selectbox(
    "📌 Optional example",
    ["None"] + list(examples.keys()),
)

default_text = examples.get(choice, "")

news_text = st.text_area(
    "News article",
    value=default_text,
    height=260,
    placeholder="Paste the complete news article here...",
)

live_words = len(news_text.split())
live_characters = len(news_text)

st.caption(
    f"Current input: **{live_words} words** · "
    f"**{live_characters} characters**"
)

predict_button = st.button(
    "🔎 Analyse Article with BERT",
    use_container_width=True,
)


# ============================================================
# MAIN ANALYSIS
# ============================================================
if predict_button:

    article_text = news_text.strip()

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------
    if not article_text:
        st.warning("Please enter a news article.")
        st.stop()

    if len(article_text) < 30:
        st.warning(
            "Please enter a longer article. "
            "The minimum accepted input is 30 characters."
        )
        st.stop()

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------
    with st.spinner("Running BERT classification..."):
        (
            prediction,
            confidence,
            fake_prob,
            real_prob,
        ) = predict_news(article_text)

    predicted_class = CLASS_NAMES[prediction]

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True,
    )

    if prediction == REAL_CLASS:
        result_class = "result-real"
        result_icon = "✅"
        result_label = "REAL NEWS"
        result_note = (
            "The final BERT classifier assigns the highest probability "
            "to the Real News class."
        )
    else:
        result_class = "result-fake"
        result_icon = "🚨"
        result_label = "FAKE NEWS"
        result_note = (
            "The final BERT classifier assigns the highest probability "
            "to the Fake News class."
        )

    left, right = st.columns([3.2, 1.0])

    with left:
        st.markdown(
            f"""
            <div class="result-card {result_class}">
                <div class="result-label">
                    {result_icon} BERT CLASSIFICATION
                </div>
                <div class="result-title">{result_label}</div>
                <div style="color:#64748b;margin-top:0.45rem;">
                    {result_note}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Confidence</div>
                <div class="confidence-number">
                    {confidence * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Confidence is the model's estimated classification probability. "
        "It is not independent evidence that the underlying article is "
        "factually true or false."
    )

    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------
    st.markdown(
        '<div class="section-title">📈 Class Probabilities</div>',
        unsafe_allow_html=True,
    )

    p1, p2 = st.columns(2)

    with p1:
        st.markdown(
            f"""
            <div class="soft-card">
                <div class="prob-header">
                    <span>🚨 Fake News</span>
                    <span>{fake_prob * 100:.2f}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fake"
                         style="width:{fake_prob * 100:.4f}%"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p2:
        st.markdown(
            f"""
            <div class="soft-card">
                <div class="prob-header">
                    <span>✅ Real News</span>
                    <span>{real_prob * 100:.2f}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-real"
                         style="width:{real_prob * 100:.4f}%"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    prob_df = pd.DataFrame(
        {
            "Class": CLASS_NAMES,
            "Probability": [fake_prob, real_prob],
        }
    )

    fig_prob, ax_prob = plt.subplots(figsize=(9, 4.6))

    bars = ax_prob.bar(
        prob_df["Class"],
        prob_df["Probability"],
        width=0.55,
    )

    ax_prob.set_ylim(0, 1.08)
    ax_prob.set_ylabel("Probability")
    ax_prob.set_title(
        "BERT Prediction Probabilities",
        fontsize=15,
        fontweight="bold",
    )

    ax_prob.grid(
        axis="y",
        alpha=0.18,
    )

    ax_prob.spines["top"].set_visible(False)
    ax_prob.spines["right"].set_visible(False)

    for bar, value in zip(
        bars,
        prob_df["Probability"]
    ):
        ax_prob.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.025, 1.04),
            f"{value * 100:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    st.pyplot(
        fig_prob,
        use_container_width=True,
    )
    plt.close(fig_prob)

    # --------------------------------------------------------
    # ARTICLE STATISTICS
    # --------------------------------------------------------
    st.markdown(
        '<div class="section-title">📄 Article Statistics</div>',
        unsafe_allow_html=True,
    )

    sentences = [
        sentence
        for sentence in re.split(
            r"[.!?]+",
            article_text
        )
        if sentence.strip()
    ]

    stat_columns = st.columns(4)

    stats = [
        ("Words", len(article_text.split())),
        ("Characters", len(article_text)),
        ("Sentences", len(sentences)),
        ("BERT Max Tokens", MAX_LENGTH),
    ]

    for column, (label, value) in zip(
        stat_columns,
        stats,
    ):
        with column:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">
                        {label}
                    </div>
                    <div class="stat-value">
                        {value:,}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # EXPLAINABILITY
    # --------------------------------------------------------
    st.markdown(
        '<div class="section-title">🧠 Explainable AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Generate local explanations for this specific prediction. '
        'These methods describe model behaviour; they do not perform factual verification.'
        '</div>',
        unsafe_allow_html=True,
    )

    lime_tab, shap_tab = st.tabs(
        [
            "🔍 LIME Explanation",
            "📊 SHAP Explanation",
        ]
    )

    # ========================================================
    # LIME
    # ========================================================
    with lime_tab:

        st.markdown(
            """
            **LIME** creates perturbed versions of the submitted article and
            observes how the BERT probabilities change. It then builds a local
            interpretable approximation for this individual prediction.
            """
        )

        try:
            with st.spinner(
                "Generating memory-safe LIME explanation..."
            ):
                lime_exp = lime_explainer.explain_instance(
                    article_text,
                    lime_predict,
                    num_features=12,
                    num_samples=300,
                )

            st.success(
                "LIME explanation generated successfully."
            )

            lime_df = pd.DataFrame(
                lime_exp.as_list(),
                columns=[
                    "Feature",
                    "Contribution",
                ],
            )

            lime_df["Contribution"] = (
                lime_df["Contribution"]
                .astype(float)
            )

            st.markdown(
                "#### Top LIME Features"
            )

            lime_rows = ""

            for _, row in lime_df.iterrows():

                contribution = float(
                    row["Contribution"]
                )

                css_class = (
                    "positive"
                    if contribution > 0
                    else "negative"
                )

                direction = (
                    "↑ Positive local contribution"
                    if contribution > 0
                    else "↓ Negative local contribution"
                )

                lime_rows += f"""
                <tr>
                    <td>{html.escape(str(row["Feature"]))}</td>
                    <td class="{css_class}">
                        {contribution:.6f}
                    </td>
                    <td class="{css_class}">
                        {direction}
                    </td>
                </tr>
                """

            st.markdown(
                f"""
                <table class="xai-table">
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>LIME Weight</th>
                            <th>Direction</th>
                        </tr>
                    </thead>
                    <tbody>
                        {lime_rows}
                    </tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "#### LIME Feature Importance"
            )

            lime_plot = lime_df.sort_values(
                "Contribution"
            )

            fig_lime, ax_lime = plt.subplots(
                figsize=(10, 6)
            )

            ax_lime.barh(
                lime_plot["Feature"],
                lime_plot["Contribution"],
            )

            ax_lime.axvline(
                0,
                linewidth=1,
            )

            ax_lime.set_xlabel(
                "LIME Weight"
            )

            ax_lime.set_ylabel(
                "Feature"
            )

            ax_lime.set_title(
                "LIME Explanation of BERT Prediction",
                fontsize=15,
                fontweight="bold",
            )

            ax_lime.grid(
                axis="x",
                alpha=0.16,
            )

            ax_lime.spines[
                "top"
            ].set_visible(False)

            ax_lime.spines[
                "right"
            ].set_visible(False)

            plt.tight_layout()

            st.pyplot(
                fig_lime,
                use_container_width=True,
            )

            plt.close(
                fig_lime
            )

            with st.expander(
                "🔬 Interactive LIME Explanation"
            ):
                components.html(
                    lime_exp.as_html(),
                    height=650,
                    scrolling=True,
                )

        except RuntimeError as e:

            if "out of memory" in str(e).lower():
                st.error(
                    "GPU memory was exhausted while generating LIME. "
                    "The prediction is still valid. Try again after closing "
                    "other GPU applications or reduce the batch size in "
                    "lime_predict() from 4 to 2."
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            else:
                st.error(
                    "Unable to generate LIME explanation."
                )
                st.exception(e)

        except Exception as e:
            st.error(
                "Unable to generate LIME explanation."
            )
            st.exception(e)

    # ========================================================
    # SHAP
    # ========================================================
    with shap_tab:

        st.markdown(
            """
            **SHAP** provides token-level attribution values for the submitted
            article. The output indicates how strongly individual features are
            associated with the selected prediction.
            """
        )

        try:

            with st.spinner(
                "Generating SHAP explanation..."
            ):
                shap_explainer = load_shap_explainer()

                shap_values = shap_explainer(
                    [article_text],
                    max_evals=500,
                )

            st.success(
                "SHAP explanation generated successfully."
            )

            sv = shap_values[0]

            tokens = np.asarray(
                sv.data,
                dtype=object,
            ).flatten()

            raw_values = np.asarray(
                sv.values
            )

            # -----------------------------------------------
            # ROBUST SHAP SHAPE HANDLING
            # -----------------------------------------------
            if raw_values.ndim == 1:
                values = raw_values

            elif raw_values.ndim == 2:

                if (
                    raw_values.shape[1]
                    > prediction
                ):
                    values = raw_values[
                        :,
                        prediction
                    ]
                else:
                    values = raw_values[
                        :,
                        0
                    ]

            elif raw_values.ndim == 3:

                if (
                    raw_values.shape[0]
                    == len(tokens)
                ):
                    temp = raw_values[
                        :,
                        0,
                        :
                    ]
                else:
                    temp = raw_values[
                        0,
                        :,
                        :
                    ]

                if (
                    temp.shape[1]
                    > prediction
                ):
                    values = temp[
                        :,
                        prediction
                    ]
                else:
                    values = temp[
                        :,
                        0
                    ]

            else:
                values = raw_values.reshape(-1)

            n = min(
                len(tokens),
                len(values)
            )

            shap_df = pd.DataFrame(
                {
                    "Feature": [
                        str(token).strip()
                        for token
                        in tokens[:n]
                    ],

                    "SHAP Contribution": [
                        float(value)
                        for value
                        in values[:n]
                    ],
                }
            )

            shap_df = shap_df[
                (
                    shap_df[
                        "Feature"
                    ] != ""
                )
                &
                (
                    ~shap_df[
                        "Feature"
                    ].isin(
                        [
                            "[CLS]",
                            "[SEP]",
                            "[PAD]",
                        ]
                    )
                )
            ]

            shap_df[
                "Absolute Contribution"
            ] = (
                shap_df[
                    "SHAP Contribution"
                ].abs()
            )

            shap_df = shap_df.sort_values(
                "Absolute Contribution",
                ascending=False,
            )

            top_shap = shap_df.head(
                15
            ).copy()

            st.markdown(
                "#### Top SHAP Features"
            )

            shap_rows = ""

            for _, row in top_shap.iterrows():

                contribution = float(
                    row[
                        "SHAP Contribution"
                    ]
                )

                css_class = (
                    "positive"
                    if contribution > 0
                    else "negative"
                )

                direction = (
                    "↑ Towards selected output"
                    if contribution > 0
                    else "↓ Away from selected output"
                )

                shap_rows += f"""
                <tr>
                    <td>{html.escape(str(row["Feature"]))}</td>
                    <td class="{css_class}">
                        {contribution:.6f}
                    </td>
                    <td>
                        {abs(contribution):.6f}
                    </td>
                    <td class="{css_class}">
                        {direction}
                    </td>
                </tr>
                """

            st.markdown(
                f"""
                <table class="xai-table">
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>SHAP Contribution</th>
                            <th>Absolute Contribution</th>
                            <th>Direction</th>
                        </tr>
                    </thead>
                    <tbody>
                        {shap_rows}
                    </tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "#### SHAP Feature Importance"
            )

            shap_plot = top_shap.sort_values(
                "SHAP Contribution"
            )

            fig_shap, ax_shap = plt.subplots(
                figsize=(10, 7)
            )

            ax_shap.barh(
                shap_plot[
                    "Feature"
                ],
                shap_plot[
                    "SHAP Contribution"
                ],
            )

            ax_shap.axvline(
                0,
                linewidth=1,
            )

            ax_shap.set_xlabel(
                "SHAP Contribution"
            )

            ax_shap.set_ylabel(
                "Feature"
            )

            ax_shap.set_title(
                "SHAP Explanation of BERT Prediction",
                fontsize=15,
                fontweight="bold",
            )

            ax_shap.grid(
                axis="x",
                alpha=0.16,
            )

            ax_shap.spines[
                "top"
            ].set_visible(False)

            ax_shap.spines[
                "right"
            ].set_visible(False)

            plt.tight_layout()

            st.pyplot(
                fig_shap,
                use_container_width=True,
            )

            plt.close(
                fig_shap
            )

        except RuntimeError as e:

            if "out of memory" in str(e).lower():

                st.error(
                    "GPU memory was exhausted while generating SHAP. "
                    "The BERT prediction is still valid. Try again after "
                    "closing other GPU applications."
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            else:
                st.error(
                    "Unable to generate SHAP explanation."
                )
                st.exception(e)

        except Exception as e:
            st.error(
                "Unable to generate SHAP explanation."
            )
            st.exception(e)

    # --------------------------------------------------------
    # RESPONSIBLE INTERPRETATION
    # --------------------------------------------------------
    st.markdown(
        '<div class="section-title">💡 Responsible Interpretation</div>',
        unsafe_allow_html=True,
    )

    interpretation_col1, interpretation_col2 = st.columns(2)

    with interpretation_col1:
        st.markdown(
            """
            <div class="soft-card">
                <h4>🔍 LIME</h4>
                <p>
                LIME approximates the BERT classifier locally around this
                specific article. Its feature weights are local explanatory
                signals, not universal indicators of misinformation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with interpretation_col2:
        st.markdown(
            """
            <div class="soft-card">
                <h4>📊 SHAP</h4>
                <p>
                SHAP estimates how textual features are associated with the
                model output. These attributions describe model behaviour and
                should not be interpreted as independent factual evidence.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "Important: this system performs machine-learning classification. "
        "It does not retrieve external evidence, verify sources, or perform "
        "professional fact-checking."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================
st.markdown(
    '<div class="section-title">ℹ️ Model Information</div>',
    unsafe_allow_html=True,
)

info_columns = st.columns(4)

model_cards = [
    ("Final Model", "Fine-tuned BERT"),
    ("Base Model", "bert-base-uncased"),
    ("Maximum Length", "256 tokens"),
    ("Explainability", "LIME + SHAP"),
]

for column, (label, value) in zip(
    info_columns,
    model_cards,
):
    with column:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">
                    {label}
                </div>
                <div class="stat-value"
                     style="font-size:1.15rem;">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# RESEARCH NOTICE
# ============================================================
st.markdown(
    """
    <div class="soft-card" style="margin-top:1.5rem;">
        <b>Research-use notice</b><br>
        This application is an AI-assisted classification and decision-support
        prototype developed for academic research. The output should not be
        treated as definitive evidence of factual truth or falsity. Performance
        reported in the interface reflects evaluation on the project's held-out
        dataset and may not generalise to all real-world news.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        Explainable Fake News Detection System ·
        Fine-tuned BERT · LIME · SHAP · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
