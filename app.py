import streamlit as st
import pandas as pd
import numpy as np
import pickle
import math
import re
from urllib.parse import urlparse
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="WhizCyber - Phishing Detector",
    layout="wide",
    page_icon="🔐"
)

# ==============================
# DARK UI + BUTTON STYLING
# ==============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Headings */
.hero-title {
    font-size: 52px;
    font-weight: 800;
}
.hero-sub {
    font-size: 18px;
    color: #cbd5e1;
}

/* Styled Button */
div.stButton > button {
    background-color: #2563eb !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 18px !important;
    padding: 14px 35px !important;
    border-radius: 10px !important;
    border: none !important;
    width: 260px;
    transition: 0.3s ease-in-out;
}

div.stButton > button:hover {
    background-color: #1d4ed8 !important;
    transform: scale(1.05);
}

div.stButton > button:focus {
    outline: none !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
with open("phishing_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
feature_names = data["features"]

# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_features(url):

    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    url_len = len(url)
    dom_len = len(domain)
    is_ip = 1 if re.match(r"\d+\.\d+\.\d+\.\d+", domain) else 0
    tld_len = len(domain.split('.')[-1]) if '.' in domain else 0
    subdom_cnt = domain.count('.') - 1 if domain.count('.') > 1 else 0

    letter_cnt = sum(c.isalpha() for c in url)
    digit_cnt = sum(c.isdigit() for c in url)
    special_cnt = sum(not c.isalnum() for c in url)

    eq_cnt = url.count('=')
    qm_cnt = url.count('?')
    amp_cnt = url.count('&')
    dot_cnt = url.count('.')
    dash_cnt = url.count('-')
    under_cnt = url.count('_')

    total_chars = len(url)
    letter_ratio = letter_cnt / total_chars if total_chars else 0
    digit_ratio = digit_cnt / total_chars if total_chars else 0
    spec_ratio = special_cnt / total_chars if total_chars else 0

    is_https = 1 if parsed.scheme == "https" else 0
    slash_cnt = url.count('/')
    path_len = len(path)
    query_len = len(query)

    prob = [float(url.count(c)) / total_chars for c in dict.fromkeys(list(url))]
    entropy = -sum([p * math.log2(p) for p in prob])

    features = {
        'url_len': url_len,
        'dom_len': dom_len,
        'is_ip': is_ip,
        'tld_len': tld_len,
        'subdom_cnt': subdom_cnt,
        'letter_cnt': letter_cnt,
        'digit_cnt': digit_cnt,
        'special_cnt': special_cnt,
        'eq_cnt': eq_cnt,
        'qm_cnt': qm_cnt,
        'amp_cnt': amp_cnt,
        'dot_cnt': dot_cnt,
        'dash_cnt': dash_cnt,
        'under_cnt': under_cnt,
        'letter_ratio': letter_ratio,
        'digit_ratio': digit_ratio,
        'spec_ratio': spec_ratio,
        'is_https': is_https,
        'slash_cnt': slash_cnt,
        'entropy': entropy,
        'path_len': path_len,
        'query_len': query_len
    }

    df_features = pd.DataFrame([features])
    df_features = df_features[feature_names]
    df_features = df_features.apply(pd.to_numeric, errors="coerce")
    df_features = df_features.fillna(0.0)

    return df_features

# ==============================
# UI HEADER
# ==============================
st.markdown('<div class="hero-title">ENTER URL TO VERIFY</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Automated Feature Extraction and Classification of Phishing URLs</div>', unsafe_allow_html=True)

url = st.text_input("Enter URL")
detect = st.button("DETECT PHISHING")

# ==============================
# PREDICTION
# ==============================
if detect and url:

    features = extract_features(url)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error("🚨 Phishing URL Detected")
    else:
        st.success("✅ Legitimate URL")

    # BIG PROBABILITY DISPLAY
    prob_percent = round(probability * 100, 2)
    color = "#dc2626" if prediction == 1 else "#16a34a"

    st.markdown(f"""
    <div style="
        background-color:#0f172a;
        padding:25px;
        border-radius:12px;
        margin-top:15px;
        margin-bottom:25px;
        border:1px solid #334155;
    ">
        <div style="font-size:18px; color:#cbd5e1;">
            Phishing Probability
        </div>
        <div style="font-size:52px; font-weight:800; color:{color};">
            {prob_percent} %
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==============================
    # SHAP EXPLAINABILITY
    # ==============================
    st.markdown("---")
    st.subheader("🧠 SHAP Explainability — Why this prediction?")

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_series = pd.Series(shap_values[0], index=feature_names)

        top_features = shap_series.abs().sort_values(ascending=False).head(6).index

        st.markdown("### 🔎 Model Decision Explanation")

        for feat in top_features:
            value = features.iloc[0][feat]
            impact = shap_series[feat]

            if impact > 0:
                st.markdown(f"• **{feat}** = {value:.3f} → increases phishing risk")
            else:
                st.markdown(f"• **{feat}** = {value:.3f} → reduces phishing risk")

        # SHAP Bar Chart
        fig = go.Figure()
        colors = ["#dc2626" if shap_series[f] > 0 else "#16a34a" for f in top_features]

        fig.add_trace(go.Bar(
            x=[shap_series[f] for f in top_features][::-1],
            y=top_features[::-1],
            orientation='h',
            marker=dict(color=colors),
            text=[f"{shap_series[f]:+.3f}" for f in top_features][::-1],
            textposition="auto"
        ))

        fig.update_layout(
            template="plotly_dark",
            title="Top Features Influencing Prediction",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

# ==============================
# ROC CURVE
# ==============================
st.markdown("---")
st.subheader("📈 ROC Curve (Dataset)")

try:
    df = pd.read_csv("Dataset.csv")
    X = df[feature_names]
    y = df["label"].astype(int)

    y_proba = model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                             name=f'ROC Curve (AUC = {roc_auc:.4f})'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1],
                             mode='lines',
                             name='Random Guess',
                             line=dict(dash='dash')))

    fig.update_layout(
        template="plotly_dark",
        title="Receiver Operating Characteristic",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.metric("AUC Score", f"{roc_auc:.4f}")

except Exception as e:
    st.warning(f"ROC not available: {e}")