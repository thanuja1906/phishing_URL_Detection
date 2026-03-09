import streamlit as st
import pandas as pd
import numpy as np
import pickle
import math
import re
from urllib.parse import urlparse
# shap can import heavy deps (torch); import lazily and handle failures
shap = None
shap_available = False
try:
    import shap
    shap_available = True
except Exception:
    shap_available = False

from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go

# ===== Load Model =====
with open("phishing_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
feature_names = data["features"]

# ===== Feature Extraction =====
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

    # Ensure correct column order
    df_features = df_features[feature_names]

    # Force numeric conversion safely
    df_features = df_features.apply(pd.to_numeric, errors="coerce")
    
    return df_features

# Replace any NaN (if conversion failed)
    df_features = df_features.fillna(0.0)

    return df_features
    # return pd.DataFrame([features])[feature_names]






# ===== MODERN UI =====
st.set_page_config(page_title="WhizCyber - Phishing Detector", layout="wide", page_icon="🔐")

st.markdown("""
<style>
/* Full background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
    font-size: 18px; /* global base font size */
    line-height: 1.5;
}

/* Navbar */
.navbar {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:15px 40px;
    font-size:18px;
    font-weight:600;
}

.nav-links span {
    margin-left:25px;
    cursor:pointer;
    color:#cbd5e1;
}

.nav-links span:hover {
    color:white;
}

/* Hero Section */
.hero {
    padding:60px 80px;
}

.hero-title {
    font-size:64px;
    font-weight:800;
    margin-bottom:10px;
}

.hero-sub {
    font-size:20px;
    color:#cbd5e1;
    margin-bottom:30px;
}

.result-box {
    display:inline-block;
    padding:10px 16px;
    border-radius:6px;
    font-weight:700;
    margin-bottom:18px;
    font-size:16px;
}

.safe {
    background:#16a34a;
}

.phish {
    background:#dc2626;
}

.detect-btn > button {
    background-color:#2563eb;
    color:white;
    font-weight:700;
    border-radius:8px;
    padding:10px 25px;
}
</style>
""", unsafe_allow_html=True)

# ----- NAVBAR -----
# Removed navbar

# ----- HERO SECTION -----
col1 = st.columns(1)[0]

with col1:
    st.markdown('<div class="hero">', unsafe_allow_html=True)

    st.markdown('<div class="hero-title">ENTER URL TO VERIFY</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Automated Feature Extraction and Classification of Phishing URLs</div>', unsafe_allow_html=True)

    url = st.text_input("", placeholder="Enter URL here")

    detect = st.button("DETECT PHISHING", key="detect")

    # ----- Prediction -----
    if detect and url:
        features = extract_features(url)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        if prediction == 1:
            st.markdown('<div class="result-box phish">Phishing URL</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box safe">Legitimate URL</div>', unsafe_allow_html=True)

        st.metric("Phishing Probability", f"{round(probability*100,2)} %")

        # ----- Explainable AI - Feature Analysis (Dynamic per URL) -----
        st.markdown("---")
        st.subheader("🔎 Explainable AI — Feature Analysis")

        try:
            # Load dataset to get average values
            df_dataset = pd.read_csv("Dataset.csv")
            
            # Get feature columns
            X_dataset = df_dataset[feature_names].copy()
            
            # Clean data: convert to numeric and handle errors
            for col in X_dataset.columns:
                X_dataset[col] = pd.to_numeric(X_dataset[col], errors='coerce')
            
            # Calculate average values
            feature_avg = X_dataset.mean()
            
            # Get current URL features
            current_features = features.iloc[0]
            
            # Calculate deviation from average (as percentage)
            deviation = ((current_features - feature_avg) / (feature_avg + 1e-6)) * 100
            deviation = deviation.fillna(0)
            
            # Sort by absolute deviation to show most impactful features
            deviation_sorted = deviation.abs().sort_values(ascending=False)[:12]
            top_features = deviation_sorted.index.tolist()[::-1]
            top_deviations = [deviation[f] for f in top_features]
            
            # Create visualization
            fig_features = go.Figure()
            colors = ['#dc2626' if v > 0 else '#16a34a' for v in top_deviations]
            
            fig_features.add_trace(go.Bar(
                x=top_deviations,
                y=top_features,
                orientation='h',
                marker=dict(color=colors),
                text=[f'{v:+.1f}%' for v in top_deviations],
                textposition='auto'
            ))
            
            fig_features.update_layout(
                title="Feature Deviation from Average (This URL vs Dataset Average)",
                xaxis_title="% Difference from Average",
                yaxis_title="Feature",
                template="plotly_dark",
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig_features, use_container_width=True)
            st.write("**Red** = Higher than average (→ more phishing-like) | **Green** = Lower than average (→ more safe-like)")
            
            # Show top features affecting this URL
            st.write("**Top Features Impacting This URL:**")
            for i, feat in enumerate(top_features[:5], 1):
                val = current_features[feat]
                avg = feature_avg[feat]
                dev = deviation[feat]
                st.write(f"{i}. **{feat}**: {val:.4f} (avg: {avg:.4f}) → **{dev:+.1f}%**")
            
        except Exception as e:
            st.error(f"Error in feature analysis: {str(e)}")
            # Fallback: show feature values
            st.write("**Extracted Features:**")
            features_display = features.iloc[0].to_dict()
            for feat, val in list(features_display.items())[:10]:
                st.write(f"• {feat}: {val:.4f}")

# if hasattr(model, "feature_importances_"):

#     importances = model.feature_importances_
#     fi = dict(zip(feature_names, importances))

#     # Determine sign using dataset correlation
#     try:
#         df_sign = pd.read_csv("Dataset.csv")
#         signs = {}
#         for f in feature_names:
#             if f in df_sign.columns:
#                 corr = df_sign[f].corr(df_sign["label"])
#                 signs[f] = np.sign(corr) if not np.isnan(corr) else 1
#             else:
#                 signs[f] = 1
#         signed_fi = {k: fi[k] * signs.get(k, 1) for k in fi}
#     except:
#         signed_fi = fi

#     sorted_fi = sorted(signed_fi.items(), key=lambda x: abs(x[1]), reverse=True)

#     top = sorted_fi[:12]

#     feat_names = [f[0] for f in top][::-1]
#     vals = [f[1] for f in top][::-1]

#     colors = ["#dc2626" if v > 0 else "#16a34a" for v in vals]
#     labels = [f"{v:+.4f}" for v in vals]

#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         x=vals,
#         y=feat_names,
#         orientation='h',
#         marker_color=colors,
#         text=labels,
#         textposition='auto'
#     ))

#     fig.update_layout(
#         title="Explainable AI — Feature Contributions (fallback)",
#         xaxis_title="Signed Importance (approx)",
#         template="plotly_dark",
#         height=500
#     )

#     st.plotly_chart(fig, use_container_width=True)

#     st.subheader("Top Feature Contributions (model feature_importances)")
#     for feature, val in sorted_fi[:7]:
#         st.write(f"• {feature}: {round(val,4)}")

# else:
#     st.info("Model does not support feature importances.")   

    st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")
st.subheader("📊 Model Evaluation")
show_dataset_metrics = st.checkbox("Show ROC / AUC (Dataset)", value=True)
sample_frac = 0.1

# ===== Dataset ROC/AUC =====
if show_dataset_metrics:
    st.markdown("---")
    st.header("📈 ROC Curve and AUC")

    try:
        df = pd.read_csv("Dataset.csv")

        X = df[feature_names].copy()
        y = df["label"].astype(int)

        if 0 < sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
            X = df[feature_names]
            y = df["label"].astype(int)

        y_proba = model.predict_proba(X)[:, 1]

        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)

        # ===== PROFESSIONAL STYLE ROC =====
        fig = go.Figure()

        # ROC Curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.4f})',
            line=dict(color='#2563eb', width=4)
        ))

        # Random baseline
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Guess',
            line=dict(color='gray', dash='dash')
        ))

        fig.update_layout(
            title="Receiver Operating Characteristic (ROC)",
            xaxis=dict(title="False Positive Rate", range=[0,1]),
            yaxis=dict(title="True Positive Rate", range=[0,1]),
            template="plotly_white",
            width=900,
            height=600,
            legend=dict(
                x=0.6,
                y=0.1,
                bgcolor="rgba(255,255,255,0.6)"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show AUC metric nicely
        col1, col2 = st.columns(2)
        with col1:
            st.metric("AUC Score", f"{roc_auc:.4f}")
        with col2:
            st.metric("Model Quality", 
                      "Excellent" if roc_auc > 0.9 else
                      "Very Good" if roc_auc > 0.8 else
                      "Good" if roc_auc > 0.7 else
                      "Average")

    except Exception as e:
        st.error(f"Could not compute ROC/AUC: {e}")