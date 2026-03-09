import pandas as pd
import numpy as np
import pickle
import math
import re
import shap
import matplotlib.pyplot as plt
from urllib.parse import urlparse

# ===== LOAD MODEL =====
with open("phishing_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
feature_names = data["features"]

# ===== FEATURE EXTRACTION FUNCTION =====
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

    # entropy
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

    return pd.DataFrame([features])[feature_names]


# ===== TEST INPUT =====
url = input("Enter URL to test: ")

features = extract_features(url)

prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0][1]

if prediction == 1:
    print(f"\n⚠️ Phishing Website Detected")
else:
    print(f"\n✅ Legitimate Website")

print("Phishing Probability:", round(probability * 100, 2), "%")

# ===== EXPLAINABLE AI =====
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features)

print("\n🔎 Feature Contribution:")

for i, feature in enumerate(feature_names):
    impact = shap_values[0][i]
    if impact > 0:
        print(f"+ {feature} increased phishing score ({round(impact,4)})")
    else:
        print(f"- {feature} decreased phishing score ({round(impact,4)})")

# Visual Plot
shap.summary_plot(shap_values, features)