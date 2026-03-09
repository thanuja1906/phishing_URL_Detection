
# Phishing URL Detection using Machine Learning

##  Overview

Phishing attacks are a major cybersecurity threat where attackers create fake websites to steal user credentials and sensitive data.

This project uses **Machine Learning with XGBoost** to detect whether a URL is **phishing or legitimate**. The system extracts features from the input URL and predicts the probability of phishing.

The model is optimized using **RandomizedSearchCV** and provides explainability using **SHAP (SHapley Additive exPlanations)**.

---

# Features

* Detect phishing URLs using **XGBoost classifier**
* Hyperparameter tuning using **RandomizedSearchCV**
* Explainable AI using **SHAP**
* Displays **phishing probability**
* Model evaluation using **ROC Curve**
* Simple interface for URL verification

---

#  How the System Works

1️⃣ User enters a **URL** in the input field.

2️⃣ The system performs **automatic feature extraction** from the URL.

Examples of extracted features:

* URL length
* Presence of special characters
* Domain properties
* HTTPS usage
* Subdomain analysis

3️⃣ Extracted features are sent to the **trained XGBoost model**.

4️⃣ The model predicts whether the URL is:

* **Legitimate**
* **Phishing**

5️⃣ The system displays:

* Phishing probability
* Prediction result

6️⃣ **SHAP explainability** helps understand why the model made the prediction.

---

# 📊 Example Prediction

Example output from the system:

* **Phishing URL Detected**
* **Phishing Probability:** 82.51%

---

## 🖥️ Project Demo

### 🔎 URL Detection Interface

User enters a suspicious URL and the system analyzes it using the trained **XGBoost model**.


<img width="1298" height="846" alt="image" src="https://github.com/user-attachments/assets/8842f238-d7c2-45f2-afad-ce12a9478ec2" />




---

### 🚨 Phishing Prediction Result

The system predicts whether the entered URL is phishing and shows the **phishing probability score**.



<img width="1293" height="612" alt="image" src="https://github.com/user-attachments/assets/2177aa67-3e62-40dd-a594-eeda727d5953" />



---

### 🧠 SHAP Explainability

SHAP explainability shows **which features influenced the model's decision**, helping understand why the URL was classified as phishing.



<img width="1309" height="839" alt="image" src="https://github.com/user-attachments/assets/93005a2f-26e7-40aa-8ac0-2e73658860f1" />


---

### 📊 ROC Curve Performance

The ROC curve evaluates the model performance.
The model achieved a **high AUC score of 0.9983**, indicating excellent phishing detection capability.



<img width="1294" height="767" alt="image" src="https://github.com/user-attachments/assets/003d61bf-1c2c-496c-84e6-8c268719c387" />




# 📁 Project Structure

```id="grt4io"
phishing_URL_Detection/
│
├── app.py
├── train_model.py
├── test_model.py
├── phishing_model.pkl
├── Dataset.csv
├── README.md
```

---

# 🛠 Technologies Used

* Python
* XGBoost
* Scikit-learn
* RandomizedSearchCV
* SHAP (Explainable AI)
* Pandas
* NumPy
* Matplotlib

---

# Installation

Clone the repository

```id="sdd8cd"
git clone https://github.com/thanuja1906/phishing_URL_Detection.git
```

Move to project directory

```id="0wqz9d"
cd phishing_URL_Detection
```

Install required libraries

```id="7g5hhn"
pip install pandas numpy scikit-learn xgboost shap matplotlib
```

---

# Run the Project

Train the model

```id="gb6wsa"
python train_model.py
```

Test the model

```id="l7yzn0"
python test_model.py
```

Run the application

```id="59f7q0"
python app.py
```

---

# Model Performance

* **Algorithm:** XGBoost
* **Hyperparameter tuning:** RandomizedSearchCV
* **Evaluation Metric:** ROC-AUC
* **AUC Score:** 0.9983

The high ROC-AUC score indicates excellent model performance in detecting phishing URLs.

---

# Future Improvements

* Deploy the model as a web application
* Create a browser extension for real-time phishing detection
* Improve SHAP visualization for feature importance
* Integrate real-time threat intelligence


