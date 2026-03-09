#======IMPORT LIBRARIES======
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier

#======LOAD DATASET======
df = pd.read_csv("Dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())

#======PREPROCESSING======
df = df.dropna()

# Remove string columns
X = df.drop(["label", "url", "dom", "tld"], axis=1)
y = df["label"]

print("Feature Columns:", X.columns.tolist())
#======TRAIN-TEST SPLIT======
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Class Distribution:\n", y_train.value_counts())

#======HANDLE IMBALANCE======
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

#======XGBOOST MODEL======
xgb = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)

#======HYPERPARAMETER GRID======
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 1.5]
}

#======RANDOMIZED SEARCH======
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=20,
    scoring='f1',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

#======BEST MODEL======
print("Best Parameters:", random_search.best_params_)
best_model = random_search.best_estimator_

#======TEST EVALUATION======
y_pred = best_model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#======CROSS VALIDATION======
cv_scores = cross_val_score(best_model, X, y, cv=5)
print("Cross Validation Accuracy:", cv_scores.mean())

#======CHECK OVERFITTING======remove this mart
train_pred = best_model.predict(X_train)

train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, y_pred)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print("Overfitting %:", train_accuracy - test_accuracy)

#======SAVE MODEL======
with open("phishing_model.pkl", "wb") as f:
    pickle.dump({
    "model": best_model,
    "features": X.columns.tolist()
}, f)

print("Model saved successfully!")