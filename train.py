import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


# Load dataset
df = pd.read_csv("data/transactions.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)

print("\nClass distribution:")
print(df["Class"].value_counts())


# Separate input and output
X = df.drop("Class", axis=1)
y = df["Class"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Scale Amount
scaler = StandardScaler()

X_train["Amount"] = scaler.fit_transform(
    X_train[["Amount"]]
)

X_test["Amount"] = scaler.transform(
    X_test[["Amount"]]
)


# -------------------------------
# LOGISTIC REGRESSION
# -------------------------------

print("\nTraining Logistic Regression...")

lr_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_probability = lr_model.predict_proba(X_test)[:, 1]

lr_auc = roc_auc_score(
    y_test,
    lr_probability
)

print("\nLogistic Regression Results:")
print(classification_report(y_test, lr_pred))

print("ROC-AUC:", lr_auc)


# -------------------------------
# RANDOM FOREST
# -------------------------------

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_probability = rf_model.predict_proba(X_test)[:, 1]

rf_auc = roc_auc_score(
    y_test,
    rf_probability
)

print("\nRandom Forest Results:")
print(classification_report(y_test, rf_pred))

print("ROC-AUC:", rf_auc)


# -------------------------------
# MODEL COMPARISON
# -------------------------------

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print("Logistic Regression ROC-AUC:", lr_auc)
print("Random Forest ROC-AUC:", rf_auc)


# Choose better model
if rf_auc > lr_auc:
    best_model = rf_model
    best_name = "Random Forest"
else:
    best_model = lr_model
    best_name = "Logistic Regression"


print("\nBest Model:", best_name)


# -------------------------------
# CONFUSION MATRIX
# -------------------------------

cm = confusion_matrix(
    y_test,
    rf_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")

plt.savefig(
    "screenshot/confusion_matrix.png"
)

plt.show()


# -------------------------------
# ROC CURVE
# -------------------------------

lr_fpr, lr_tpr, _ = roc_curve(
    y_test,
    lr_probability
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_probability
)


plt.figure(figsize=(7, 5))

plt.plot(
    lr_fpr,
    lr_tpr,
    label=f"Logistic Regression (AUC = {lr_auc:.3f})"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {rf_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig(
    "screenshot/roc_curve.png"
)

plt.show()


# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------

if best_name == "Random Forest":

    importance = pd.Series(
        rf_model.feature_importances_,
        index=X.columns
    )

    importance = importance.sort_values(
        ascending=False
    ).head(10)

    plt.figure(figsize=(8, 5))

    importance.sort_values().plot(
        kind="barh"
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.title(
        "Top 10 Important Features"
    )

    plt.savefig(
        "screenshot/feature_importance.png"
    )

    plt.show()


# -------------------------------
# SAVE MODEL
# -------------------------------

joblib.dump(
    best_model,
    "model/fraud_model.pkl"
)

joblib.dump(
    scaler,
    "model/scaler.pkl"
)

print("\n==============================")
print("MODEL SAVED SUCCESSFULLY")
print("==============================")

print("Model:", best_name)
print("Saved: model/fraud_model.pkl")
print("Saved: model/scaler.pkl")