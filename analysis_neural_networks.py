# Generated from: analysis.ipynb
# Converted at: 2026-09-21T16:01:39.743Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# **Installation Of Libraries**


!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install scikit-learn
!pip install jupyter

# **Task 1: Data Loading and Exploration**


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report
)

df = pd.read_csv("/content/HR_Attrition.csv")

original_df = df.copy()

df.head(10)

df.shape

df.info()

df["Attrition"].value_counts()

attrition_rate = (
    df["Attrition"].value_counts(normalize=True)*100
)

print(attrition_rate)

numeric = df.select_dtypes(include=np.number)

categorical = df.select_dtypes(include="object")

# **Task 2:Data Cleaning & Pre-Processing**


df.isnull().sum()

df.drop(
    columns=[
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours"
    ],
    inplace=True
)

df["Attrition"] = df["Attrition"].map({
    "Yes":1,
    "No":0
})

df = pd.get_dummies(
    df,
    drop_first=True
)

X = df.drop("Attrition",axis=1)

y = df["Attrition"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# **Task 3:Exploratory Data Analysis (EDA)**


import pandas as pd

eda_df = original_df.copy()

print("="*70)
print("TASK 3 - EXPLORATORY DATA ANALYSIS")
print("="*70)

print("\n1. Attrition Rate by Department (%)")

dept_attrition = (
    eda_df.groupby("Department")["Attrition"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

print(dept_attrition.round(2))

highest_dept = dept_attrition.idxmax()
highest_dept_rate = dept_attrition.max()

print(f"\nDepartment with Highest Attrition: {highest_dept} ({highest_dept_rate:.2f}%)")

print("\n" + "="*70)
print("2. Attrition Rate by Job Role (%)")

job_attrition = (
    eda_df.groupby("JobRole")["Attrition"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

print(job_attrition.round(2))

highest_job = job_attrition.idxmax()
highest_job_rate = job_attrition.max()

print(f"\nJob Role with Highest Attrition: {highest_job} ({highest_job_rate:.2f}%)")

print("\n" + "="*70)
print("3. Monthly Income vs Attrition")

income_stats = (
    eda_df.groupby("Attrition")["MonthlyIncome"]
    .agg(["count","mean","median","min","max"])
)

print(income_stats)

avg_yes = income_stats.loc["Yes","mean"]
avg_no = income_stats.loc["No","mean"]

print()

if avg_yes < avg_no:
    print("Observation:")
    print("Employees who left have LOWER average monthly income.")
else:
    print("Observation:")
    print("Employees who left do NOT have lower average monthly income.")

print("\n" + "="*70)
print("4. Work-Life Balance vs Attrition (%)")

wlb_attrition = (
    eda_df.groupby("WorkLifeBalance")["Attrition"]
    .apply(lambda x: (x=="Yes").mean()*100)
)

print(wlb_attrition.round(2))

highest_wlb = wlb_attrition.idxmax()

print(f"\nHighest Attrition occurs at Work-Life Balance Rating: {highest_wlb}")

print("\n" + "="*70)
print("5. Years at Company vs Attrition")

years_attrition = (
    eda_df[eda_df["Attrition"]=="Yes"]
    .groupby("YearsAtCompany")
    .size()
    .sort_values(ascending=False)
)

print(years_attrition)

peak_year = years_attrition.idxmax()

print(f"\nEmployees leave the most after {peak_year} years in the company.")

print("\n" + "="*70)
print("EDA Completed Successfully")
print("="*70)

# **Task 4: Neural Network Model Building & Comparison**

# Install TensorFlow if required in Google Colab:
# !pip install tensorflow

import os
import random
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv1D, MaxPooling1D,
    Flatten, Concatenate, BatchNormalization, Reshape
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------
# Recreate the processed data only once.
# The original Task 2 already created X, y, X_train, X_test,
# y_train, y_test and scaled data. We use those variables here.
# ------------------------------------------------------------

input_dim = X_train.shape[1]

# Convert targets to NumPy arrays
X_train_nn = np.asarray(X_train, dtype=np.float32)
X_test_nn = np.asarray(X_test, dtype=np.float32)
y_train_nn = np.asarray(y_train, dtype=np.float32)
y_test_nn = np.asarray(y_test, dtype=np.float32)

# Class weights for the imbalanced Attrition target
negative_count = np.sum(y_train_nn == 0)
positive_count = np.sum(y_train_nn == 1)
total_count = len(y_train_nn)

class_weight = {
    0: total_count / (2.0 * negative_count),
    1: total_count / (2.0 * positive_count)
}

print("Input features:", input_dim)
print("Class weights:", class_weight)

# ============================================================
# MODEL 1: MLP - MULTILAYER PERCEPTRON
# ============================================================

def build_mlp(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.30),
        Dense(64, activation="relu"),
        Dropout(0.25),
        Dense(32, activation="relu"),
        Dropout(0.20),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ============================================================
# MODEL 2: 1D CNN
# ============================================================

def build_cnn(input_dim):
    model = Sequential([
        Input(shape=(input_dim, 1)),
        Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(32, kernel_size=3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.30),
        Dense(32, activation="relu"),
        Dropout(0.20),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ============================================================
# MODEL 3: CNN + MLP HYBRID
# ============================================================

def build_cnn_mlp_hybrid(input_dim):
    inputs = Input(shape=(input_dim,))

    # CNN branch
    cnn_branch = Reshape((input_dim, 1))(inputs)
    cnn_branch = Conv1D(64, 3, padding="same", activation="relu")(cnn_branch)
    cnn_branch = BatchNormalization()(cnn_branch)
    cnn_branch = MaxPooling1D(2)(cnn_branch)
    cnn_branch = Conv1D(32, 3, padding="same", activation="relu")(cnn_branch)
    cnn_branch = BatchNormalization()(cnn_branch)
    cnn_branch = MaxPooling1D(2)(cnn_branch)
    cnn_branch = Flatten()(cnn_branch)

    # MLP branch
    mlp_branch = Dense(64, activation="relu")(inputs)
    mlp_branch = Dropout(0.25)(mlp_branch)
    mlp_branch = Dense(32, activation="relu")(mlp_branch)

    # Merge branches
    combined = Concatenate()([cnn_branch, mlp_branch])
    combined = Dense(64, activation="relu")(combined)
    combined = Dropout(0.30)(combined)
    combined = Dense(32, activation="relu")(combined)
    outputs = Dense(1, activation="sigmoid")(combined)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Build models
mlp = build_mlp(input_dim)
cnn = build_cnn(input_dim)
cnn_mlp = build_cnn_mlp_hybrid(input_dim)

print("\nMLP Architecture")
mlp.summary()

print("\n1D CNN Architecture")
cnn.summary()

print("\nCNN-MLP Hybrid Architecture")
cnn_mlp.summary()

# Early stopping
# A fresh callback is used for each model so training state is not shared.
def make_early_stopping():
    return EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

# ============================================================
# TRAIN MLP
# ============================================================

print("\n" + "=" * 70)
print("TRAINING MLP")
print("=" * 70)

history_mlp = mlp.fit(
    X_train_nn,
    y_train_nn,
    validation_split=0.20,
    epochs=100,
    batch_size=32,
    class_weight=class_weight,
    callbacks=[make_early_stopping()],
    verbose=1
)

# ============================================================
# TRAIN 1D CNN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING 1D CNN")
print("=" * 70)

X_train_cnn = X_train_nn.reshape(X_train_nn.shape[0], X_train_nn.shape[1], 1)
X_test_cnn = X_test_nn.reshape(X_test_nn.shape[0], X_test_nn.shape[1], 1)

history_cnn = cnn.fit(
    X_train_cnn,
    y_train_nn,
    validation_split=0.20,
    epochs=100,
    batch_size=32,
    class_weight=class_weight,
    callbacks=[make_early_stopping()],
    verbose=1
)

# ============================================================
# TRAIN CNN + MLP HYBRID
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CNN + MLP HYBRID")
print("=" * 70)

history_hybrid = cnn_mlp.fit(
    X_train_nn,
    y_train_nn,
    validation_split=0.20,
    epochs=100,
    batch_size=32,
    class_weight=class_weight,
    callbacks=[make_early_stopping()],
    verbose=1
)

# **Task 5: Model Evaluation**

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

print("\n" + "=" * 70)
print("TASK 5 - MODEL EVALUATION")
print("=" * 70)

# Probability predictions
mlp_prob = mlp.predict(X_test_nn, verbose=0).ravel()
cnn_prob = cnn.predict(X_test_cnn, verbose=0).ravel()
hybrid_prob = cnn_mlp.predict(X_test_nn, verbose=0).ravel()

# Binary predictions at the standard 0.50 threshold
mlp_pred = (mlp_prob >= 0.50).astype(int)
cnn_pred = (cnn_prob >= 0.50).astype(int)
hybrid_pred = (hybrid_prob >= 0.50).astype(int)


def evaluate_model(model_name, y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Stayed", "Left"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }

results = [
    evaluate_model("MLP", y_test, mlp_pred, mlp_prob),
    evaluate_model("1D CNN", y_test, cnn_pred, cnn_prob),
    evaluate_model("CNN-MLP Hybrid", y_test, hybrid_pred, hybrid_prob)
]

comparison_df = pd.DataFrame(results).round(4)

print("\nNEURAL NETWORK MODEL COMPARISON")
display(comparison_df)

# Use a reproducible project rule for selecting the model for detailed analysis.
# Primary metric: F1; tie-breakers: ROC-AUC, Recall, Precision.
comparison_ranked = comparison_df.sort_values(
    by=["F1 Score", "ROC-AUC", "Recall", "Precision"],
    ascending=False
).reset_index(drop=True)

best_model_name = comparison_ranked.iloc[0]["Model"]

if best_model_name == "MLP":
    best_model = mlp
    best_predictions = mlp_pred
    best_probabilities = mlp_prob
    best_X_test = X_test_nn
elif best_model_name == "1D CNN":
    best_model = cnn
    best_predictions = cnn_pred
    best_probabilities = cnn_prob
    best_X_test = X_test_cnn
else:
    best_model = cnn_mlp
    best_predictions = hybrid_pred
    best_probabilities = hybrid_prob
    best_X_test = X_test_nn

print("\nModel selected for detailed analysis:", best_model_name)

print("\nSelected Model Metrics:")
print("Accuracy :", round(accuracy_score(y_test, best_predictions), 4))
print("Precision:", round(precision_score(y_test, best_predictions, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, best_predictions, zero_division=0), 4))
print("F1 Score :", round(f1_score(y_test, best_predictions, zero_division=0), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test, best_probabilities), 4))

# **Task 6: Visualization**

import os
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("charts", exist_ok=True)

# ------------------------------------------------------------
# Chart 1: Attrition rate by Department
# ------------------------------------------------------------

department_attrition = (
    original_df.groupby("Department")["Attrition"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="Attrition Rate")
)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=department_attrition,
    x="Department",
    y="Attrition Rate",
    hue="Department",
    palette="viridis",
    legend=False
)
plt.title("Attrition Rate by Department")
plt.xlabel("Department")
plt.ylabel("Attrition Rate (%)")
plt.xticks(rotation=10)
plt.savefig("charts/chart1_department_attrition.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("Chart saved: charts/chart1_department_attrition.png")

# ------------------------------------------------------------
# Chart 2: Monthly income box plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))
sns.boxplot(
    data=original_df,
    x="Attrition",
    y="MonthlyIncome",
    hue="Attrition",
    palette="Set2",
    legend=False
)
plt.title("Monthly Income of Employees Who Left vs Stayed")
plt.xlabel("Attrition")
plt.ylabel("Monthly Income")
plt.savefig("charts/chart2_monthly_income_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("Chart saved: charts/chart2_monthly_income_boxplot.png")

# ------------------------------------------------------------
# Chart 3: Confusion matrix of selected neural network
# ------------------------------------------------------------

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed", "Left"],
    yticklabels=["Stayed", "Left"]
)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("charts/chart3_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("Chart saved: charts/chart3_confusion_matrix.png")

# ------------------------------------------------------------
# Chart 4: Top 10 feature importance using permutation importance
# ------------------------------------------------------------
# Neural networks do not have feature_importances_ like Random Forest.
# Permutation importance is therefore used. A feature is shuffled and
# the decrease in ROC-AUC is used as its importance score.

print("\nCalculating permutation feature importance...")

rng = np.random.RandomState(SEED)
X_test_for_importance = X_test_nn.copy()


def selected_model_predict(X_input):
    X_input = np.asarray(X_input)
    if best_model_name == "1D CNN":
        X_input = X_input.reshape(X_input.shape[0], X_input.shape[1], 1)
    return best_model.predict(X_input, verbose=0).ravel()

baseline_auc = roc_auc_score(
    y_test,
    selected_model_predict(X_test_for_importance)
)

importance_values = []

for column_index, feature_name in enumerate(X.columns):
    shuffled_X = X_test_for_importance.copy()
    shuffled_X[:, column_index] = rng.permutation(shuffled_X[:, column_index])

    shuffled_auc = roc_auc_score(
        y_test,
        selected_model_predict(shuffled_X)
    )

    importance_values.append(baseline_auc - shuffled_auc)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance_values
}).sort_values(
    by="Importance",
    ascending=False
).head(10)

print("\nTop 10 Features:")
display(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature",
    hue="Feature",
    palette="viridis",
    legend=False
)
plt.title(f"Top 10 Feature Importances - {best_model_name}")
plt.xlabel("Permutation Importance (ROC-AUC Decrease)")
plt.ylabel("Feature")
plt.savefig("charts/chart4_feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("Chart saved: charts/chart4_feature_importance.png")

# ------------------------------------------------------------
# Chart 5: ROC curves of all three neural networks
# ------------------------------------------------------------

mlp_fpr, mlp_tpr, _ = roc_curve(y_test, mlp_prob)
cnn_fpr, cnn_tpr, _ = roc_curve(y_test, cnn_prob)
hybrid_fpr, hybrid_tpr, _ = roc_curve(y_test, hybrid_prob)

mlp_auc = roc_auc_score(y_test, mlp_prob)
cnn_auc = roc_auc_score(y_test, cnn_prob)
hybrid_auc = roc_auc_score(y_test, hybrid_prob)

plt.figure(figsize=(8, 6))
plt.plot(mlp_fpr, mlp_tpr, label=f"MLP (AUC = {mlp_auc:.3f})")
plt.plot(cnn_fpr, cnn_tpr, label=f"1D CNN (AUC = {cnn_auc:.3f})")
plt.plot(hybrid_fpr, hybrid_tpr, label=f"CNN-MLP Hybrid (AUC = {hybrid_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.title("ROC Curve Comparison of Neural Networks")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.savefig("charts/chart5_roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("Chart saved: charts/chart5_roc_curve.png")

# Optional: validation-loss comparison
plt.figure(figsize=(9, 6))
plt.plot(history_mlp.history["val_loss"], label="MLP")
plt.plot(history_cnn.history["val_loss"], label="1D CNN")
plt.plot(history_hybrid.history["val_loss"], label="CNN-MLP Hybrid")
plt.title("Validation Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("charts/chart6_validation_loss.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("Optional Chart 6 saved: charts/chart6_validation_loss.png")

# **Task 7: HR Insights & Business Recommendations**

print("\n" + "=" * 70)
print("TASK 7 - HR INSIGHTS & BUSINESS RECOMMENDATIONS")
print("=" * 70)

# 1. Top three model-associated features
print("\n1. Top 3 Features Associated with the Selected Neural Network:")
top_3_features = feature_importance.head(3)["Feature"].tolist()
for i, feature in enumerate(top_3_features, start=1):
    print(f"{i}. {feature}")

# 2. Department and job role with highest observed attrition rate
print("\n2. Department with Highest Observed Attrition Rate:")
print(f"{highest_dept} ({highest_dept_rate:.2f}%)")

print("\nJob Role with Highest Observed Attrition Rate:")
print(f"{highest_job} ({highest_job_rate:.2f}%)")

# 3. Monthly income observation
print("\n3. Monthly Income Observation:")
if avg_yes < avg_no:
    print("Employees who left have a lower average monthly income in this dataset.")
else:
    print("Employees who left do not have a lower average monthly income in this dataset.")

# 4. HR recommendations
print("\n4. HR Recommendations:")
print("1. Review workload and overtime patterns, especially for groups identified as higher risk.")
print("2. Prioritize retention discussions and career-development support for departments and roles with higher observed attrition rates.")
print("3. Monitor work-life balance and business travel conditions when designing retention programs.")
print("4. Use regular employee feedback and one-to-one discussions to identify factors not captured in the dataset.")
print("5. Use model predictions as decision-support information rather than as the sole basis for HR decisions.")

# 5. Limitation
print("\n5. Model Limitation:")
print("The neural networks learn patterns only from the available HR dataset.")
print("They cannot capture unrecorded factors such as employee motivation, workplace culture, manager relationships, personal circumstances, or future organizational changes.")
print("Predictions should therefore support HR analysis rather than replace human judgment.")

# ============================================================
# FINAL OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROJECT EXECUTION COMPLETED")
print("=" * 70)
print("\nNeural Network Models:")
print("1. MLP")
print("2. 1D CNN")
print("3. CNN-MLP Hybrid")
print("\nGenerated Charts:")
print("1. charts/chart1_department_attrition.png")
print("2. charts/chart2_monthly_income_boxplot.png")
print("3. charts/chart3_confusion_matrix.png")
print("4. charts/chart4_feature_importance.png")
print("5. charts/chart5_roc_curve.png")
print("6. charts/chart6_validation_loss.png")
print("\nModel Comparison:")
display(comparison_df)
print("\nSelected Model for Detailed Analysis:")
print(best_model_name)
