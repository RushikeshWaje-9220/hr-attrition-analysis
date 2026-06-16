# =============================================================================
# HR Attrition Analysis & Prediction
# Author: Rushikesh Waje
# Dataset: IBM HR Analytics Employee Attrition Dataset (1,470 records)
# Tools: Python | Scikit-learn | Pandas | Matplotlib | Seaborn
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")

# Set plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# =============================================================================
# STEP 1: Load Dataset
# =============================================================================

print("=" * 60)
print("HR ATTRITION ANALYSIS & PREDICTION")
print("=" * 60)

# Using IBM HR dataset via URL (publicly available)
url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/IBM-HR-Employee-Attrition.csv"

try:
    df = pd.read_csv(url)
    print(f"\n[✔] Dataset loaded successfully: {df.shape[0]} records, {df.shape[1]} features")
except Exception:
    # Fallback: generate synthetic IBM HR-style dataset
    print("\n[!] Remote dataset unavailable. Generating synthetic IBM HR dataset...")
    np.random.seed(42)
    n = 1470

    departments = ["Sales", "Research & Development", "Human Resources"]
    job_roles = ["Sales Executive", "Research Scientist", "Laboratory Technician",
                 "Manufacturing Director", "Healthcare Representative", "Manager",
                 "Sales Representative", "Research Director", "Human Resources"]
    education_fields = ["Life Sciences", "Other", "Medical", "Marketing",
                        "Technical Degree", "Human Resources"]
    marital_statuses = ["Single", "Married", "Divorced"]
    overtime_vals = ["Yes", "No"]

    df = pd.DataFrame({
        "Age": np.random.randint(18, 60, n),
        "Attrition": np.random.choice(["Yes", "No"], n, p=[0.16, 0.84]),
        "BusinessTravel": np.random.choice(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], n),
        "DailyRate": np.random.randint(100, 1500, n),
        "Department": np.random.choice(departments, n),
        "DistanceFromHome": np.random.randint(1, 30, n),
        "Education": np.random.randint(1, 6, n),
        "EducationField": np.random.choice(education_fields, n),
        "EmployeeCount": 1,
        "EmployeeNumber": range(1, n + 1),
        "EnvironmentSatisfaction": np.random.randint(1, 5, n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "HourlyRate": np.random.randint(30, 100, n),
        "JobInvolvement": np.random.randint(1, 5, n),
        "JobLevel": np.random.randint(1, 6, n),
        "JobRole": np.random.choice(job_roles, n),
        "JobSatisfaction": np.random.randint(1, 5, n),
        "MaritalStatus": np.random.choice(marital_statuses, n),
        "MonthlyIncome": np.random.randint(1000, 20000, n),
        "MonthlyRate": np.random.randint(2000, 27000, n),
        "NumCompaniesWorked": np.random.randint(0, 10, n),
        "Over18": "Y",
        "OverTime": np.random.choice(overtime_vals, n, p=[0.28, 0.72]),
        "PercentSalaryHike": np.random.randint(11, 25, n),
        "PerformanceRating": np.random.choice([3, 4], n, p=[0.85, 0.15]),
        "RelationshipSatisfaction": np.random.randint(1, 5, n),
        "StandardHours": 80,
        "StockOptionLevel": np.random.randint(0, 4, n),
        "TotalWorkingYears": np.random.randint(0, 40, n),
        "TrainingTimesLastYear": np.random.randint(0, 7, n),
        "WorkLifeBalance": np.random.randint(1, 5, n),
        "YearsAtCompany": np.random.randint(0, 40, n),
        "YearsInCurrentRole": np.random.randint(0, 18, n),
        "YearsSinceLastPromotion": np.random.randint(0, 15, n),
        "YearsWithCurrManager": np.random.randint(0, 17, n),
    })

    # Make attrition more realistic: higher for young tenure employees
    df.loc[df["YearsAtCompany"] < 2, "Attrition"] = np.random.choice(
        ["Yes", "No"], (df["YearsAtCompany"] < 2).sum(), p=[0.40, 0.60]
    )
    print(f"[✔] Synthetic dataset generated: {df.shape[0]} records, {df.shape[1]} features")

# =============================================================================
# STEP 2: Exploratory Data Analysis (EDA)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print(f"\nDataset Shape   : {df.shape}")
print(f"Missing Values  : {df.isnull().sum().sum()}")
print(f"\nAttrition Distribution:\n{df['Attrition'].value_counts()}")
attrition_rate = (df["Attrition"] == "Yes").mean() * 100
print(f"\nOverall Attrition Rate: {attrition_rate:.1f}%")

# --- Plot 1: Attrition Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df["Attrition"].value_counts().plot(
    kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"], edgecolor="black"
)
axes[0].set_title("Attrition Distribution", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Attrition")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=0)

# Attrition by Department
dept_attrition = df.groupby("Department")["Attrition"].apply(
    lambda x: (x == "Yes").mean() * 100
).reset_index()
dept_attrition.columns = ["Department", "AttritionRate"]
axes[1].bar(dept_attrition["Department"], dept_attrition["AttritionRate"],
            color=["#3498db", "#e74c3c", "#f39c12"], edgecolor="black")
axes[1].set_title("Attrition Rate by Department (%)", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Department")
axes[1].set_ylabel("Attrition Rate (%)")
axes[1].tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig("plots/01_attrition_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("[✔] Plot saved: 01_attrition_distribution.png")

# --- Plot 2: Attrition by Age Group, Tenure, and Salary Band ---
import os
os.makedirs("plots", exist_ok=True)

df["AgeGroup"] = pd.cut(df["Age"], bins=[18, 25, 35, 45, 60],
                         labels=["18-25", "26-35", "36-45", "46-60"])
df["TenureBand"] = pd.cut(df["YearsAtCompany"], bins=[-1, 2, 5, 10, 40],
                           labels=["<2 yrs", "2-5 yrs", "5-10 yrs", "10+ yrs"])
df["SalaryBand"] = pd.cut(df["MonthlyIncome"], bins=[0, 3000, 6000, 10000, 25000],
                           labels=["Low", "Mid", "High", "Very High"])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, col, title, color in zip(
    axes,
    ["AgeGroup", "TenureBand", "SalaryBand"],
    ["Attrition Rate by Age Group", "Attrition Rate by Tenure", "Attrition Rate by Salary Band"],
    [["#3498db"] * 4, ["#e74c3c"] * 4, ["#f39c12"] * 4]
):
    grp = df.groupby(col)["Attrition"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).reset_index()
    grp.columns = [col, "AttritionRate"]
    ax.bar(grp[col].astype(str), grp["AttritionRate"], color=color, edgecolor="black")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Attrition Rate (%)")
    ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("plots/02_attrition_patterns.png", dpi=150, bbox_inches="tight")
plt.show()
print("[✔] Plot saved: 02_attrition_patterns.png")

# Tenure insight
tenure_yes = df[df["TenureBand"] == "<2 yrs"]["Attrition"].eq("Yes").mean()
tenure_overall = df["Attrition"].eq("Yes").mean()
ratio = tenure_yes / tenure_overall if tenure_overall > 0 else 0
print(f"\n[Insight] Employees with <2 yrs tenure have {ratio:.1f}x higher attrition risk")

# --- Plot 3: Correlation Heatmap ---
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(14, 10))
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm", linewidths=0.5,
            cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("[✔] Plot saved: 03_correlation_heatmap.png")

# --- Plot 4: OverTime vs Attrition ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ot_attr = df.groupby("OverTime")["Attrition"].value_counts(normalize=True).unstack() * 100
ot_attr.plot(kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"], edgecolor="black")
axes[0].set_title("Attrition Rate by OverTime", fontsize=13, fontweight="bold")
axes[0].set_xlabel("OverTime")
axes[0].set_ylabel("Percentage (%)")
axes[0].tick_params(axis="x", rotation=0)
axes[0].legend(["No Attrition", "Attrition"])

role_attr = df.groupby("JobRole")["Attrition"].apply(
    lambda x: (x == "Yes").mean() * 100
).sort_values(ascending=True)
role_attr.plot(kind="barh", ax=axes[1], color="#3498db", edgecolor="black")
axes[1].set_title("Attrition Rate by Job Role (%)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Attrition Rate (%)")
plt.tight_layout()
plt.savefig("plots/04_overtime_jobrole_attrition.png", dpi=150, bbox_inches="tight")
plt.show()
print("[✔] Plot saved: 04_overtime_jobrole_attrition.png")

# =============================================================================
# STEP 3: Feature Engineering
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 60)

# Drop non-predictive columns
drop_cols = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Drop derived columns used for plotting
df.drop(columns=["AgeGroup", "TenureBand", "SalaryBand"], inplace=True, errors="ignore")

# Encode target variable
df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)
df.drop(columns=["Attrition"], inplace=True)

# Label encode categorical columns (job role, department, overtime, etc.)
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
    print(f"  [Encoded] {col}")

print(f"\n[✔] Categorical variables encoded: {len(cat_cols)} columns")
print(f"[✔] Final feature matrix shape: {df.shape}")

# =============================================================================
# STEP 4: Handle Class Imbalance (Oversampling minority class)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: HANDLING CLASS IMBALANCE")
print("=" * 60)

X = df.drop(columns=["AttritionFlag"])
y = df["AttritionFlag"]

print(f"Class distribution before resampling:\n{y.value_counts()}")

# Upsample minority class
df_majority = df[y == 0]
df_minority = df[y == 1]
df_minority_upsampled = resample(df_minority, replace=True,
                                  n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

X_bal = df_balanced.drop(columns=["AttritionFlag"])
y_bal = df_balanced["AttritionFlag"]
print(f"Class distribution after resampling:\n{y_bal.value_counts()}")
print("[✔] Class imbalance handled via oversampling")

# =============================================================================
# STEP 5: Train-Test Split & Scaling
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"\n[✔] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# =============================================================================
# STEP 6: Model Training — Logistic Regression
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: MODEL TRAINING")
print("=" * 60)

# --- Logistic Regression ---
lr = LogisticRegression(max_iter=500, random_state=42, C=0.5)
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)
y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]

acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)
print(f"\nLogistic Regression  |  Accuracy: {acc_lr:.2%}  |  AUC: {auc_lr:.2f}")
print(classification_report(y_test, y_pred_lr, target_names=["No Attrition", "Attrition"]))

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)
print(f"\nRandom Forest        |  Accuracy: {acc_rf:.2%}  |  AUC: {auc_rf:.2f}")
print(classification_report(y_test, y_pred_rf, target_names=["No Attrition", "Attrition"]))

# Best model summary
best_model = "Random Forest" if auc_rf >= auc_lr else "Logistic Regression"
best_acc = acc_rf if auc_rf >= auc_lr else acc_lr
best_auc = max(auc_rf, auc_lr)
print(f"\n[✔] Best Model: {best_model}  |  Accuracy: {best_acc:.2%}  |  AUC: {best_auc:.2f}")

# =============================================================================
# STEP 7: Model Evaluation & Visualizations
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: MODEL EVALUATION")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Confusion Matrix (Random Forest) ---
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["No Attr.", "Attrition"],
            yticklabels=["No Attr.", "Attrition"])
axes[0].set_title("Confusion Matrix — Random Forest", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# --- ROC Curves ---
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
axes[1].plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={auc_lr:.2f})",
             color="#3498db", linewidth=2)
axes[1].plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.2f})",
             color="#e74c3c", linewidth=2)
axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[1].set_title("ROC Curve Comparison", fontsize=13, fontweight="bold")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()

# --- Feature Importance (Random Forest) ---
feat_imp = pd.Series(rf.feature_importances_, index=X_train.columns)
top15 = feat_imp.nlargest(15).sort_values()
top15.plot(kind="barh", ax=axes[2], color="#2ecc71", edgecolor="black")
axes[2].set_title("Top 15 Feature Importances (RF)", fontsize=13, fontweight="bold")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("plots/05_model_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()
print("[✔] Plot saved: 05_model_evaluation.png")

# =============================================================================
# STEP 8: Key Insights Summary
# =============================================================================

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

top_features = feat_imp.nlargest(5).index.tolist()
print(f"\nTop 5 Attrition Drivers: {', '.join(top_features)}")
print(f"\nEmployees with <2 years tenure have ~{ratio:.1f}x higher attrition risk")
print(f"Overall attrition rate: {attrition_rate:.1f}%")
print(f"\nModel Performance Summary:")
print(f"  Logistic Regression  →  Accuracy: {acc_lr:.2%}  |  AUC: {auc_lr:.2f}")
print(f"  Random Forest        →  Accuracy: {acc_rf:.2%}  |  AUC: {auc_rf:.2f}")
print(f"\n[✔] Analysis Complete. All plots saved to /plots/ directory.")
