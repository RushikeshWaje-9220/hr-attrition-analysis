# hr-attrition-analysis
Employee attrition prediction using Logistic Regression &amp; Random Forest | Python | Scikit-learn
# 🧑‍💼 HR Attrition Analysis & Prediction

> Predicting employee attrition using machine learning on the IBM HR Analytics dataset (1,470 records).

---

## 📌 Project Overview

Employee attrition is a critical challenge for organizations. This project performs end-to-end analysis of the IBM HR dataset to:

- Identify **key drivers of employee attrition** using EDA and correlation analysis
- Engineer features from categorical variables (job role, department, overtime)
- Handle **class imbalance** via oversampling
- Build and compare **Logistic Regression** and **Random Forest** classifiers
- Visualize attrition patterns by **age group, tenure, and salary band**

---

## 📊 Key Results

| Model                | Accuracy | AUC Score |
|---------------------|----------|-----------|
| Logistic Regression  | ~86%     | 0.82      |
| Random Forest        | ~97%     | 0.99      |

**Key Insight:** Employees with **< 2 years tenure** have **3× higher attrition risk** compared to the overall workforce.

---

## 🗂️ Project Structure

```
hr_attrition_project/
│
├── hr_attrition_analysis.py    # Main analysis and ML pipeline
├── plots/                      # Auto-generated visualizations
│   ├── 01_attrition_distribution.png
│   ├── 02_attrition_patterns.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_overtime_jobrole_attrition.png
│   └── 05_model_evaluation.png
└── README.md
```

---

## 🔧 Tech Stack

| Category       | Tools / Libraries                        |
|----------------|------------------------------------------|
| Language       | Python 3.x                               |
| Data Handling  | Pandas, NumPy                            |
| Visualization  | Matplotlib, Seaborn                      |
| ML Models      | Scikit-learn (LogisticRegression, RandomForestClassifier) |
| Resampling     | imbalanced-learn (`resample`)            |
| Evaluation     | Accuracy, AUC-ROC, Confusion Matrix      |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/hr-attrition-analysis.git
cd hr-attrition-analysis
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

**3. Run the analysis**
```bash
python hr_attrition_analysis.py
```

> All plots will be saved automatically to the `/plots/` folder.

---

## 📈 Visualizations

The script generates 5 plots:

1. **Attrition Distribution** — Overall & by Department
2. **Attrition Patterns** — By Age Group, Tenure, Salary Band
3. **Correlation Heatmap** — Feature relationships
4. **OverTime & Job Role** — High-risk segments
5. **Model Evaluation** — Confusion Matrix, ROC Curves, Feature Importance

---

## 💡 Key Insights

- **OverTime** employees have significantly higher attrition rates
- **Sales Representatives** and **Laboratory Technicians** show highest role-specific attrition
- **MonthlyIncome**, **TotalWorkingYears**, and **YearsAtCompany** are top predictors
- Employees with **<2 years tenure** are at 3× higher risk of leaving
- Class imbalance (84% No / 16% Yes) was addressed using minority class oversampling

---

## 📚 Dataset

**IBM HR Analytics Employee Attrition Dataset**
- 1,470 employee records
- 35 features including demographics, job info, satisfaction scores
- Target variable: `Attrition` (Yes / No)
- Publicly available on [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

## 👤 Author

**Rushikesh Waje**  
MCA Student | Data Analyst  
📧 rushikeshwaje39@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/rushikesh-waje)
