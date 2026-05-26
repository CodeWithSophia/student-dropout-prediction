# 🎓 Student Dropout Prediction — Machine Learning Project

## Overview
This project builds an end-to-end machine learning system to predict whether a university student is at risk of dropping out based on academic performance, financial status, and demographic data. Early identification of at-risk students enables timely intervention and improves retention rates.

## Problem Statement
Student dropout is a major challenge for higher education institutions worldwide. It represents a loss for students, families, and institutions alike. This project explores whether academic and financial indicators collected early in a student's academic journey can predict dropout risk before it is too late to intervene.

## Dataset
- **Source:** [Kaggle - Higher Education Predictors of Student Retention](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)
- **Size:** 3,630 students (after removing enrolled students with no final outcome)
- **Target variable:** `Target` (1 = Dropout, 0 = Graduate)
- **Class distribution:** 2,209 Graduates (60.8%) vs 1,421 Dropouts (39.2%)

## Key Features
| Feature | Description |
|---|---|
| Curricular units approved | Number of subjects passed per semester |
| Curricular units grade | Average grade per semester |
| Tuition fees up to date | Whether student is current with payments |
| Scholarship holder | Whether student has financial support |
| Debtor | Whether student owes money to institution |
| Age at enrollment | Age when student first enrolled |
| International | Whether student is an international student |
| Unemployment rate | National unemployment rate at enrollment |

## EDA Findings
- Students **without scholarships** were **3.5x more likely** to drop out
- Students **not up to date with tuition fees** had a **94% dropout rate**
- Academic performance in **both semesters** was the strongest overall predictor
- International students showed elevated dropout risk due to additional pressures
- The dataset had a moderate class imbalance handled using SMOTE

## Methodology
1. **Exploratory Data Analysis** — Investigated distributions, financial patterns and academic performance
2. **Data Cleaning** — Removed enrolled students with no final outcome, confirmed no missing values
3. **Class Imbalance** — Applied SMOTE to balance dropout vs graduate cases in training data only
4. **Feature Scaling** — Applied StandardScaler to normalise feature ranges before modelling
5. **Modelling** — Trained and compared Logistic Regression and Random Forest classifiers
6. **Evaluation** — Prioritised Recall and AUC to ensure maximum detection of at-risk students
7. **Deployment** — Built and deployed interactive Streamlit web application

## Model Performance
| Model | Recall (Dropout) | AUC Score | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.92 | 0.972 | 92% |
| Random Forest | 0.92 | 0.969 | 93% |

## Key Finding
Both models achieved outstanding performance with 92% recall and AUC scores above 0.969. Logistic Regression was selected as the final model due to its higher AUC score and superior interpretability — critical in educational settings where decisions affect students' lives.

## Feature Importance
Top predictors identified by the model:
1. **Curricular units approved — 2nd semester (3.06)** — strongest predictor overall
2. **Curricular units approved — 1st semester (2.32)** — early academic performance is critical
3. **Tuition fees up to date (1.10)** — financial status is a key early warning indicator
4. **International student status (0.94)** — international students face elevated risk

## Live Demo
🚀 **[Try the Student Dropout Risk Prediction App](#)** — *(link coming soon)*

## Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Streamlit

## Project Structure
