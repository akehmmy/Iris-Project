# CodeAlpha_IrisFlowerClassification

**Data Science Internship — Task 1: Iris Flower Classification**

## Overview
This project builds and compares several machine learning models to classify Iris flowers into one of three species — *setosa*, *versicolor*, or *virginica* — based on four measurements: sepal length, sepal width, petal length, and petal width.

## Workflow
1. **Load & inspect** the dataset (150 samples, 4 features, 3 balanced classes, no missing values or duplicates)
2. **Clean** the data (drop the `Id` column)
3. **Exploratory Data Analysis**: class distribution, pairplot, correlation heatmap, boxplots by species
4. **Preprocess**: train/test split (80/20, stratified) + feature scaling with `StandardScaler`
5. **Train & compare** four models:
   - Logistic Regression
   - K-Nearest Neighbors
   - Support Vector Machine (linear kernel)
   - Random Forest
6. **Evaluate** the best model with a classification report and confusion matrix

## Results
| Model | Accuracy |
|---|---|
| Support Vector Machine | **100%** |
| Logistic Regression | 93.3% |
| K-Nearest Neighbors | 93.3% |
| Random Forest | 90.0% |

The SVM achieved perfect precision, recall, and F1-score across all three species on the test set, with zero misclassifications.

## Tech Stack
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook Iris_Flower_Classification.ipynb
```

## Dataset
`Iris.csv` — 150 samples, provided by CodeAlpha.

## Author
Akor Emmanuel — CodeAlpha Data Science Intern (Student ID: CA/DF1/305488)
