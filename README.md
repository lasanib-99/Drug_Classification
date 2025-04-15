# Drug Classification

This project implements a machine learning model to classify drugs into five categories (DrugY, drugA, drugB, drugC, drugX) based on patient attributes such as age, sex, blood pressure (BP), cholesterol levels, and sodium/potassium levels.​

## Dataset

The dataset is sourced from UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified). It contains 200 instances with the following attributes:​
- Age
- Sex
- Blood Pressure (BP)
- Cholesterol
- Na_to_K (Sodium to Potassium ratio)
- Drug (target variable)

## Project Structure

- 'drug_classifier.ipynb' – Jupyter Notebook containing data preprocessing, model training, and evaluation.
- 'drug200.csv' – Dataset file.
- 'README.md' – Project overview and instructions.​

## Getting Started

### 1. Clone the repository:
```
git clone https://github.com/lasanib-99/Drug_Classification.git
cd Drug_Classification
```

### 2. Install dependencies:
Ensure you have Python 3.x installed. Then, install the required packages:
```
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Run the notebook:
Launch Jupyter Notebook and open drug_classifier.ipynb:
```
jupyter notebook
```

## Model Overview

The project explores various classification algorithms, including:​
- Decision Tree
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest​
Each model is evaluated using metrics such as accuracy, precision, recall, and F1-score.​

## Results

The models are compared based on their performance metrics to determine the most effective algorithm for drug classification.
