# Banking-Domain
# Fraud Detection Model

This repository contains the code for building and evaluating various machine learning models to predict fraudulent transactions based on given datasets. The process includes data preprocessing, feature engineering, model training, evaluation, and comparison of different algorithms.

## Table of Contents
- [Data Description](#data-description)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Future Work](#future-work)
- [Dependencies](#dependencies)
- [How to Use](#how-to-use)


## Data Description
The datasets used in this project include:
- `Geo_scores.csv`: Contains geographical scores for each transaction.
- `Instance_scores.csv`: Contains instance scores for each transaction.
- `Lambda_wts.csv`: Contains lambda weights for each group.
- `Qset_tats.csv`: Contains qset normalized TAT for each transaction.
- `test_share.csv`: Contains test data.
- `train.csv`: Contains training data.

## Data Preprocessing
1. **Reading the Data**:
   - Load all the datasets using `pd.read_csv()`.
   - Check the shape and columns of each dataset.
   - Count the unique values in key columns.

2. **Handling Missing Values**:
   - Fill missing values in `geo_score` and `qsets_normalized_tat` columns with their median values.

3. **Grouping and Merging**:
   - Group the `geo`, `qset`, and `instance` data by `id` and compute their mean.
   - Merge these datasets with the combined train and test data.

4. **Creating a Combined Dataset**:
   - Combine the train and test datasets into `all_data`.

## Exploratory Data Analysis
- Check for missing values.
- Describe the datasets to understand their statistical properties.
- Visualize correlations using a heatmap.
- Plot distributions and boxplots for numerical features to identify outliers.

## Feature Engineering
- Drop irrelevant columns (`id`, `Group`, `Target`, `data`).
- Separate features (X) and target (y) for training data.

## Model Building
The following machine learning models were trained and evaluated:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **XGBoost Classifier**
5. **Support Vector Machine**
6. **K-Nearest Neighbors**
7. **Naive Bayes**
8. **Voting Classifier**
9. **Stacking Classifier**

Additionally, anomaly detection models were implemented:
- **Isolation Forest**
- **Local Outlier Factor**
- **One-Class SVM**

## Model Evaluation
- **Confusion Matrix**: Displayed for both training and testing predictions.
- **Classification Report**: Provided precision, recall, and F1-score.
- **Accuracy Score**: Calculated for each model.
- **Bar Plot**: Visual comparison of the accuracy of different models.

## Future Work
- **Anomaly Detection**: Implement models such as Isolation Forest, Local Outlier Factor, and One-Class SVM for anomaly detection.
- **Neural Networks**: Explore using Multi-Layer Perceptron (MLP) or Deep Neural Networks (DNN).

## Dependencies
To run the code, you need the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

You can install these libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

git clone <repository-url>
cd <repository-directory>

