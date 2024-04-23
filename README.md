# ML-Project-to-predict-Customer-Churn

This project contains all the information and code regarding a project on customer churn for a large telecommunications company. This is a Machine learning classification project where we aim to predict customer churn

## 📃Table of contents

- Introduction
- Installation
- Usage
- Data
- Preliminary cleaning
- EDA
- Univariate and Bivariate analysis
- Models
- Pipelines
- Models after balancing dataset with SMOTE
- Feature importance and visualisation
- AUC-ROC

## 💻Installation

- Install the required dependencies: pip install -r requirements.txt

## 👨‍💻Data

The dataset for this project was acquired from different sources (Microsoft SQL server and websites). The data didn't need much in terms of cleaning. The two training datasets were concatenated immediately since they had the same columns.

Cleaning done for this dataset was mostly replacing values, the 5 missing values in totalCharges was replaced with values from monthly charges

The dataset is moderately imbalanced, the Yes values in the target column were about 75% of the dataset against about 25 percent for the No values.

## ✈️EDA

The first training dataset has 3000 rows, the second dataset has 2043 rows. After concatenation the training dataset hd a total of 5043 rows. There were missing Values in the following columns: Churn, Streaming Movies, Total charges and 6 other columns. Some columns had both 'True and False' values and 'Yes and No' values, we changed all of those to become yes and no. After running descriptive statistics we can tell that the Total charges column had a rather large standard deviation even though there were no outliers

[Countplot for Total charges](Picture/Countplot for Total charges.png)

[Violinplot using churn and TotalCharges](Picture/Violinplot using churn and TotalCharges.png)

[Distribution of Churn by Payment method](Picture\Distribution of Churn by Payment method.png)

The random state select for all models was 42 as it is perceived to be one of the best random states to use

## 🧹Pipelines and transformation

The data was split into the usual suspects(X train and test and y train and test), the data was then grouped into categorical and numerical by using their 'dtype'.
The data was put into pipelines, for the categorical data we imputed all missing values with the mode of their respective columns then used the one hot encoder to transform the values. In the numerical pipeline we used a standard scaler due to the large standard deviation in out total charges. The pipelines were passed into the preprocessor to do its thing.

Please note that i had separated the dependent variable and encoded it with a label encoder.

## 👨‍💻Model

For this project we will be training 8 different models to see which one gives us the best results. I selected these models as a recommendation form ChatGPT:

- Logistic_regression
- Decision_tree
- Random_forest
- Support_vector
- KNN (KNeighborsClassifier)
- Gradient_boost
- Naive_bayes
- XGBoost

The confusion matrix was used to calculate the metrics for the Models:
The logistic regression model performed best with the provided dataset, with about 80% for it's F1 score, followed closely by support vector and gradient boost with 79% F1 scores each.

## Models with balanced dataset using SMOTE

As mentioned earlier, the dataset provided was moderately imbalanced, i employed the SMOTE technique to balance the dataset and provide the models with enough data on both ends. the SMOTE technique was used before of it's interpolation method as opposed to the duplication and deletion of relevant data by other balancing techniques.

After using the SMOTE technique some models appeared to ave performed better: Gradient boost came out on top with 80% followed by random forest and gradient boost with 78% each

## Feature importance and Visualisation

[Feature Importance with unbalanced data](Picture\Feature Importance with unbalanced data.png)

[Feature Importance after SMOTE](Picture\Feature Importance after SMOTE.png)
