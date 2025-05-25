# Titanic---Machine-Learning-from-Disaster
This repository contains a machine learning project where I built a Logistic Regression model to predict survival outcomes on the Titanic dataset from Kaggle. The goal was to apply different data preprocessing techniques, feature engineering, and hyperparameter tuning using L1 regularization (Lasso) to improve model performance. The Final accuracy achieved was 78.229%.

## Overview
The Titanic dataset is a classic problem in machine learning, where we aim to predict whether a passenger survived the disaster based on features like age, sex, class, and family size. The dataset can be found on Kaggle:  
🔗 [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

- **Data Cleaning**: Filled missing values for Age and Embarked columns, removed redundant columns like Cabin, Name, and Ticket.
- **Feature Engineering**: Extracted and mapped titles from passenger names, created a `FamilySize` feature by combining siblings, spouses, parents, and children. Mapped categorical variables such as Sex and Embarked into numerical format.
- **Modeling**: Built a Logistic Regression model with L1 regularization (Lasso) to encourage feature sparsity and reduce overfitting. The model was optimized using **GridSearchCV** to find the best hyperparameters (`C` values).
- **Cross-Validation**: Applied K-Fold Cross-Validation within GridSearchCV for robust performance estimates.
- **Performance**: Achieved a final Kaggle submission accuracy of **78.229%**.
## Libraries Used

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Highlights

- **Feature Engineering**: Mapped titles into meaningful categories and created new features like `FamilySize`.
- **Regularization**: Implemented L1 regularization with Logistic Regression to improve model generalization and feature selection.
- **Hyperparameter Tuning**: Used GridSearchCV to find the optimal regularization strength (`C` parameter).
- **Model Evaluation**: Evaluated model performance using cross-validation and confusion matrices.

## Results
After applying feature engineering techniques, data preprocessing, and regularization (L1 penalty) using Logistic Regression, the model achieved a validation accuracy of 84% on the local train-validation split. Upon submitting predictions to the Kaggle Titanic competition, the model scored an accuracy of 0.77272 on the public leaderboard. This discrepancy highlights the importance of ensuring model generalization beyond local datasets. 

## Final Notes

This project demonstrates how simple yet effective preprocessing and hyperparameter tuning can improve the performance of machine learning models. Further improvements could be made by experimenting with different algorithms (e.g., Random Forest, XGBoost) or by creating more advanced features.
