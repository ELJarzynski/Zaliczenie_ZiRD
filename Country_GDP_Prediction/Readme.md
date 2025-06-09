# Country DGP Forecast Model Using Machine Learning

This project aims to build a high-quality regression model to forecast the GDP of countries with at least 25 years of historical data.
Initially, the dataset included only 3 rows, but we expanded it by feature engineering 6 additional rows from Wikipedia.
These rows represent differences in GDP over time and help improve prediction accuracy.

The final dataset contains 9 columns, reflecting various factors that may influence a country’s GDP.
We obtained the data by scraping tables from Wikipedia and used the Scikit-learn library to train and evaluate the machine learning models.

# Model Results
We tested 5 regression models. The following 4 produced the best results:
## Random Forest Regressor
![K Nearest Neighbors Error](https://github.com/ELJarzynski/Zaliczenie_ZiRD/blob/master/Country_GDP_Prediction/images/randomforest_output.png)

## Linear Regression
![Logistic Regression Error](https://github.com/ELJarzynski/Zaliczenie_ZiRD/blob/master/Country_GDP_Prediction/images/lr_output.png)

## Decision Tree Classifier
![Decision Tree Classifier Error](https://github.com/ELJarzynski/Zaliczenie_ZiRD/blob/master/Country_GDP_Prediction/images/tree_output.png)

## But the best one was XGBoost
![Random Forest Classifier Error](https://github.com/ELJarzynski/Zaliczenie_ZiRD/blob/master/Country_GDP_Prediction/images/xgboost_output.png)

# Importance of models

## Random forest regressor
![K Nearest Neighbors Error](https://github.com/ELJarzynski/Zaliczenie_ZiRD/blob/master/Country_GDP_Prediction/images/importance_RandomForest.png)

## Linear Regression
![Logistic Regression Error](https://github.com/ELJarzynski/Zaliczenie_ZiRD/blob/master/Country_GDP_Prediction/images/importance_LR.png)

## Decision Tree Classifier
![Decision Tree Classifier Error](https://github.com/ELJarzynski/Zaliczenie_ZiRD/blob/master/Country_GDP_Prediction/images/importance_Tree.png)

## The best one was XGBoost
![Random Forest Classifier Error](https://github.com/ELJarzynski/Zaliczenie_ZiRD/blob/master/Country_GDP_Prediction/images/importance_XGBoost.png)

# Summary
We are satisfied with the model's performance, which exceeded our initial expectations.
Interestingly, the Random Forest Regressor ranked only fourth, while the Decision Tree Regressor performed slightly better.
Ultimately, the XGBoost Regressor outperformed all other models, delivering the most accurate GDP forecasts across multiple countries.