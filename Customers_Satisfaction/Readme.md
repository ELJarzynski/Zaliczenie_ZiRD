# Airline Customer Satisfaction Prediction Using Machine Learning

This project aims to achieve high-quality classification of airline customer satisfaction. 
The dataset consists of 22 columns, which present factors that influence passenger satisfaction. 

We are using the dataset from [Kaggle](https://www.kaggle.com/datasets/yakhyojon/customer-satisfaction-in-airline) 
and will employ the Scikit-learn library to train the machine learning model.

# Model Results

## K Nearest Neighbors
![K Nearest Neighbors Error](https://github.com/ELJarzynski/UM-Customer-Airline-Satisfaction-Prediction-RandomForestClassifier/blob/main/images/errorKNN.png)

## Logistic Regression
![Logistic Regression Error](https://github.com/ELJarzynski/UM-Customer-Airline-Satisfaction-Prediction-RandomForestClassifier/blob/main/images/errorLR.png)

## Decision Tree Classifier
![Decision Tree Classifier Error](https://github.com/ELJarzynski/UM-Customer-Airline-Satisfaction-Prediction-RandomForestClassifier/blob/main/images/errorTree.png)

## Random Forest Classifier
![Random Forest Classifier Error](https://github.com/ELJarzynski/UM-Customer-Airline-Satisfaction-Prediction-RandomForestClassifier/blob/main/images/errorForest.png)

# Summary

The Random Forest Classifier achieved the best results with:
- **Accuracy:** 0.952
- **Precision:** 0.969
- **Recall:** 0.943

# Confusion Matrix
![Confusion Matrix](https://github.com/ELJarzynski/UM-Customer-Airline-Satisfaction-Prediction-RandomForestClassifier/blob/main/images/CMDRF.png)