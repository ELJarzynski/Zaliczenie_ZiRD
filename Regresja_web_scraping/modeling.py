"""
This script trains and evaluates a classification model to predict airline customer satisfaction.
Steps:
1. Loads preprocessed data from a Parquet file.
2. Splits the dataset into training, validation, and test sets.
3. Trains a RandomForestClassifier on the training set.
4. Evaluates the model on the validation set using recall, precision, accuracy, and confusion matrix.
5. Visualizes the confusion matrix.
"""

import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def metrics(y_validation, y_predicted_val):
    """Displays performance metrics for a classification model."""
    print("Recall:", recall_score(y_validation, y_predicted_val))
    print("Precision:", precision_score(y_validation, y_predicted_val))
    print("Accuracy:", accuracy_score(y_validation, y_predicted_val))
    print("Confusion Matrix:\n", confusion_matrix(y_validation, y_predicted_val))


# Load preprocessed dataset
file_directory = "prepared_data.parquet"
df = pd.read_parquet(file_directory)

# Print column names for debugging (optional)
# print(df.columns)

"""Train-test split"""
# Define the target column
target_column = "satisfaction"

# Split features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

"""Validation split"""
# Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

"""Model initialization"""
# RandomForestClassifier was chosen due to its strong performance on this dataset
model = RandomForestClassifier(
    n_estimators=1000,
    criterion='log_loss',
    max_depth=40,
    min_samples_split=2,
    min_samples_leaf=4,
    class_weight='balanced_subsample',
    n_jobs=-1,
)

# Train the model
model.fit(X_train, y_train)

# Predict on validation set
y_predict_val = model.predict(X_val)

# Display performance metrics
metrics(y_val, y_predict_val)

# Confusion matrix visualization
cm = confusion_matrix(y_val, y_predict_val)
CMD = ConfusionMatrixDisplay(confusion_matrix=cm)
CMD.plot(cmap='Blues')
plt.show()
