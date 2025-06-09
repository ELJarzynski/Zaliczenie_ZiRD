"""
This script trains and evaluates several regression models to predict GDP values.
Steps:
1. Loads preprocessed data from a Parquet file.
2. Splits the dataset into training, validation, and test sets.
3. Trains different regression models on the training set.
4. Evaluates each model using MSE, MAE, and R² on the validation set.
5. Visualizes actual vs predicted GDP values for each model.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Set matplotlib backend to avoid compatibility issues with PyCharm
import matplotlib
matplotlib.use('TkAgg')


def evaluate_model(model, X_val, y_val, model_name):
    """
    Evaluates the given regression model and displays performance metrics and a scatter plot.
    """
    y_pred = model.predict(X_val)

    # Display performance metrics
    print(f"\nModel: {model_name}")
    print("MSE:", mean_squared_error(y_val, y_pred))
    print("MAE:", mean_absolute_error(y_val, y_pred))
    print("R²:", r2_score(y_val, y_pred))

    # Visualization: Actual vs Predicted
    plt.figure(figsize=(7, 5))
    plt.scatter(y_val, y_pred, alpha=0.6)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', label="Ideal fit")
    plt.xlabel("Actual GDP")
    plt.ylabel("Predicted GDP")
    plt.title(f"Actual vs Predicted GDP - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Load preprocessed dataset
file_path = "prepared_data.parquet"
df = pd.read_parquet(file_path)

# Drop rows with missing GDP values
df = df.dropna(subset=["GDP"])

# Define target and features
target_column = "GDP"
X = df.drop(columns=[target_column])
y = df[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further validation split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Dictionary of models to evaluate
models = {
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    evaluate_model(model, X_val, y_val, name)
