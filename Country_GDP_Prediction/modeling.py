"""
This script trains multiple regression models to predict GDP based on various features.
It evaluates their performance, visualizes regression results, and estimates feature importance using permutation importance.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# Regressors
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Set matplotlib backend to avoid compatibility issues with PyCharm
import matplotlib
matplotlib.use('TkAgg')

# Load dataset
file_directory = "prepared_data.parquet"
df = pd.read_parquet(file_directory)

# Drop missing target values
df = df.dropna(subset=["GDP"])

# Define features and target
target_column = "GDP"
X = df.drop(columns=[target_column, "Country"])  # Drop also non-numeric 'Country'
y = df[target_column]
feature_names = X.columns

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "KNN": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

# Evaluation loop
for name, model in models.items():
    print(f"\n===== {name} =====")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))

    # === Plot regression ===
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual GDP")
    plt.ylabel("Predicted GDP")
    plt.title(f"Regression Plot - {name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Permutation Importance ===
    print("Computing permutation importance...")
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()[::-1]

    # Plot importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], align='center', color='steelblue')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel("Mean Importance (permutation)")
    plt.title(f"Permutation Importance - {name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
