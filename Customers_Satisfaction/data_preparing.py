"""
This script preprocesses airline customer satisfaction data for classification modeling.
It performs the following steps:
1. Loads the dataset from a Parquet file.
2. Removes rows with missing values in the 'Arrival Delay in Minutes' column.
3. Applies appropriate transformations:
   - Ordinal encoding for the target variable 'satisfaction'.
   - One-hot encoding for categorical features.
   - MinMax scaling for selected numerical columns.
4. Combines transformed columns with the rest of the dataset.
5. Saves the final preprocessed dataset to a new Parquet file.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer


# Configure terminal output settings for better readability
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Load the Parquet file
file_directory = r"Airline_customer_satisfaction.parquet"
df = pd.read_parquet(file_directory)
print(df.info())

# Drop rows with missing values in 'Arrival Delay in Minutes' (many examples exist, so the impact is negligible)
df = df.dropna(subset=['Arrival Delay in Minutes'])

# Define column groups for different encoders
ordinal_cols = ['satisfaction']
onehot_cols = ['Customer Type', 'Class', 'Type of Travel']

# Define columns to be scaled
scaling_cols = ['Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Departure/Arrival time convenient']

# Initialize encoders and scaler
ordinal_preprocessor = OrdinalEncoder()
onehot_preprocessor = OneHotEncoder()
scaler_pipeline = make_pipeline(MinMaxScaler())

# Build the preprocessing pipeline
preprocessing_pipeline = ColumnTransformer(
    transformers=[
        ('ordinal', ordinal_preprocessor, ordinal_cols),
        ('onehot', onehot_preprocessor, onehot_cols),
        ('scaling', scaler_pipeline, scaling_cols)
    ]
)

# Apply transformations and create a new DataFrame with the transformed data
data_preprocessed = pd.DataFrame(
    preprocessing_pipeline.fit_transform(df),
    columns=ordinal_cols + list(preprocessing_pipeline.named_transformers_['onehot'].get_feature_names_out(onehot_cols)) + scaling_cols,
    index=df.index
)

# Replace original columns with transformed ones
df = df.drop(columns=scaling_cols + ordinal_cols + onehot_cols).join(data_preprocessed)

# Move 'satisfaction' column to the front
df.insert(0, 'satisfaction', df.pop('satisfaction'))

print(df.head())

# Save the processed dataset to a new Parquet file
df.to_parquet("prepared_data.parquet", index=False)
