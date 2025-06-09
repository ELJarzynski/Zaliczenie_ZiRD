"""
In this file, we focused on preparing the data for training and performing feature engineering.

We removed rows containing NaN values, and we created 6 new columns that represent the difference
in GDP over various time periods (1, 5, 10, 15, 20, 25 years) per country. These differences help
capture the historical growth or decline trends in GDP for use in predictive modeling.

To make the dataset suitable for machine learning algorithms, we then applied preprocessing:
- We used `OrdinalEncoder` to convert the 'Country' column from categorical to numerical format,
  allowing algorithms to work with this feature.
- We applied `MinMaxScaler` to scale the GDP difference columns to a range between 0 and 1,
  which is beneficial for many models that are sensitive to feature scale.

Finally, we joined the encoded and scaled columns back with the original dataset (excluding the
original categorical and unscaled columns), removed any remaining rows with missing values, and
saved the processed dataset to a new Parquet file for later use in modeling.
"""


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

# Configure terminal display settings for better visibility of DataFrame output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Load the cleaned GDP dataset
df = pd.read_parquet("gdp_1980_2025_wiki.parquet")

# Remove rows with missing GDP values
df_clean = df.dropna(subset=['GDP'])

# Filter out rows with non-positive GDP values and keep data between 1980 and 2025
df_clean = df_clean[df_clean['GDP'] > 0]
df_clean = df_clean[(df_clean['Year'] >= 1980) & (df_clean['Year'] <= 2025)]

# Sort the data by country and year to prepare for calculating differences
df_clean = df_clean.sort_values(by=['Country', 'Year'])

# Add a column for the difference in GDP compared to the previous year
df_clean['GDP_diff_1'] = df_clean.groupby('Country')['GDP'].diff()

# Define a function to calculate GDP difference over 'n' years
def diff_n_years(df, n):
    return df.groupby('Country')['GDP'].diff(periods=n)

# Add columns for GDP differences over 5, 10, 15, 20, and 25 years
for n in [5, 10, 15, 20, 25]:
    col_name = f'GDP_diff_{n}'
    df_clean[col_name] = diff_n_years(df_clean, n)

"""Defining different groups of columns that will be processed using ColumnTransformer"""
# Defining columns for different encoders
ordinal_cols = ['Country']
scaling_cols = ["GDP_diff_1",  "GDP_diff_5",  "GDP_diff_10",  "GDP_diff_15",  "GDP_diff_20",  "GDP_diff_25"]

"""Preprocessing and scaling using ColumnTransformer"""
# Initializing the encoders
ordinal_preprocessor = OrdinalEncoder()

# Initializing the scaler pipeline
scaler_pipeline = make_pipeline(
    MinMaxScaler()
)


# Configuring the ColumnTransformer
preprocessing_pipeline = ColumnTransformer(
    transformers=[
        ('ordinal', ordinal_preprocessor, ordinal_cols),
        ('scaling', scaler_pipeline, scaling_cols)
    ]
)

"""Making new DataFrame"""
# Transforming the data and creating a new DataFrame with the processed features
data_preprocessed = pd.DataFrame(
    preprocessing_pipeline.fit_transform(df_clean),
    columns=ordinal_cols + scaling_cols,
    index=df_clean.index
)
df_processed = df_clean.drop(columns=scaling_cols + ordinal_cols).join(data_preprocessed)

# Drop any remaining rows with NaN values (i.e., where historical data is missing)
df = df_processed.dropna()

# Save the cleaned DataFrame with GDP differences to a Parquet file
df.to_parquet("prepared_data.parquet", index=False)
print(df.head())