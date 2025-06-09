"""
This script reads the 'Airline_customer_satisfaction.csv' file,
verifies that each column matches the expected data type,
and saves the validated data as a Parquet file.

Raises:
    - ValueError: If any expected column is missing.
    - TypeError: If any column has a data type different from the expected one.
"""

import pandas as pd


df = pd.read_csv("Airline_customer_satisfaction.csv")

# Expected column types
expected_types = {
    "satisfaction": "object",
    "Customer Type": "object",
    "Age": "int64",
    "Type of Travel": "object",
    "Class": "object",
    "Flight Distance": "int64",
    "Seat comfort": "int64",
    "Departure/Arrival time convenient": "int64",
    "Food and drink": "int64",
    "Gate location": "int64",
    "Inflight wifi service": "int64",
    "Inflight entertainment": "int64",
    "Online support": "int64",
    "Ease of Online booking": "int64",
    "On-board service": "int64",
    "Leg room service": "int64",
    "Baggage handling": "int64",
    "Checkin service": "int64",
    "Cleanliness": "int64",
    "Online boarding": "int64",
    "Departure Delay in Minutes": "int64",
    "Arrival Delay in Minutes": "float64",
}

# Check if each column exists and has the correct data type
for col, expected_type in expected_types.items():
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")
    actual_type = df[col].dtype
    if str(actual_type) != expected_type:
        raise TypeError(f"Column '{col}' has type '{actual_type}', expected '{expected_type}'.")

# Save to a .parquet file if all checks pass
df.to_parquet("Airline_customer_satisfaction.parquet", index=False)
print("Column types are valid. Data saved to Airline_customer_satisfaction.parquet.")
