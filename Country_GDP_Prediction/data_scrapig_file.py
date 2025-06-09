"""
This script downloads historical and projected GDP data (1980–2025) from Wikipedia,
cleans and restructures it into a long format (Country, Year, GDP), and saves it to
CSV, Parquet, and Pickle formats.

Steps performed:
1. Extract GDP tables from the Wikipedia page.
2. Select and normalize tables containing relevant year and country data.
3. Flatten MultiIndex columns if present and rename them consistently.
4. Reshape the combined data into a long-format DataFrame.
5. Convert data types and remove invalid entries.
6. Save the cleaned dataset to disk in multiple formats for future use.
"""

import pandas as pd

# URL of the Wikipedia page with past and projected nominal GDP data
url = "https://en.wikipedia.org/wiki/List_of_countries_by_past_and_projected_GDP_(nominal)"

# Load all tables from the page into a list of DataFrames
tables = pd.read_html(url)

dfs = []  # list to store the selected tables for further processing

for tbl in tables:
    # Check if the table has MultiIndex columns (multi-level headers)
    cols = tbl.columns
    if isinstance(cols, pd.MultiIndex):
        # Flatten MultiIndex by joining header levels into a single string
        cols = [' '.join(map(str, c)).strip() for c in cols.values]
        tbl.columns = cols  # overwrite columns with the flattened names
    else:
        # If columns are single-level, convert names to strings
        tbl.columns = [str(c) for c in cols]

    cols = tbl.columns  # refresh column names after any potential flattening

    # Check if the table contains a 'Country' column and year columns (as digits)
    if any("Country" in c for c in cols) and any(c.strip().isdigit() for c in cols):
        # Find the column that contains 'Country'
        country_col = next(c for c in cols if "Country" in c)
        # Select only columns with years between 1980 and 2025
        wanted_years = [c for c in cols if c.strip().isdigit() and 1980 <= int(c) <= 2025]
        if wanted_years:
            # Create a copy of the relevant columns (Country + years)
            df = tbl[[country_col] + wanted_years].copy()
            # Rename the country column to "Country" for consistency
            df.rename(columns={country_col: "Country"}, inplace=True)
            # Append the filtered table to the list
            dfs.append(df)

# Raise an error if no matching tables were found
if not dfs:
    raise ValueError("No GDP tables found with data for 1980–2025.")

# Concatenate all selected tables into one DataFrame and remove duplicates
gdp_all = pd.concat(dfs, ignore_index=True).drop_duplicates()

# Reshape the data from wide format (years as columns) to long format (one row per year)
gdp_long = pd.melt(
    gdp_all,
    id_vars=["Country"],
    var_name="Year",        # new column for years
    value_name="GDP"        # new column for GDP values
)

# Convert column types
gdp_long["Year"] = pd.to_numeric(gdp_long["Year"], errors='coerce')
gdp_long["GDP"] = pd.to_numeric(gdp_long["GDP"], errors='coerce')
gdp_long["Country"] = gdp_long["Country"].astype(str)

# Save the transformed data to a CSV file without the index
gdp_long.to_csv("gdp_1980-2025-wiki.csv", index=False)


# Drop rows with missing GDP values
gdp_long_clean = gdp_long.dropna(subset=["GDP"])

# Check that all rows are within the valid year range and have non-null GDP values
if gdp_long_clean["Year"].between(1980, 2025).all() and gdp_long_clean["GDP"].notnull().all():
    gdp_long_clean.to_parquet("gdp_1980_2025_wiki.parquet", index=False)
    print("Data saved to Parquet file: gdp_1980_2025_wiki.parquet")

    gdp_long_clean.to_pickle("gdp_1980_2025_wiki.pkl")
    print("Data saved to Pickle file: gdp_1980_2025_wiki.pkl")
else:
    print("Data did not meet quality criteria – Parquet and Pickle files not saved.")
