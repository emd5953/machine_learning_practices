import pandas as pd
import matplotlib.pyplot as plt

# Read in the data
df = pd.read_csv("data.csv")

# Strip leading/trailing whitespace from all string columns
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

# 1. Identify and list numeric columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
print("Numeric columns:")
for col in numeric_cols:
    print(f"  {col}")

print("\n" + "="*40 + "\n")

# 2. Print all unique values in non-numeric (object) columns
non_numeric_cols = df.select_dtypes(include=["object"]).columns
for col in non_numeric_cols:
    print(f"Column: {col}")
    unique_values = df[col].unique()
    print("Unique values:", unique_values)
    print("-"*40)

# Example dictionary mapping each non-numeric column to a sub-dict of {old_value: new_value}
# Adjust this to match the actual columns and categories in your data.
replacement_mappings = {
    'Feature A': {'low': 0, 'mid': 1, 'high': 2},
    'Feature G': {'stable': 0, 'mod-stable': 1, 'unstable': 2},
    'Feature H': {'A': 0, 'S': 1, 'I': 2}
}


# Loop through each column mapping and apply to the DataFrame
for column, mapping in replacement_mappings.items():
    if column in df.columns:
        df[column] = df[column].map(mapping)

# (Optional) Check the updated DataFrame
print(df.head())


# Boxplot for Feature G
plt.boxplot(df['Feature G'].dropna(), vert=False)
plt.title("Boxplot of Feature G")
plt.xlabel("Value")
plt.show()

# Boxplot for Feature H
plt.boxplot(df['Feature H'].dropna(), vert=False)
plt.title("Boxplot of Feature H")
plt.xlabel("Value")
plt.show()


import pandas as pd

def find_outliers_iqr(series):
    """Return a boolean mask where True indicates the row is an outlier (using 1.5 * IQR)."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return (series < lower_bound) | (series > upper_bound)

# Example: Replace outliers in 'Feature G'
# (Assume df is your DataFrame and 'Feature G' is already numeric)

# 1. Identify outliers
outliers_mask = find_outliers_iqr(df['Feature G'])

# 2. Compute the median
median_val = df['Feature G'].median()

# 3. Replace outliers with the median
df.loc[outliers_mask, 'Feature G'] = median_val

# Check results
print("Replaced outliers with median in Feature G.")
print("Number of outliers replaced:", outliers_mask.sum())

# Boxplot
plt.boxplot(df['Feature G'].dropna(), vert=False)
plt.title("Boxplot of Feature G (with potential outliers)")
plt.show()


outliers_mask_h = find_outliers_iqr(df['Feature H'])
median_h = df['Feature H'].median()
df.loc[outliers_mask_h, 'Feature H'] = median_h

print("Replaced outliers with median in Feature H.")
print("Number of outliers replaced:", outliers_mask_h.sum())

plt.boxplot(df['Feature H'].dropna(), vert=False)
plt.title("Boxplot of Feature H (with potential outliers)")
plt.show()


# Before dropping duplicates, check how many rows you have
initial_count = len(df)

# Drop duplicate rows based on the 'ID' column, keep only the first occurrence
df.drop_duplicates(subset=['ID'], keep='first', inplace=True)

# Check how many rows remain
final_count = len(df)
print("Number of rows removed:", initial_count - final_count)
