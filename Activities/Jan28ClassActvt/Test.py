import pandas as pd
import matplotlib.pyplot as plt

# Read in the data
df = pd.read_csv("data.csv")

# Strip leading/trailing whitespace from all string columns
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

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



