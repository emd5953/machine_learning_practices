import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # styling 

# 1) Load dataset
df = pd.read_csv("Hw_1_data1.csv")

# -----------------------------
# 2) Clean the dataset
# -----------------------------

# a) Check duplicates
# - Used conditionals (if statements) to remove duplicates only if any exist.
num_duplicates = df.duplicated().sum()
if num_duplicates > 0:
    print(f"Found {num_duplicates} duplicate rows; removing them.")
    df.drop_duplicates(inplace=True)
else:
    print("No duplicate rows found.")

# b) Check missing values
# - Used conditionals to check for missing values and impute only if needed.
missing_count = df.isnull().sum().sum() 
if missing_count > 0:
    print(f"\nFound {missing_count} missing values; applying imputation as an example.")
    # Example numeric fill: mean imputation
for col in df.columns:
    # If the column is numeric
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].mean())
    else:
        # If the column is non-numeric (categorical)
        most_frequent = df[col].value_counts().idxmax()
        df[col] = df[col].fillna(most_frequent)
else:
    print("\nNo missing values found.")

# c) Standardize text casing for Feature E
df['Feature E'] = df['Feature E'].astype(str).str.lower()

# d) Convert Feature D, E to categorical if needed
df['Feature D'] = df['Feature D'].astype('category')
df['Feature E'] = df['Feature E'].astype('category')


# 3) Show first 5 rows of the cleaned data
print("\nCleaned Data (first 5 rows):")
print(df.head())

# 4) Boxplot of Feature B
plt.figure(figsize=(6,4))
sns.boxplot(x=df['Feature B'])
plt.title("Boxplot of Feature B")
plt.xlabel("Feature B")
plt.show()

# 5) Pie chart of Feature G
counts_g = df['Feature G'].value_counts()
plt.figure(figsize=(5,5))
counts_g.plot(kind='pie', autopct='%1.1f%%')
plt.title("Pie Chart of Feature G")
plt.ylabel("")  # Hide the default y‐label
plt.show()
