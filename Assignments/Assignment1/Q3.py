import pandas as pd

# 1) LOAD THE DATA
df = pd.read_csv("Hw_1_data2.csv")

# -------------------------------------------------------------
# 2) PEARSON CORRELATION WITH THE TARGET VARIABLE (quality)
# -------------------------------------------------------------
feature_cols = [col for col in df.columns if col != "quality"]

corrs = {}
for feature in feature_cols:
    corrs[feature] = df[feature].corr(df["quality"], method='pearson')

# -------------------------------------------------------------
# 3) RANK THE FEATURES BY CORRELATION WITH 'quality'
# -------------------------------------------------------------
# Sort by absolute correlation value in descending order
sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)

print("Features ranked by correlation (absolute value) with 'quality':")
for feat, val in sorted_corrs:
    print(f"{feat:20s} : {val:.3f}")

# -------------------------------------------------------------
# 4) CORRELATION MATRIX FOR ALL FEATURES
# -------------------------------------------------------------
corr_matrix = df.corr(method='pearson')
print("\nFull Correlation Matrix:")
print(corr_matrix)

# -------------------------------------------------------------
# 5) LIST OUT REDUNDANT FEATURE PAIRS (THRESHOLD >= 0.51)
# -------------------------------------------------------------
threshold = 0.51
redundant_pairs = []

cols = corr_matrix.columns
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) >= threshold:
            redundant_pairs.append((cols[i], cols[j], corr_val))

print(f"\nRedundant feature pairs with |correlation| >= {threshold}:")
for f1, f2, val in redundant_pairs:
    print(f"{f1} & {f2} : {val:.3f}")
