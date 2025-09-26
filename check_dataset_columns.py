"""
Script to check what columns are in the original titanic dataset
"""

import seaborn as sns

# Load the dataset
df = sns.load_dataset('titanic')

print("Original Titanic dataset columns:")
print(df.columns.tolist())
print(f"\nTotal columns: {len(df.columns)}")

print("\n" + "="*50)
print("Checking for 'alone' column:")
if 'alone' in df.columns:
    print("✅ 'alone' column exists")
    print(f"Unique values: {df['alone'].unique()}")
    print(f"Value counts:\n{df['alone'].value_counts()}")
else:
    print("❌ 'alone' column does not exist")

print("\n" + "="*50)
print("Checking for 'adult_male' column:")
if 'adult_male' in df.columns:
    print("✅ 'adult_male' column exists")
    print(f"Unique values: {df['adult_male'].unique()}")
    print(f"Value counts:\n{df['adult_male'].value_counts()}")
else:
    print("❌ 'adult_male' column does not exist")

print("\n" + "="*50)
print("First few rows of relevant columns:")
relevant_cols = ['adult_male', 'alone'] if 'alone' in df.columns else ['adult_male']
if any(col in df.columns for col in relevant_cols):
    print(df[relevant_cols].head())
else:
    print("None of the relevant columns found")
