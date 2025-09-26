"""
Script to understand what the 'alone' column represents
"""

import seaborn as sns
import pandas as pd

# Load the dataset
df = sns.load_dataset('titanic')

print("Understanding the 'alone' column:")
print("="*50)

# Check relationship between alone and other columns
print("alone vs sibsp + parch:")
print(df.groupby('alone')[['sibsp', 'parch']].agg(['mean', 'count']))

print("\n" + "="*50)
print("alone vs is_alone (calculated):")
df['is_alone'] = (df['sibsp'] + df['parch'] == 0).astype(int)
print(pd.crosstab(df['alone'], df['is_alone'], margins=True))

print("\n" + "="*50)
print("alone vs adult_male:")
print(pd.crosstab(df['alone'], df['adult_male'], margins=True))

print("\n" + "="*50)
print("alone vs who:")
print(pd.crosstab(df['alone'], df['who'], margins=True))

print("\n" + "="*50)
print("Sample data:")
print(df[['alone', 'sibsp', 'parch', 'adult_male', 'who', 'is_alone']].head(10))

# Check if alone is equivalent to (sibsp + parch == 0)
print("\n" + "="*50)
print("Is alone equivalent to (sibsp + parch == 0)?")
df['calculated_alone'] = (df['sibsp'] + df['parch'] == 0)
print(f"Match rate: {(df['alone'] == df['calculated_alone']).mean():.2%}")

# Check if alone is equivalent to adult_male
print("\nIs alone equivalent to adult_male?")
print(f"Match rate: {(df['alone'] == df['adult_male']).mean():.2%}")

# Check if alone is equivalent to being man and alone
print("\nIs alone equivalent to being man and (sibsp + parch == 0)?")
df['man_alone'] = (df['who'] == 'man') & (df['sibsp'] + df['parch'] == 0)
print(f"Match rate: {(df['alone'] == df['man_alone']).mean():.2%}")
