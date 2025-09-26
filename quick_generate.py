"""
Quick script to generate essential missing files
This will create the model and visualizations without lengthy optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os

print("🚀 Quick Generation of Missing Files...")

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Load data
print("🔄 Loading Titanic dataset...")
df = sns.load_dataset('titanic')
print(f"✅ Dataset loaded: {df.shape}")

# Data preprocessing
print("🧹 Preprocessing data...")
# Handle missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df['fare'].fillna(df['fare'].median(), inplace=True)

# Drop columns with too many missing values
df.drop('deck', axis=1, inplace=True)

# Encode categorical variables
df['sex'] = LabelEncoder().fit_transform(df['sex'])
df = pd.get_dummies(df, columns=['embarked'], prefix='embarked', drop_first=True)

# Create features
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)

# Drop unnecessary columns
columns_to_drop = ['alive', 'who', 'adult_male', 'embark_town', 'class']
df.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')

# Prepare features and target
X = df.drop('survived', axis=1)
y = df['survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
numerical_features = ['age', 'fare', 'sibsp', 'parch', 'family_size']
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Train model
print("🤖 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
print("📊 Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"✅ Model Performance:")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"   F1-Score: {f1:.4f}")

# Save model
print("💾 Saving model...")
with open('models/titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model saved to models/titanic_model.pkl")

# Create visualizations
print("📊 Creating visualizations...")

# 1. EDA Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Titanic Dataset EDA', fontsize=16)

# Age distribution by survival
sns.histplot(data=df, x='age', hue='survived', multiple='stack', bins=30, ax=axes[0, 0])
axes[0, 0].set_title('Age Distribution by Survival')

# Survival by gender
gender_survival = df.groupby('sex')['survived'].mean().reset_index()
sns.barplot(x=gender_survival['sex'], y=gender_survival['survived'], ax=axes[0, 1])
axes[0, 1].set_title('Survival Rate by Gender')
axes[0, 1].set_xticklabels(['Female', 'Male'])

# Survival by class
class_survival = df.groupby('pclass')['survived'].mean().reset_index()
sns.barplot(x='pclass', y='survived', data=class_survival, ax=axes[1, 0])
axes[1, 0].set_title('Survival Rate by Passenger Class')

# Correlation heatmap
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('reports/eda_visualizations.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ EDA visualizations saved to reports/eda_visualizations.png")

# 2. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Not Survived', 'Survived'],
           yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Confusion matrix saved to reports/confusion_matrix.png")

# 3. Feature Importance
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Top 10 Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('reports/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Feature importance plot saved to reports/feature_importance.png")

print("\n🎉 All missing files generated successfully!")
print("📁 Check the following directories:")
print("   - models/ (contains titanic_model.pkl)")
print("   - reports/ (contains PNG visualizations)")
print("\n🚀 You can now run the deployment applications!")
