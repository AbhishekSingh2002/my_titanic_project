"""
Script to check what features the model expects
"""

import pickle
import pandas as pd

# Load the model
try:
    with open('models/titanic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Check if model has feature_names_in_ attribute
    if hasattr(model, 'feature_names_in_'):
        print("Model expects these features:")
        print(model.feature_names_in_)
    else:
        print("Model doesn't have feature_names_in_ attribute")
        print("Model type:", type(model))
        
    # Try to get feature importances to see feature names
    if hasattr(model, 'feature_importances_'):
        print("\nFeature importances available")
        print("Number of features:", len(model.feature_importances_))
        
except Exception as e:
    print("Error loading model:", e)

# Also check what features were in the training data
print("\n" + "="*50)
print("Checking training data structure...")

try:
    import seaborn as sns
    df = sns.load_dataset('titanic')
    
    # Handle missing values
    df['age'].fillna(df['age'].median(), inplace=True)
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
    df['fare'].fillna(df['fare'].median(), inplace=True)
    
    # Drop columns with too many missing values
    df.drop('deck', axis=1, inplace=True)
    
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
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
    
    print("Training data features:")
    print(X.columns.tolist())
    print("Number of features:", len(X.columns))
    
except Exception as e:
    print("Error checking training data:", e)
