"""
Complete Titanic Machine Learning Project
This script covers the entire ML pipeline from data loading to model optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TitanicMLPipeline:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_params = None
        
    def load_data(self):
        """Load Titanic dataset from seaborn"""
        print("🔄 Loading Titanic dataset...")
        self.data = sns.load_dataset('titanic')
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Dataset shape: {self.data.shape}")
        print(f"📋 Columns: {list(self.data.columns)}")
        return self.data
    
    def explore_data(self):
        """Perform initial data exploration"""
        print("\n🔍 Data Exploration:")
        print("\n📋 Dataset Info:")
        print(self.data.info())
        
        print("\n📊 Descriptive Statistics:")
        print(self.data.describe())
        
        print("\n❓ Missing Values:")
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0])
        
        print(f"\n💾 Survival Rate: {self.data['survived'].mean():.3f} ({self.data['survived'].mean()*100:.1f}%)")
        
        return missing_values
    
    def clean_data(self):
        """Clean and preprocess the data"""
        print("\n🧹 Data Cleaning:")
        
        # Create a copy to avoid modifying original data
        df_clean = self.data.copy()
        
        # Handle missing values
        print("Handling missing values...")
        
        # Age - fill with median
        df_clean['age'].fillna(df_clean['age'].median(), inplace=True)
        
        # Embarked - fill with mode
        df_clean['embarked'].fillna(df_clean['embarked'].mode()[0], inplace=True)
        
        # Deck - too many missing values, drop the column
        df_clean.drop('deck', axis=1, inplace=True)
        
        # Drop unnecessary columns
        columns_to_drop = ['alive', 'who', 'adult_male', 'embark_town', 'class']
        df_clean.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')
        
        # Encode categorical variables
        print("Encoding categorical variables...")
        
        # Sex - Label Encoding
        le_sex = LabelEncoder()
        df_clean['sex'] = le_sex.fit_transform(df_clean['sex'])
        
        # Embarked - One Hot Encoding
        df_clean = pd.get_dummies(df_clean, columns=['embarked'], prefix='embarked', drop_first=True)
        
        # Create new features
        print("Creating new features...")
        
        # Family size
        df_clean['family_size'] = df_clean['sibsp'] + df_clean['parch'] + 1
        
        # Is alone
        df_clean['is_alone'] = (df_clean['family_size'] == 1).astype(int)
        
        # Age groups
        df_clean['age_group'] = pd.cut(df_clean['age'], bins=[0, 12, 18, 35, 60, 100], 
                                      labels=['child', 'teenager', 'young_adult', 'adult', 'senior'])
        
        # Fare groups
        df_clean['fare_group'] = pd.qcut(df_clean['fare'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        # Encode age and fare groups
        df_clean['age_group'] = LabelEncoder().fit_transform(df_clean['age_group'])
        df_clean['fare_group'] = LabelEncoder().fit_transform(df_clean['fare_group'])
        
        self.data = df_clean
        
        print("✅ Data cleaning completed!")
        print(f"📊 Cleaned dataset shape: {self.data.shape}")
        print(f"❓ Missing values after cleaning: {self.data.isnull().sum().sum()}")
        
        return self.data
    
    def detect_outliers(self):
        """Detect outliers using IQR method"""
        print("\n🔍 Outlier Detection:")
        
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        
        outlier_info = {}
        for column in numeric_columns:
            if column in ['survived', 'sex', 'embarked_Q', 'embarked_S']:
                continue
                
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
            outlier_count = len(outliers)
            
            outlier_info[column] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(self.data)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            print(f"{column}: {outlier_count} outliers ({outlier_count/len(self.data)*100:.1f}%)")
        
        return outlier_info
    
    def create_visualizations(self):
        """Create comprehensive visualizations for EDA"""
        print("\n📊 Creating Visualizations...")
        
        # Create a figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Titanic Dataset Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Age Distribution
        sns.histplot(data=self.data, x='age', hue='survived', multiple='stack', bins=30, ax=axes[0, 0])
        axes[0, 0].set_title('Age Distribution by Survival')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Count')
        
        # 2. Age vs Fare Scatter Plot
        scatter = axes[0, 1].scatter(self.data['age'], self.data['fare'], c=self.data['survived'], 
                                    alpha=0.6, cmap='coolwarm')
        axes[0, 1].set_title('Age vs Fare by Survival')
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Fare')
        plt.colorbar(scatter, ax=axes[0, 1], label='Survived')
        
        # 3. Correlation Heatmap
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 2])
        axes[0, 2].set_title('Correlation Heatmap')
        
        # 4. Survival by Gender
        gender_survival = self.data.groupby('sex')['survived'].mean().reset_index()
        gender_labels = ['Female', 'Male']
        sns.barplot(x=gender_labels, y=gender_survival['survived'], ax=axes[1, 0])
        axes[1, 0].set_title('Survival Rate by Gender')
        axes[1, 0].set_ylabel('Survival Rate')
        axes[1, 0].set_ylim(0, 1)
        
        # 5. Survival by Passenger Class
        class_survival = self.data.groupby('pclass')['survived'].mean().reset_index()
        sns.barplot(x='pclass', y='survived', data=class_survival, ax=axes[1, 1])
        axes[1, 1].set_title('Survival Rate by Passenger Class')
        axes[1, 1].set_ylabel('Survival Rate')
        axes[1, 1].set_ylim(0, 1)
        
        # 6. Family Size Distribution
        sns.countplot(data=self.data, x='family_size', hue='survived', ax=axes[1, 2])
        axes[1, 2].set_title('Family Size Distribution by Survival')
        axes[1, 2].set_xlabel('Family Size')
        axes[1, 2].set_ylabel('Count')
        
        # 7. Age Box Plot by Survival
        sns.boxplot(data=self.data, x='survived', y='age', ax=axes[2, 0])
        axes[2, 0].set_title('Age Distribution by Survival')
        axes[2, 0].set_xlabel('Survived')
        axes[2, 0].set_ylabel('Age')
        
        # 8. Fare Distribution by Survival
        sns.histplot(data=self.data, x='fare', hue='survived', multiple='stack', bins=30, ax=axes[2, 1])
        axes[2, 1].set_title('Fare Distribution by Survival')
        axes[2, 1].set_xlabel('Fare')
        axes[2, 1].set_ylabel('Count')
        
        # 9. Survival by Embarkation Port
        if 'embarked_S' in self.data.columns:
            embarked_data = self.data.copy()
            embarked_data['embarked'] = 'C'  # Default to Cherbourg
            embarked_data.loc[embarked_data['embarked_S'] == 1, 'embarked'] = 'S'
            embarked_data.loc[embarked_data['embarked_Q'] == 1, 'embarked'] = 'Q'
            
            embarked_survival = embarked_data.groupby('embarked')['survived'].mean().reset_index()
            sns.barplot(x='embarked', y='survived', data=embarked_survival, ax=axes[2, 2])
            axes[2, 2].set_title('Survival Rate by Embarkation Port')
            axes[2, 2].set_ylabel('Survival Rate')
            axes[2, 2].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('C:\\Users\\Abhishek Singh\\CascadeProjects\\my_titanic_project\\reports\\eda_visualizations.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved to reports/eda_visualizations.png")
        
        # Generate key insights
        self.generate_insights()
    
    def generate_insights(self):
        """Generate key insights from the data"""
        print("\n📈 Key Insights:")
        
        # Gender survival rates
        female_survival = self.data[self.data['sex'] == 0]['survived'].mean()
        male_survival = self.data[self.data['sex'] == 1]['survived'].mean()
        print(f"• Female survival rate: {female_survival:.3f} ({female_survival*100:.1f}%)")
        print(f"• Male survival rate: {male_survival:.3f} ({male_survival*100:.1f}%)")
        
        # Class survival rates
        for pclass in sorted(self.data['pclass'].unique()):
            class_survival = self.data[self.data['pclass'] == pclass]['survived'].mean()
            print(f"• {pclass}st class survival rate: {class_survival:.3f} ({class_survival*100:.1f}%)")
        
        # Age insights
        avg_age_survived = self.data[self.data['survived'] == 1]['age'].mean()
        avg_age_not_survived = self.data[self.data['survived'] == 0]['age'].mean()
        print(f"• Average age of survivors: {avg_age_survived:.1f} years")
        print(f"• Average age of non-survivors: {avg_age_not_survived:.1f} years")
        
        # Family size insights
        family_survival = self.data.groupby('family_size')['survived'].mean()
        best_family_size = family_survival.idxmax()
        best_survival_rate = family_survival.max()
        print(f"• Optimal family size for survival: {best_family_size} (survival rate: {best_survival_rate:.3f})")
    
    def prepare_features(self):
        """Prepare features for modeling"""
        print("\n🔧 Preparing features for modeling...")
        
        # Separate features and target
        X = self.data.drop('survived', axis=1)
        y = self.data['survived']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        numerical_features = ['age', 'fare', 'sibsp', 'parch', 'family_size']
        self.X_train[numerical_features] = self.scaler.fit_transform(self.X_train[numerical_features])
        self.X_test[numerical_features] = self.scaler.transform(self.X_test[numerical_features])
        
        print(f"✅ Features prepared!")
        print(f"📊 Training set shape: {self.X_train.shape}")
        print(f"📊 Test set shape: {self.X_test.shape}")
        print(f"📊 Training set survival rate: {self.y_train.mean():.3f}")
        print(f"📊 Test set survival rate: {self.y_test.mean():.3f}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple models"""
        print("\n🤖 Training Models...")
        
        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"🔄 Training {name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"✅ {name} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        self.models = results
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"📊 Best F1-Score: {results[best_model_name]['f1_score']:.4f}")
        
        return results
    
    def evaluate_models(self):
        """Evaluate all models with detailed metrics"""
        print("\n📊 Model Evaluation:")
        
        # Create results table
        results_table = []
        for name, result in self.models.items():
            results_table.append({
                'Model': name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'F1-Score': f"{result['f1_score']:.4f}",
                'CV Mean': f"{result['cv_mean']:.4f}",
                'CV Std': f"{result['cv_std']:.4f}"
            })
        
        results_df = pd.DataFrame(results_table)
        print(results_df.to_string(index=False))
        
        # Plot confusion matrix for best model
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['f1_score'])
        best_model = self.models[best_model_name]['model']
        
        y_pred = best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Survived', 'Survived'],
                   yticklabels=['Not Survived', 'Survived'])
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('C:\\Users\\Abhishek Singh\\CascadeProjects\\my_titanic_project\\reports\\confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance for tree-based models
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
            plt.title(f'Top 10 Feature Importance - {best_model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('C:\\Users\\Abhishek Singh\\CascadeProjects\\my_titanic_project\\reports\\feature_importance.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\n🔝 Top 5 Feature Importance ({best_model_name}):")
            for i, row in feature_importance.head().iterrows():
                print(f"• {row['feature']}: {row['importance']:.4f}")
        
        # Detailed classification report
        print(f"\n📋 Detailed Classification Report ({best_model_name}):")
        print(classification_report(self.y_test, y_pred, target_names=['Not Survived', 'Survived']))
        
        return results_df
    
    def optimize_model(self):
        """Optimize the best model using hyperparameter tuning"""
        print("\n⚡ Hyperparameter Optimization...")
        
        # Find best model
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['f1_score'])
        base_model = self.models[best_model_name]['model']
        base_f1 = self.models[best_model_name]['f1_score']
        
        print(f"🎯 Optimizing {best_model_name}...")
        print(f"📊 Base F1-Score: {base_f1:.4f}")
        
        # Define parameter grids for different models
        param_grids = {
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['liblinear', 'lbfgs']
            },
            'Decision Tree': {
                'max_depth': [3, 5, 7, 10, 15, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'criterion': ['gini', 'entropy']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear', 'poly']
            }
        }
        
        # Use RandomizedSearchCV for efficiency
        param_grid = param_grids.get(best_model_name, {})
        
        if param_grid:
            # RandomizedSearchCV
            random_search = RandomizedSearchCV(
                base_model, 
                param_distributions=param_grid,
                n_iter=20,  # Number of parameter combinations to try
                cv=5,
                scoring='f1',
                random_state=42,
                n_jobs=-1
            )
            
            random_search.fit(self.X_train, self.y_train)
            
            # Best parameters and model
            self.best_params = random_search.best_params_
            self.best_model = random_search.best_estimator_
            
            # Evaluate optimized model
            y_pred = self.best_model.predict(self.X_test)
            optimized_f1 = f1_score(self.y_test, y_pred)
            optimized_accuracy = accuracy_score(self.y_test, y_pred)
            
            print(f"✅ Optimization completed!")
            print(f"🔧 Best Parameters: {self.best_params}")
            print(f"📊 Optimized F1-Score: {optimized_f1:.4f}")
            print(f"📊 Optimized Accuracy: {optimized_accuracy:.4f}")
            print(f"📈 F1-Score Improvement: {optimized_f1 - base_f1:.4f}")
            
            # Save the optimized model
            with open('C:\\Users\\Abhishek Singh\\CascadeProjects\\my_titanic_project\\models\\titanic_model.pkl', 'wb') as f:
                pickle.dump(self.best_model, f)
            
            print("💾 Optimized model saved to models/titanic_model.pkl")
            
            return {
                'base_f1': base_f1,
                'optimized_f1': optimized_f1,
                'improvement': optimized_f1 - base_f1,
                'best_params': self.best_params
            }
        else:
            print("⚠️ No parameter grid defined for this model. Skipping optimization.")
            return None
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        print("🚀 Starting Complete Titanic ML Pipeline...")
        print("=" * 60)
        
        # Step 1: Load Data
        self.load_data()
        
        # Step 2: Explore Data
        self.explore_data()
        
        # Step 3: Clean Data
        self.clean_data()
        
        # Step 4: Detect Outliers
        self.detect_outliers()
        
        # Step 5: Create Visualizations
        self.create_visualizations()
        
        # Step 6: Prepare Features
        self.prepare_features()
        
        # Step 7: Train Models
        model_results = self.train_models()
        
        # Step 8: Evaluate Models
        evaluation_results = self.evaluate_models()
        
        # Step 9: Optimize Model
        optimization_results = self.optimize_model()
        
        print("\n🎉 Pipeline Completed Successfully!")
        print("=" * 60)
        
        return {
            'model_results': model_results,
            'evaluation_results': evaluation_results,
            'optimization_results': optimization_results
        }

# Main execution
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = TitanicMLPipeline()
    
    # Run the complete pipeline
    results = pipeline.run_complete_pipeline()
    
    print("\n📋 Project Summary:")
    print("✅ Data loading and cleaning completed")
    print("✅ Exploratory data analysis completed")
    print("✅ Model training and evaluation completed")
    print("✅ Hyperparameter optimization completed")
    print("✅ Model and visualizations saved")
    print("\n🎯 Next Steps:")
    print("1. Check the reports/ folder for visualizations")
    print("2. Check the models/ folder for the saved model")
    print("3. Run deployment scripts to create web applications")
    print("4. Create project report using the generated insights")
