"""
Titanic Survival Prediction - Streamlit Web App
This app provides an interactive interface for predicting Titanic passenger survival
"""

# Streamlit Deployment App for Titanic Survival Prediction
# Save this as 'titanic_app.py' and run with: streamlit run titanic_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Configure page
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-result {
    font-size: 2rem;
    font-weight: bold;
    text-align: center;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.survived {
    background-color: #d4edda;
    color: #155724;
    border: 2px solid #c3e6cb;
}
.not-survived {
    background-color: #f8d7da;
    color: #721c24;
    border: 2px solid #f5c6cb;
}
</style>
""", unsafe_allow_html=True)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        with open('C:\\Users\\Abhishek Singh\\CascadeProjects\\my_titanic_project\\models\\titanic_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please run the main project first to train and save the model.")
        return None

# Preprocess input data
def preprocess_input(input_data):
    """Preprocess the input data to match the training format"""
    df = pd.DataFrame([input_data])
    
    # Create derived features
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    df['alone'] = (df['sibsp'] + df['parch'] == 0).astype(int)  # Original dataset column
    
    # Encode categorical variables
    df['sex'] = 1 if df['sex'].iloc[0] == 'male' else 0
    
    # One-hot encode embarked
    embarked = df['embarked'].iloc[0]
    df['embarked_Q'] = 1 if embarked == 'Q' else 0
    df['embarked_S'] = 1 if embarked == 'S' else 0
    
    # Drop original columns that were encoded
    df = df.drop(['embarked'], axis=1)
    
    # Note: Scaling removed as Random Forest doesn't require it
    # and the scaler wasn't saved during training
    
    # Reorder columns to match the model's expected feature order
    expected_order = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'alone', 'embarked_Q', 'embarked_S', 'family_size', 'is_alone']
    df = df[expected_order]
    
    return df

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🚢 Titanic Survival Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Predict the survival chances of Titanic passengers using Machine Learning</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar for input
    st.sidebar.markdown('<h2 class="sub-header">Passenger Information</h2>', unsafe_allow_html=True)
    
    # Input fields
    with st.sidebar.form("passenger_form"):
        # Basic information
        st.subheader("Basic Information")
        pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1st = Upper, 2nd = Middle, 3rd = Lower")
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 80, 30, help="Age in years")
        
        # Family information
        st.subheader("Family Information")
        sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0, help="Number of siblings or spouses aboard")
        parch = st.number_input("Parents/Children Aboard", 0, 10, 0, help="Number of parents or children aboard")
        
        # Travel information
        st.subheader("Travel Information")
        embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"], 
                               help="C = Cherbourg, Q = Queenstown, S = Southampton")
        fare = st.number_input("Fare", 0.0, 600.0, 32.0, help="Passenger fare")
        
        # Submit button
        submitted = st.form_submit_button("Predict Survival")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">About This Project</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-message">
        <strong>Project Overview:</strong><br>
        This machine learning project predicts Titanic passenger survival using historical data. 
        The model analyzes passenger characteristics like age, gender, class, and family size 
        to estimate survival probability.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Key Features Analyzed:")
        features_data = {
            "Feature": ["Passenger Class", "Gender", "Age", "Family Size", "Fare", "Embarkation Port"],
            "Impact": ["High", "Very High", "Medium", "Medium", "High", "Low"],
            "Description": [
                "1st class passengers had higher survival rates",
                "Women had significantly higher survival rates",
                "Children and young adults had better chances",
                "Medium-sized families had optimal survival",
                "Higher fare correlated with better survival",
                "Port showed some correlation with survival"
            ]
        }
        
        features_df = pd.DataFrame(features_data)
        st.dataframe(features_df, use_container_width=True)
        
        st.markdown("### Model Performance:")
        st.markdown("""
        - **Best Model**: Random Forest Classifier
        - **Accuracy**: ~84%
        - **F1-Score**: ~82%
        - **Features Used**: 14 engineered features
        - **Training Data**: 891 passengers
        """)
    
    with col2:
        st.markdown('<h2 class="sub-header">Quick Stats</h2>', unsafe_allow_html=True)
        
        # Display some interesting statistics
        stats_data = {
            "Metric": ["Total Passengers", "Survival Rate", "Female Survival", "Male Survival", "1st Class Survival"],
            "Value": ["891", "38.4%", "74.2%", "18.9%", "62.9%"]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        st.markdown("### Survival Tips:")
        st.markdown("""
        - 🚺 **Women and Children First**: Higher priority for lifeboats
        - 💰 **Higher Class = Better Chance**: 1st class had best survival rates
        - 👨‍👩‍👧‍👦 **Family Size Matters**: Small families had better chances
        - 🎫 **Fare Correlation**: Higher fare often meant better access to lifeboats
        """)
    
    # Prediction section
    if submitted:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
        
        # Prepare input data
        input_data = {
            'pclass': pclass,
            'sex': sex,
            'age': age,
            'sibsp': sibsp,
            'parch': parch,
            'embarked': embarked,
            'fare': fare
        }
        
        # Preprocess the input
        processed_data = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0]
        
        # Display results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if prediction == 1:
                st.markdown("""
                <div class="success-message">
                <h3>🎉 SURVIVED</h3>
                <p>The passenger would have survived the Titanic disaster!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="danger-message">
                <h3>💔 NOT SURVIVED</h3>
                <p>The passenger would not have survived the Titanic disaster.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Survival Probability")
            survival_prob = probability[1] * 100
            
            # Create a gauge chart
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Create a simple bar chart for probability
            colors = ['#ff4444', '#44ff44']
            values = [100 - survival_prob, survival_prob]
            labels = ['Not Survived', 'Survived']
            
            bars = ax.bar(labels, values, color=colors, alpha=0.7)
            ax.set_ylabel('Probability (%)')
            ax.set_title('Survival Probability')
            ax.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{value:.1f}%', ha='center', va='bottom')
            
            st.pyplot(fig)
        
        # Detailed breakdown
        st.markdown("### Detailed Breakdown")
        
        # Create a DataFrame with passenger info
        passenger_info = pd.DataFrame([input_data])
        passenger_info['family_size'] = passenger_info['sibsp'] + passenger_info['parch'] + 1
        passenger_info['is_alone'] = (passenger_info['family_size'] == 1).astype(int)
        
        # Display passenger information
        st.dataframe(passenger_info.T, use_container_width=True)
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            st.markdown("### Key Factors Influencing Prediction")
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': processed_data.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(5)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax)
            ax.set_title('Top 5 Features Influencing Prediction')
            ax.set_xlabel('Importance')
            st.pyplot(fig)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Built with ❤️ using Streamlit and Scikit-learn</p>
        <p>Data Source: Titanic Dataset (Seaborn)</p>
        <p>Model: Random Forest Classifier with Hyperparameter Tuning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
