# Flask API Deployment for Titanic Survival Prediction
# Save as 'flask_app.py' and run with: python flask_app.py

from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import os

app = Flask(__name__)

class TitanicAPI:
    """Flask API for Titanic survival prediction"""
    
    def __init__(self):
        self.model = None
        self.feature_names = [
            'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 
            'embarked', 'family_size', 'is_alone'
        ]
        self.load_or_train_model()
    
    def load_or_train_model(self):
        """Load pre-trained model or train a new one"""
        try:
            with open('titanic_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            self.train_model()
            
    def train_model(self):
        """Train a new model"""
        print("Training new model...")
        # Load titanic dataset
        df = sns.load_dataset('titanic')
        
        # Preprocessing
        df['age'].fillna(df['age'].median(), inplace=True)
        df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
        df['fare'].fillna(df['fare'].median(), inplace=True)
        
        # Encode categorical variables
        df['sex'] = df['sex'].map({'male': 1, 'female': 0})
        df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        
        # Create features
        df['family_size'] = df['sibsp'] + df['parch'] + 1
        df['is_alone'] = (df['family_size'] == 1).astype(int)
        
        # Select features and target
        X = df[self.feature_names]
        y = df['survived']
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Save model
        with open('titanic_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print("Model trained and saved!")
    
    def predict(self, passenger_data):
        """Make prediction"""
        df = pd.DataFrame([passenger_data])
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0]
        
        return {
            'prediction': int(prediction),
            'survival_probability': float(probability[1]),
            'death_probability': float(probability[0]),
            'result': 'Survived' if prediction == 1 else 'Did not survive'
        }

# Initialize the API
api = TitanicAPI()

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Titanic Survival Predictor</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        input, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        .form-row {
            display: flex;
            gap: 20px;
        }
        .form-row .form-group {
            flex: 1;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            cursor: pointer;
            width: 100%;
            transition: transform 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
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
        .probability {
            margin-top: 10px;
            font-size: 16px;
            font-weight: normal;
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 2s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚢 Titanic Survival Predictor</h1>
        
        <form id="predictionForm">
            <div class="form-row">
                <div class="form-group">
                    <label for="pclass">Passenger Class:</label>
                    <select id="pclass" name="pclass" required>
                        <option value="1">First Class</option>
                        <option value="2">Second Class</option>
                        <option value="3">Third Class</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="sex">Gender:</label>
                    <select id="sex" name="sex" required>
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="age">Age:</label>
                    <input type="number" id="age" name="age" min="0" max="100" value="30" required>
                </div>
                
                <div class="form-group">
                    <label for="fare">Fare ($):</label>
                    <input type="number" id="fare" name="fare" min="0" step="0.01" value="50" required>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="sibsp">Siblings/Spouses:</label>
                    <input type="number" id="sibsp" name="sibsp" min="0" max="10" value="0" required>
                </div>
                
                <div class="form-group">
                    <label for="parch">Parents/Children:</label>
                    <input type="number" id="parch" name="parch" min="0" max="10" value="0" required>
                </div>
            </div>
            
            <div class="form-group">
                <label for="embarked">Port of Embarkation:</label>
                <select id="embarked" name="embarked" required>
                    <option value="0">Southampton</option>
                    <option value="1">Cherbourg</option>
                    <option value="2">Queenstown</option>
                </select>
            </div>
            
            <button type="submit" class="btn">Predict Survival</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Making prediction...</p>
        </div>
        
        <div id="result"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').innerHTML = '';
            
            // Collect form data
            const formData = new FormData(this);
            const data = {
                pclass: parseInt(formData.get('pclass')),
                sex: parseInt(formData.get('sex')),
                age: parseFloat(formData.get('age')),
                sibsp: parseInt(formData.get('sibsp')),
                parch: parseInt(formData.get('parch')),
                fare: parseFloat(formData.get('fare')),
                embarked: parseInt(formData.get('embarked'))
            };
            
            // Calculate derived features
            data.family_size = data.sibsp + data.parch + 1;
            data.is_alone = (data.family_size === 1) ? 1 : 0;
            
            // Make API call
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                
                // Show result
                const resultDiv = document.getElementById('result');
                const survived = data.prediction === 1;
                const probability = (data.survival_probability * 100).toFixed(1);
                
                resultDiv.innerHTML = `
                    <div class="result ${survived ? 'survived' : 'not-survived'}">
                        ${survived ? '🎉 SURVIVED' : '💔 DID NOT SURVIVE'}
                        <div class="probability">
                            Survival Probability: ${probability}%
                        </div>
                    </div>
                `;
            })
            .catch(error => {
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                
                // Show error
                document.getElementById('result').innerHTML = `
                    <div class="result not-survived">
                        ❌ Error making prediction: ${error.message}
                    </div>
                `;
            });
        });
    </script>
</body>
</html>
"""

# Routes
@app.route('/')
def home():
    """Serve the main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Calculate derived features if not provided
        if 'family_size' not in data:
            data['family_size'] = data['sibsp'] + data['parch'] + 1
        if 'is_alone' not in data:
            data['is_alone'] = 1 if data['family_size'] == 1 else 0
        
        # Make prediction
        result = api.predict(data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': api.model is not None})

@app.route('/api/info')
def api_info():
    """API information endpoint"""
    return jsonify({
        'name': 'Titanic Survival Predictor API',
        'version': '1.0',
        'description': 'Predicts survival probability for Titanic passengers',
        'endpoints': {
            '/': 'Web interface',
            '/predict': 'POST - Make predictions',
            '/health': 'GET - Health check',
            '/api/info': 'GET - API information'
        },
        'required_fields': api.feature_names,
        'example_request': {
            'pclass': 1,
            'sex': 0,
            'age': 25,
            'sibsp': 0,
            'parch': 0,
            'fare': 80.0,
            'embarked': 0,
            'family_size': 1,
            'is_alone': 1
        }
    })

if __name__ == '__main__':
    print("Starting Titanic Survival Predictor API...")
    print("Web interface: http://localhost:5000")
    print("API endpoint: http://localhost:5000/predict")
    print("Health check: http://localhost:5000/health")
    app.run(debug=True, host='0.0.0.0', port=5000)
