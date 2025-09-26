# 🚢 Titanic Survival Prediction - Machine Learning Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](#)

A comprehensive machine learning project that predicts Titanic passenger survival using historical data. This project demonstrates the complete data science pipeline from data collection and cleaning to model training, evaluation, and deployment.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [Utility Scripts](#utility-scripts)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project implements a machine learning solution to predict whether a Titanic passenger would have survived the disaster based on passenger characteristics such as age, gender, class, fare, and family information.

### Key Objectives
- **Predict Survival**: Build a model to predict passenger survival with high accuracy
- **Feature Analysis**: Understand which factors most influenced survival chances
- **Deployment**: Create user-friendly web applications for real-time predictions
- **Documentation**: Provide comprehensive project documentation and analysis

### 📊 Dataset Information
- **Source**: Titanic Dataset (Seaborn built-in dataset)
- **Size**: 891 passengers, 15 original features
- **Target Variable**: `survived` (0 = No, 1 = Yes)
- **Survival Rate**: 38.4% (342 survivors out of 891 passengers)

## ✨ Features

### 🤖 Machine Learning
- **Multiple Algorithms**: Logistic Regression, Decision Tree, Random Forest, SVM
- **Best Model**: Random Forest with 84.36% accuracy
- **Feature Engineering**: Age groups, fare categories, family size features
- **Hyperparameter Tuning**: Optimized using RandomizedSearchCV

### 📊 Data Analysis
- **Exploratory Data Analysis**: Comprehensive statistical analysis and visualizations
- **Feature Importance**: Analysis of key survival predictors
- **Data Preprocessing**: Missing value handling, outlier detection, categorical encoding
- **Correlation Analysis**: Understanding feature relationships

### 🚀 Deployment
- **Streamlit Web App**: Interactive user interface for predictions
- **Flask REST API**: Backend API for programmatic access
- **Real-time Predictions**: Instant survival probability calculations
- **Responsive Design**: Works on desktop and mobile devices

### 🔧 Utility Scripts
- **Data Validation**: Check dataset columns and structure
- **Model Analysis**: Validate model features and performance
- **Quick Generation**: Rapid model creation for testing
- **Column Analysis**: Deep dive into feature relationships

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for version control)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/AbhishekSingh2002/my_titanic_project.git
   cd my_titanic_project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv ml_env
   ```

3. **Activate virtual environment**
   - **Windows**:
     ```bash
     ml_env\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source ml_env/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Quick Start

1. **Train the model** (full pipeline with optimization)
   ```bash
   python main_project.py
   ```

2. **Generate quick model** (rapid creation without optimization)
   ```bash
   python quick_generate.py
   ```

3. **Launch Streamlit web application**
   ```bash
   streamlit run deployment/streamlit_app.py
   ```
   Access at: http://localhost:8501

4. **Launch Flask REST API**
   ```bash
   python deployment/flask_app.py
   ```
   Access at: http://localhost:5000

### Utility Scripts

- **Quick generation**: `python quick_generate.py` (rapid model creation)
- **Column analysis**: `python understand_alone_column.py` (analyze `alone` column)

### Web Application Features

#### Streamlit App
- Interactive form for passenger data input
- Real-time survival prediction
- Probability visualization
- Feature importance charts
- Educational content about the Titanic disaster

#### Flask API
- RESTful endpoints for predictions
- JSON input/output format
- Programmatic access to model
- Integration-ready for other applications

## 📁 Project Structure

```
my_titanic_project/
├── README.md                     # This file
├── requirements.txt              # Project dependencies
├── project_report.md            # Comprehensive project report
├── main_project.py              # Complete ML pipeline
├── quick_generate.py            # Quick model generation script
├── understand_alone_column.py   # Column analysis utility
├── data/                        # Dataset storage
│   ├── train.csv
│   └── test.csv
├── models/                      # Trained model files
│   └── titanic_model.pkl        # Optimized Random Forest model
├── reports/                     # Generated reports and visualizations
│   ├── eda_visualizations.png   # EDA plots
│   ├── confusion_matrix.png     # Model confusion matrix
│   └── feature_importance.png   # Feature importance plot
├── deployment/                  # Web deployment applications
│   ├── streamlit_app.py         # Streamlit web application
│   ├── flask_app.py             # Flask REST API
│   └── templates/
│       └── index.html           # Flask web interface
└── .git/                        # Git version control
```

## 📈 Model Performance

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 81.01% | 78.57% | 71.43% | 74.83% |
| Decision Tree | 79.33% | 73.33% | 71.43% | 72.37% |
| **Random Forest** | **84.36%** | **84.21%** | **73.81%** | **79.10%** |
| SVM | 82.12% | 80.00% | 72.62% | 76.12% |

### Key Findings

- **Best Model**: Random Forest with optimized hyperparameters
- **Key Predictors**: 
  1. Sex (28.5% importance)
  2. Fare (18.2% importance)
  3. Age (14.7% importance)
  4. Family Size (12.3% importance)
  5. Passenger Class (9.8% importance)

### Feature Importance
- Gender was the strongest predictor (women had much higher survival rates)
- Economic factors (fare, class) were significant
- Age and family structure played important roles
- Embarkation port had minimal impact

## 🌐 Deployment

### Streamlit Web Application

**Features**:
- User-friendly input form
- Real-time predictions
- Survival probability gauge
- Feature importance visualization
- Responsive design
- Educational content

**Access**:
```bash
streamlit run deployment/streamlit_app.py
# Open http://localhost:8501
```

### Flask REST API

**Endpoints**:
- `GET /`: Web interface
- `POST /predict`: Make predictions
- `GET /health`: Health check

**Request Format**:
```json
{
  "pclass": 1,
  "sex": "female",
  "age": 25,
  "sibsp": 0,
  "parch": 0,
  "fare": 50.0,
  "embarked": "S"
}
```

**Response Format**:
```json
{
  "prediction": 1,
  "probability": 0.85,
  "message": "Passenger would likely survive"
}
```

**Access**:
```bash
python deployment/flask_app.py
# Open http://localhost:5000
```

## 🔧 Utility Scripts

### quick_generate.py
Rapid generation of essential model files without lengthy optimization.
```bash
python quick_generate.py
```

### understand_alone_column.py
Deep analysis of the `alone` column and its relationships.
```bash
python understand_alone_column.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Workflow
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Guidelines
- Follow the existing code style
- Add appropriate comments and documentation
- Ensure all tests pass
- Update the README as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset Source**: Seaborn built-in Titanic dataset
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, flask
- **Inspiration**: The tragic story of the Titanic and the importance of data-driven analysis

## 📞 Contact

- **Project Link**: [https://github.com/AbhishekSingh2002/my_titanic_project](https://github.com/AbhishekSingh2002/my_titanic_project)
- **Author**: Abhishek Singh
- **Email**: [your-email@example.com](mailto:your-email@example.com)

---

⭐ **Star this repository** if you find it helpful!

🚢 **Built with passion for data science and machine learning**
