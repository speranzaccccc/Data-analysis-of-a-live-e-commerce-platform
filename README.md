# Data Analysis of a Live E-commerce Platform

## Project Overview
This project focuses on predicting customer churn and optimizing operational strategies through AB testing in an e-commerce platform. The analysis leverages multi-dimensional behavioral data to identify high-risk customers and implement effective retention strategies.

### Key Objectives
- Predict customer churn using behavioral data (Tenure, login devices, order characteristics)
- Identify key churn drivers (e.g., correlation between CouponUsed and Churn)
- Design and analyze AB tests to validate retention strategies

### Business Value
- Build a customer churn early warning system to reduce churn rate by 5%+
- Develop actionable operational strategies (coupon distribution, UI optimization)

## Project Structure
```
├── data/
│   └── Project Dataset.csv
├── notebooks/
│   ├── 1_exploratory_data_analysis.ipynb
│   ├── 2_data_preprocessing.ipynb
│   ├── 3_model_development.ipynb
│   └── 4_ab_testing_analysis.ipynb
├── src/
│   ├── data_processing/
│   ├── feature_engineering/
│   ├── modeling/
│   └── visualization/
├── requirements.txt
└── README.md
```

## Modules

### 1. Exploratory Data Analysis (EDA)
- Univariate distribution analysis
- Multivariate correlation analysis
- Geographic impact analysis
- High-dimensional feature insights

### 2. Data Preprocessing & Feature Engineering
- Anomaly detection and handling
- Feature encoding and enhancement
- Imbalanced data processing

### 3. Machine Learning Modeling
- Churn prediction models (XGBoost, LightGBM, Stacking)
- Order count prediction
- Feature importance analysis

### 4. AB Testing Analysis
- Test design and implementation
- Statistical analysis
- Results visualization

### 5. Dashboard Development
- Interactive visualization using Plotly and Dash
- Real-time monitoring capabilities
- AB test results tracking

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Data-analysis-of-a-live-e-commerce-platform.git
cd Data-analysis-of-a-live-e-commerce-platform
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Key Findings and Recommendations

### High-Value Customer Retention
- Implement VIP exclusive discounts for customers with Tenure>12 months
- Optimize homepage recommendations based on user behavior

### Operational Strategy
- Trigger push notifications for users inactive for >7 days
- Implement targeted coupon distribution based on user segments

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost, LightGBM
- Plotly, Dash
- Matplotlib, Seaborn

## License
This project is licensed under the MIT License - see the LICENSE file for details. 