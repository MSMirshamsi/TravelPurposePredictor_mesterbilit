# Trip Reason Prediction for MasterBilit


## Project Overview
This project develops a machine learning model to predict the reason for travel (Work or Leisure) based on ticket reservation data from MesterBilit, an Iranian travel platform. The goal is to enhance personalization and service offerings by accurately classifying trip purposes.

## Objectives
- Achieve an F1 Score > 75% for reliable predictions.
- Conduct advanced data preprocessing and feature engineering.
- Provide insightful visualizations for data exploration and model evaluation.

## Dataset
- **Training Data**: 101,017 rows, 22 columns (including `TripReason`).
- **Test Data**: 43,293 rows, 21 columns.
- **Key Features**: Reservation time, departure time, price, vehicle type, origin, destination.

## Methodology

### 1. Data Preprocessing
- **Missing Values**: Filled `UserID` with -1, `VehicleType` with "Unknown", and `VehicleClass` with False.
- **Feature Engineering**: Extracted temporal features (e.g., hour, day of week) and days to departure.
- **Encoding**: Applied `LabelEncoder` to categorical variables (`From`, `To`, `Vehicle`, `VehicleType`).
- **Normalization**: Scaled numerical features (`Price`, `CouponDiscount`, `Days_to_Departure`) using `StandardScaler`.

### 2. Exploratory Data Analysis (EDA)
- Visualized trip reason distribution, price vs. trip reason, and departure hour patterns.

### 3. Model Training
- **Algorithm**: Random Forest Classifier with hyperparameter tuning via `GridSearchCV`.
- **Parameters**:
  - `n_estimators`: 200
  - `max_depth`: None
  - `min_samples_split`: 2
- **Evaluation Metric**: F1 Score.

### 4. Evaluation
- **F1 Score**: [Insert achieved score, e.g., 0.82]
- **Visualizations**:
  - Confusion Matrix
  - Feature Importance
  - ROC Curve (AUC ≈ [Insert value])

## Results
The model achieved an F1 Score above the target threshold, demonstrating robust performance. Key features influencing predictions include `Price`, `Days_to_Departure`, and `Departure_Hour`.

## Installation

### Prerequisites
- Python 3.8+
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`

### Setup
```bash
git clone https://github.com/[YourUsername]/TripReasonPrediction.git
cd TripReasonPrediction
pip install -r requirements.txt
