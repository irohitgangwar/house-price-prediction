"""
California Housing Price Prediction using Machine Learning
Author: Rohit Gangwar (https://github.com/irohitgangwar)
Description: End-to-end Machine Learning pipeline to predict median house values in California.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_and_preprocess_data():
    """Load California housing dataset and prepare features/target."""
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    
    # Rename target column to match conventional naming
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']
    
    return df, X, y


def evaluate_model(name: str, model, X_test, y_test):
    """Evaluate regression model and compute evaluation metrics."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f"\n--- {name} Performance ---")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    
    return {"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def main():
    print("=" * 60)
    print(" California Housing Price Prediction - Machine Learning Pipeline ")
    print("=" * 60)
    
    # 1. Load Data
    df, X, y = load_and_preprocess_data()
    print(f"Dataset Loaded Successfully: {df.shape[0]} samples, {df.shape[1]-1} features.")
    print("\nDataset Summary Statistics:")
    print(df.describe().T[['mean', 'std', 'min', '50%', 'max']])
    
    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"\nSplit: Train Set = {X_train.shape[0]} rows | Test Set = {X_test.shape[0]} rows")
    
    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Model Training & Evaluation
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = []
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        metrics = evaluate_model(name, model, X_test_scaled, y_test)
        results.append(metrics)
        trained_models[name] = model
        
    # Benchmark Summary Table
    results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
    print("\n" + "=" * 60)
    print(" Model Comparison Summary ")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # 5. Inference on Sample Data
    best_model = trained_models["Random Forest Regressor"]
    sample_house = pd.DataFrame([{
        'MedInc': 7.325,
        'HouseAge': 30.0,
        'AveRooms': 5.984,
        'AveBedrms': 1.0238,
        'Population': 280.0,
        'AveOccup': 2.20,
        'Latitude': 37.88,
        'Longitude': -122.23
    }])
    
    sample_scaled = scaler.transform(sample_house)
    predicted_price = best_model.predict(sample_scaled)[0]
    
    print("\n" + "=" * 60)
    print(" Sample Inference (New Unseen Property) ")
    print("=" * 60)
    for col, val in sample_house.iloc[0].items():
        print(f"  {col:<12}: {val}")
    print(f"\n--> Predicted Median House Value: ${predicted_price * 100000:,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
