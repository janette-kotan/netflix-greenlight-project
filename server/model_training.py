import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from feature_engineering import engineer_features

def run_ols_baseline():    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "movies.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find dataset at {data_path}. Please check your folder structure.")
        
    df_raw = pd.read_csv(data_path)
    
    # Extract features by passing the raw DataFrame into feature engineering pipeline
    X = engineer_features(df_raw)
    
    # Define the Target Vector (Y) using audience popularity
    Y = df_raw["popularity"]
    
    # Prevent data leakage by ensuring target is dropped from X
    if "popularity" in X.columns:
        X = X.drop(columns=["popularity"])
        
    print(f"Feature Matrix (X) Shape: {X.shape}")
    print(f"Target Vector (Y) Shape: {Y.shape}")

    # Segment the Data into 80/20 split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train the OLS Baseline Model
    ols_model = LinearRegression()
    ols_model.fit(X_train, Y_train)

    # Predict and Evaluate Performance
    predictions = ols_model.predict(X_test)
    
    mae = mean_absolute_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)

    print("\n--- OLS BASELINE METRICS ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-Squared (R2): {r2:.4f}")
    
    return X_train, X_test, Y_train, Y_test

if __name__ == "__main__":
    run_ols_baseline()