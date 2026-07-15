import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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


def run_pipeline():
    # 1. Load data and extract pristine feature matrix X
    df_raw = pd.read_csv("data/movies.csv")
    X = engineer_features(df_raw)
    Y = df_raw["popularity"]
    
    if "popularity" in X.columns:
        X = X.drop(columns=["popularity"])
        
    # 2. Train/Test Split (80/20)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 3. Fit OLS Baseline Model
    ols = LinearRegression()
    ols.fit(X_train, Y_train)
    ols_preds = ols.predict(X_test)

    # 4. Fit Upgraded Random Forest Ensemble Model
    # n_jobs=-1 distributes the calculation across all available CPU cores
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, Y_train)
    rf_preds = rf.predict(X_test)

    # 5. Output Evaluation Comparison Metrics
    ols_mae = mean_absolute_error(Y_test, ols_preds)
    ols_r2 = r2_score(Y_test, ols_preds)
    
    rf_mae = mean_absolute_error(Y_test, rf_preds)
    rf_r2 = r2_score(Y_test, rf_preds)

    print("--- MODEL PERFORMANCE COMPARISON ---")
    print(f"OLS Baseline  -> MAE: {ols_mae:.2f} | R2: {ols_r2:.4f}")
    print(f"Random Forest -> MAE: {rf_mae:.2f} | R2: {rf_r2:.4f}")

    # 6. Serialization Layer (FastAPI / Deployment Boundary)
    # Save the trained ensemble weights so the live server can reference them instantly
    os.makedirs("models", exist_ok=True)
    model_output_path = "models/demand_model.pkl"
    joblib.dump(rf, model_output_path)
    print(f"Production model successfully serialized to: {model_output_path}")

def analyze_feature_signals():
    df_raw = pd.read_csv("data/movies.csv")
    X = engineer_features(df_raw)
    Y = df_raw["popularity"]
    
    if "popularity" in X.columns:
        X = X.drop(columns=["popularity"])
        
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train an initial Random Forest to calculate importance weights
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, Y_train)

    # Pair column names with their importance metrics
    importances = rf.feature_importances_
    feature_ranking = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Display the Top 20 most powerful signals
    print("\n--- TOP 20 PREDICTIVE SIGNALS ---")
    print(feature_ranking.head(20).to_string(index=False))
    print("---------------------------------\n")

    # Measure how much total variance explanation remains in the top 150 features
    top_150_importance = feature_ranking.head(150)["Importance"].sum()
    print(f"Cumulative importance captured by the top 150 features: {top_150_importance:.4f}")

if __name__ == "__main__":
    run_ols_baseline()
    run_pipeline()
    analyze_feature_signals()