import os
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from feature_engineering import engineer_features

# Global container to keep the trained model cached in system memory
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the serialized model into RAM once when the web server boots up, 
    preventing slow disk-read overhead on future API prediction requests.
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "demand_model.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Production model binary missing at: {model_path}. "
            "Please ensure server/model_training.py has been executed successfully."
        )
    
    # Load and cache the model globally
    ml_models["demand_model"] = joblib.load(model_path)
    yield
    # Clean up and free memory on server shutdown
    ml_models.clear()

# Initialize FastAPI with the lifespan caching configuration
app = FastAPI(
    title="Netflix Greenlight Project API",
    description="API serving real-time audience demand predictions.",
    version="1.0.0",
    lifespan=lifespan
)

# --- PYDANTIC DATA CONTRACTS ---

class MoviePitchInput(BaseModel):
    """
    Strict input validation contract mapping the raw text and metadata variables
    sent from the browser UI down to the backend dataframe shapes.
    """
    title: str = Field(..., example="Inception 2")
    budget: float = Field(..., gt=0, example=150000000.0)
    runtime: float = Field(..., gt=0, example=148.0)
    genres: List[str] = Field(..., example=["Action", "Science Fiction", "Thriller"])
    keywords: List[str] = Field(..., example=["dream", "subconscious", "heist"])

class PredictionResponse(BaseModel):
    """
    Output structure ensuring a clean JSON payload is returned to the dashboard,
    ready to be ingested by D3.js network links.
    """
    title: str
    predicted_demand_score: float
    status: str

# --- CORE INFERENCE ROUTE ---
@app.post("/predict", response_model=PredictionResponse)
async def predict_pitch(payload: MoviePitchInput):
    try:
        # 1. Format incoming data for the AST parser
        formatted_genres = [{"name": g} for g in payload.genres]
        formatted_keywords = [{"name": k} for k in payload.keywords]
        
        raw_data = {
            "title": [payload.title],
            "budget": [payload.budget],
            "runtime": [payload.runtime],
            "genres": [str(formatted_genres)],
            "keywords": [str(formatted_keywords)]
        }
        
        df_pitch = pd.DataFrame(raw_data)
        
        # Extract the cached model from memory
        model = ml_models.get("demand_model")
        if not model:
            raise HTTPException(status_code=503, detail="Predictive model is not loaded.")
        
        # Transform the raw input using your Week 2 logic
        X_raw_processed = engineer_features(df_pitch)
        
        # FEATURE ALIGNMENT BRIDGING
        # Pull the exact 730 features the Random Forest was trained on
        expected_features = model.feature_names_in_
        
        # Reindex forces the 1-row dataframe to match the 730 columns perfectly.
        # Existing columns keep their values; missing features (like '1950s') are filled with 0.
        X_aligned = X_raw_processed.reindex(columns=expected_features, fill_value=0)
        
        # Generate the real-time continuous prediction
        prediction = model.predict(X_aligned)[0]
        
        # Return response
        return PredictionResponse(
            title=payload.title,
            predicted_demand_score=round(float(prediction), 4),
            status="Success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution failure: {str(e)}")