import os
import joblib
import pandas as pd
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from feature_engineering import engineer_features
from agents import run_production_agent_pipeline
from dotenv import load_dotenv

load_dotenv()

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
    Input validation contract mapping incoming pitch data.
    """
    title: str = Field(..., json_schema_extra={"example": "Joseon Reborn"})
    pitch: str = Field(..., json_schema_extra={"example": "A story about a girl from the Joseon era..."})
    budget: float = Field(50000000.0, gt=0, json_schema_extra={"example": 45000000.0})
    runtime: float = Field(100.0, gt=0, json_schema_extra={"example": 110.0})

# --- CORE FEATURE BRIDGING INTERFACE ---

def prepare_model_input(agent_output: dict, target_model, budget: float, runtime: float):
    """
    Formats the clean structured text arrays out of the local AI agent crew
    to perfectly mirror the TMDB Kaggle string dictionary layout.
    """
    formatted_genres = [{"name": g} for g in agent_output.get("genres", [])]
    formatted_keywords = [{"name": k} for k in agent_output.get("keywords", [])]
    
    input_dict = {
        "budget": budget,
        "runtime": runtime,
        "genres": str(formatted_genres),
        "keywords": str(formatted_keywords)
    }
    
    # Process through your pristine feature engineering script
    df = pd.DataFrame([input_dict])
    X_processed = engineer_features(df)
    
    # Dynamically align to the exact 730 features expected by the Random Forest
    X_aligned = X_processed.reindex(columns=target_model.feature_names_in_, fill_value=0)    

    return X_aligned

# --- CORE INFERENCE ROUTE ---

@app.post("/predict")
async def predict_demand(payload: MoviePitchInput):
    # 1. Fetch the globally cached Random Forest model from memory
    model = ml_models.get("demand_model")
    if not model:
        raise HTTPException(status_code=503, detail="Predictive Random Forest model is not loaded.")
        
    try:
        # 2. Kick off local AI agents asynchronously to extract deep thematic tropes
        agent_features = await run_production_agent_pipeline(payload.pitch)
        
        # 3. Translate qualitative findings into the exact 730 feature matrix shape
        X_matrix = prepare_model_input(
            agent_output=agent_features, 
            target_model=model,
            budget=payload.budget, 
            runtime=payload.runtime
        )
        
        # 4. Generate the real-time continuous popularity estimation
        popularity_prediction = model.predict(X_matrix)[0]
        
        # 5. Construct the corporate output response payload
        return {
            "status": "success",
            "title": payload.title,
            "predicted_popularity": round(float(popularity_prediction), 4),
            "extracted_metadata": agent_features,
            "input_summary": f"Your project was analyzed as a {', '.join(agent_features.get('genres', []))} concept."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution failure: {str(e)}")