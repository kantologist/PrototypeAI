from fastapi import FastAPI, HTTPException
from schema import InputData
import joblib
import pandas as pd
import logging
import pickle
from contextlib import asynccontextmanager
import uvicorn
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
# warnings.simplefilter("error", InconsistentVersionWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and transformer
model = None

async def load_model():
    """Load model asynchronously"""
    global model
    try:
        logger.info("Loading Model...")
        # model = pickle.load(open('model.pk', 'rb'))
        # model = joblib.load(open('model.pkl', 'rb'))
        model = joblib.load('model.pkl')
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model
    await load_model()
    yield
    # Shutdown: cleanup if needed
    pass

# Initialize FastAPI app with lifespan
# app = FastAPI(lifespan=lifespan)

app = FastAPI(
    title="Loan Default Prediction API",
    description="API for predicting Loan Default using a pre-trained model",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check that works even during model loading"""
    if model is None:
        return {"status": "loading", "message": "Model is still loading"}
    return {"status": "healthy", "message": "Model is loaded and ready"}

@app.get("/ready")
async def readiness_check():
    """Readiness check - only returns ready when model is loaded"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ready", "message": "Service is ready to handle requests"}

@app.post("/predict")
async def predict(data: InputData):
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again shortly.")
    
    # Convert input to dictionary
    input_dict = data.dict()
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Extract only the features needed for the model
    X_array = input_df.values
    
    # Model prediction
    y_pred = model.predict(X_array)[0]
    y_proba = model.predict_proba(X_array)[:, 1][0]
    
    return {
        "prediction": int(y_pred),
        "probability": float(y_proba)
    }

@app.get("/")
async def root():
    return {"message": "Loan Default Prediction API", "status": "running"}

if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port=8080)