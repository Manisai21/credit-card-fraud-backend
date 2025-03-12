from fastapi import APIRouter, File, UploadFile
import pandas as pd
from utils.model_utils import load_model, preprocess_data

router = APIRouter()

@router.post("/xgboost/predict/")
async def predict_xgboost(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(contents)
    df = preprocess_data(df)
    
    model = load_model('xgboost')
    predictions = model.predict(df)
    
    return {"predictions": predictions.tolist()}