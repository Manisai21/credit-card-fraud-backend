from fastapi import APIRouter, File, UploadFile
import pandas as pd
from utils.model_utils import load_model, preprocess_data

router = APIRouter()

@router.post("/random_forest/predict/")
async def predict_random_forest(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(contents)
    df = preprocess_data(df)
    
    model = load_model('random_forest')
    predictions = model.predict(df)
    
    return {"predictions": predictions.tolist()}