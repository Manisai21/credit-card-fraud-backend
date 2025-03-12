# 


from fastapi import APIRouter, File, UploadFile, HTTPException
import pandas as pd
from utils.model_utils import train_and_save_model
import io

router = APIRouter()

@router.post("/logistic/train/")
async def train_logistic(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))  # Convert bytes to a file-like object
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")

    try:
        accuracy = train_and_save_model(df, 'logistic_regression')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {e}")

    return {"message": "Model trained and saved successfully", "accuracy": accuracy}