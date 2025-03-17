# from fastapi import APIRouter, File, UploadFile, HTTPException
# import pandas as pd
# from utils.model_utils import train_and_save_model
# import io

# router = APIRouter()

# @router.post("/train/")
# async def train_random_forest(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         df = pd.read_csv(io.BytesIO(contents))
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")

#     try:
#         accuracy, data_to_return, _ = train_and_save_model(df, 'random_forest')
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error training model: {e}")

#     return {
#         "message": "Model trained and saved successfully",
#         "accuracy": accuracy,
#         "data": data_to_return.to_dict(orient='records')
#     }
from fastapi import APIRouter, File, UploadFile, HTTPException
import pandas as pd
from utils.model_utils import train_and_save_model
import io

router = APIRouter()

@router.post("/train/")
async def train_random_forest(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")

    try:
        result = train_and_save_model(df, 'random_forest')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {e}")

    return {
        "message": "Model trained and saved successfully",
        "accuracy": result["accuracy"],
        "data": result["data"].to_dict(orient='records'),
        "confusion_matrix": result["confusion_matrix"],
        "roc_curve": result["roc_curve"],
        "feature_importances": result["feature_importances"],
        "fraud_distribution": result["fraud_distribution"],
        "fraud_over_time": result["fraud_over_time"]
    }