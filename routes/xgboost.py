# from fastapi import APIRouter, File, UploadFile, HTTPException
# import pandas as pd
# from utils.model_utils import train_and_save_model
# import io

# router = APIRouter()

# @router.post("/train/")
# async def train_xgboost(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         df = pd.read_csv(io.BytesIO(contents))
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")

#     try:
#         accuracy, data_to_return, _ = train_and_save_model(df, 'xgboost')
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
from utils.email_utils import send_email  # Import the email utility
import os  # Import os to access environment variables

router = APIRouter()

@router.post("/train/")
async def train_xgboost(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")

    try:
        result = train_and_save_model(df, 'xgboost')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {e}")
    # Filter for fraud predictions (where Prediction is 1)
    fraud_data = result["data"]
    fraud_predictions = fraud_data[fraud_data['Prediction'] == 1]

    if not fraud_predictions.empty:
        # Convert fraud predictions to a list of dictionaries
        fraud_details = fraud_predictions.to_dict(orient='records')
        
        # Prepare email content
        email_content = {
            "subject": "Fraud Alert: Transactions Detected",
            "body": f"The following transactions have been flagged as fraud:\n\n{fraud_details}",
            "to": os.getenv("ADMIN_EMAIL", "manisaisaduvala21@gmail.com")  # Use the environment variable for the recipient
        }
        send_email(email_content)
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