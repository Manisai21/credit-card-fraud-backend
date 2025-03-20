import os
from fastapi import APIRouter, File, UploadFile, HTTPException
import pandas as pd
from utils.model_utils import train_and_save_model
import io
from utils.email_utils import send_email

router = APIRouter()

@router.post("/train/")
async def train_logistic(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")

    try:
        result = train_and_save_model(df, 'logistic_regression')
    except Exception as e:
        if "The target variable must have at least two classes for training" in str(e):
            raise HTTPException(status_code=500, detail="Error training model: The target variable must have at least two classes for training")
        else:
            raise HTTPException(status_code=500, detail=f"Error training model: {e}")

    # Check if 'data' key exists in result
    if "data" not in result:
        raise HTTPException(status_code=500, detail="Error: 'data' key not found in result")

    # Filter for fraud predictions (where Prediction is 0)
    fraud_data = result["data"]
    fraud_predictions = fraud_data[fraud_data['Prediction'] == 0]

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
        "accuracy": result.get("accuracy", "N/A"),
        "data": fraud_data.to_dict(orient='records'),
        "confusion_matrix": result.get("confusion_matrix", "N/A"),
        "roc_curve": result.get("roc_curve", "N/A"),
        "feature_importances": result.get("feature_importances", "N/A"),
        "fraud_distribution": result.get("fraud_distribution", "N/A"),
        "fraud_over_time": result.get("fraud_over_time", "N/A")
    }