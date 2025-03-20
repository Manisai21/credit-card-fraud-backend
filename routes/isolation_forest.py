import os
from fastapi import APIRouter, File, UploadFile, HTTPException
import pandas as pd
import io
from utils.model_utils import train_and_save_model  # Import the utility function
from utils.email_utils import send_email

router = APIRouter()

@router.post("/train/")
async def train_isolation_forest(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")

    try:
        # Use the utility function to train the model
        result = train_and_save_model(df, 'isolation_forest')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {e}")

    # Filter for fraud predictions (where Prediction is -1)
    fraud_data = result.get("data", None)  # Use get to avoid KeyError

    if fraud_data:
        # Convert fraud predictions to a list of dictionaries
        fraud_details = fraud_data
    else:
        fraud_details = "N/A"  # Set to "N/A" if no fraud data

    # Prepare email content
    email_content = {
        "subject": "Fraud Alert: Transactions Detected",
        "body": f"The following transactions have been flagged as fraud:\n\n{fraud_details}",
        "to": os.getenv("ADMIN_EMAIL", "manisaisaduvala21@gmail.com")  # Use the environment variable for the recipient
    }
    send_email(email_content)

    # Return the exact structure from the utility function
    return {
        "message": "Isolation Forest model trained and saved successfully",
        "accuracy": result.get("accuracy", "N/A"),  # Set to "N/A" if not present
        "data": result.get("data", "N/A"),  # Set to "N/A" if not present
        "confusion_matrix": result.get("confusion_matrix", "N/A"),  # Set to "N/A" if not present
        "anomaly_scores": result.get("anomaly_scores", "N/A"),  # Set to "N/A" if not present
        "fraud_distribution": result.get("fraud_distribution", "N/A"),  # Set to "N/A" if not present
        "fraud_over_time": result.get("fraud_over_time", "N/A")  # Set to "N/A" if not present
    }