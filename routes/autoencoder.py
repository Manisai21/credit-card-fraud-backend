import os
from fastapi import APIRouter, File, UploadFile, HTTPException
import pandas as pd
import io
from utils.model_utils import train_and_save_model  # Import the utility function
from utils.email_utils import send_email

router = APIRouter()

@router.post("/train/")
async def train_autoencoder(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")

    try:
        # Use the utility function to train the model
        result = train_and_save_model(df, 'autoencoder')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {e}")

    # Filter for fraud predictions (where Prediction is 1)
    fraud_data = result["data"]

    if fraud_data:
        # Convert fraud predictions to a list of dictionaries
        fraud_details = fraud_data

        # Prepare email content
        email_content = {
            "subject": "Fraud Alert: Transactions Detected",
            "body": f"The following transactions have been flagged as fraud:\n\n{fraud_details}",
            "to": os.getenv("ADMIN_EMAIL", "manisaisaduvala21@gmail.com")  # Use the environment variable for the recipient
        }
        send_email(email_content)

    # Return the exact structure from the utility function
    return {
        "message": "Autoencoder model trained and saved successfully",
        "accuracy": result["accuracy"],
        "data": result["data"],
        "confusion_matrix": result["confusion_matrix"],
        "reconstruction_errors": result["reconstruction_errors"],  # Include reconstruction errors in the response
        "fraud_distribution": result["fraud_distribution"],
        "fraud_over_time": result["fraud_over_time"]
    }