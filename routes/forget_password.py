from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database.database import SessionLocal
from models.user import AuthUser  # Import AuthUser from the correct module
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import uuid  # For generating a unique token
from utils.security import hash_password  # Import hash_password for hashing

router = APIRouter()

# Define a Pydantic model for the request body
class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/forgot_password")
def forgot_password(request: ForgotPasswordRequest, db: Session = Depends(get_db)):
    # Check if the user exists in the database
    user = db.query(AuthUser).filter(AuthUser.email == request.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Generate a unique token for password reset
    token = str(uuid.uuid4())
    user.reset_token = token  # Assuming you have a reset_token field in your AuthUser model
    db.commit()

    # Send password reset email
    smtp_server = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER", "cursor1199@gmail.com")
    smtp_password = os.getenv("SMTP_PASSWORD", "bpgg apkp qpso syam")

    reset_link = f"http://localhost:3000/reset_password?token={token}"  # Link to reset password
    body = f"Click the link to reset your password: {reset_link}"

    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = user.email
    msg['Subject'] = "Password Reset Request"
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        return {"message": "Password reset email sent!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to send email: " + str(e))


@router.post("/reset_password")
def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    # Check if the token is valid
    user = db.query(AuthUser).filter(AuthUser.reset_token == request.token).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid token")  # Changed status code to 400

    # Hash the new password before saving
    hashed_password = hash_password(request.new_password)  # Use hash_password from utils
    user.password = hashed_password  # Store the hashed password
    user.reset_token = None  # Clear the reset token after use
    db.commit()

    return {"message": "Password has been reset successfully!"}