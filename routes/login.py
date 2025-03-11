from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models.user import AuthUser
from utils.security import verify_password
from database.database import SessionLocal
from pydantic import BaseModel
import jwt
import os
from datetime import datetime, timedelta

router = APIRouter()

# Define a Pydantic model for the login request body
class LoginRequest(BaseModel):
    email: str
    password: str

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/login")
def login_user(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(AuthUser).filter(AuthUser.email == request.email).first()
    if not user or not verify_password(request.password, user.password):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    # Generate a JWT token using pyjwt
    token_data = {
        "user_id": user.register_id,
        "exp": datetime.utcnow() + timedelta(hours=1)  # Token expires in 1 hour
    }
    token = jwt.encode(token_data, os.getenv("SECRET_KEY", "your_secret_key"), algorithm="HS256")
    
    return {"message": "Login successful", "user_id": user.register_id, "token": token}