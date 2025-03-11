from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models.user import AuthUser
from utils.security import hash_password
from database.database import SessionLocal
from pydantic import BaseModel

router = APIRouter()

# Define a Pydantic model for the request body
class RegisterRequest(BaseModel):
    full_name: str
    email: str
    phone_no: str
    loc: str
    password: str
    
# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/register")
def register_user(request: RegisterRequest, db: Session = Depends(get_db)):
    hashed_password = hash_password(request.password)
    
    new_user = AuthUser(
        full_name=request.full_name,
        phone_no=request.phone_no,
        email=request.email,
        loc=request.loc,
        password=hashed_password
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {"message": "User registered successfully!", "user_id": new_user.register_id}
