from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models.user import AuthUser
from database.database import SessionLocal

router = APIRouter()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/users/{user_id}")
def get_user_profile(user_id: int, db: Session = Depends(get_db)):
    user = db.query(AuthUser).filter(AuthUser.register_id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "name": user.full_name,  # Using full_name from your AuthUser model
        "email": user.email
    }