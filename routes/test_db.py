from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy.sql import text
from database.database import SessionLocal

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/test-db")
def test_db_connection(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))  # Check if DB is connected
        return {"message": "Database connected successfully!"}
    except Exception as e:
        return {"error": str(e)}
