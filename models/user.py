from sqlalchemy import Column, Integer, String
from database.database import Base, engine

class AuthUser(Base):
    __tablename__ = "authuser"

    register_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    full_name = Column(String(45), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    phone_no = Column(String(15), nullable=False)  # Change to String
    loc = Column(String(45), nullable=False)
    password = Column(String(255), nullable=False)
    reset_token = Column(String(255), nullable=True)

Base.metadata.create_all(bind=engine)  # Create table if not exists
