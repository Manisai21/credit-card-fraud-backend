from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import auth, login  # Import the login route
from routes.test_db import router as test_db_router

app = FastAPI()

# CORS setup (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow frontend
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Register Routes
app.include_router(auth.router, prefix="/auth")
app.include_router(login.router, prefix="/auth")  # Add the login route
app.include_router(test_db_router)

@app.get("/")
def home():
    return {"message": "Welcome to Fraud Detection API"}
