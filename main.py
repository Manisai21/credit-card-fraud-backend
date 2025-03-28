from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import auth, login, profile, contact, forget_password  # Import the forget_password route
from routes.test_db import router as test_db_router
from routes.logistic import router as logistic_router
from routes.random_forest import router as random_forest_router
from routes.xgboost import router as xgboost_router
from routes.risk_score import router as risk_score_router
from routes.isolation_forest import router as isolation_forest_router
from routes.autoencoder import router as  autoencoder_router
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
app.include_router(login.router, prefix="/auth")
app.include_router(profile.router)  
app.include_router(contact.router)
app.include_router(forget_password.router, prefix="/auth")  # Include the forget_password router
app.include_router(test_db_router)
app.include_router(logistic_router, prefix="/logistic")
app.include_router(random_forest_router, prefix="/random_forest")
app.include_router(xgboost_router, prefix="/xgboost")
app.include_router(autoencoder_router,prefix="/autoencoder")
app.include_router(risk_score_router, prefix="/risk")
app.include_router(isolation_forest_router, prefix="/isolation_forest")

@app.get("/")
def home():
    return {"message": "Welcome to Fraud Detection API"}
