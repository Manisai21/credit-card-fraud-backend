# 


from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.risk_score_utils import RiskScoreAnalyzer

router = APIRouter()

class TransactionData(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    Amount: float

@router.post("/{model_name}/calculate-risk")
async def calculate_risk_score(model_name: str, transaction: TransactionData):
    """Calculate risk score for a specific model"""
    try:
        valid_models = ['logistic_regression', 'random_forest', 'xgboost']
        if model_name not in valid_models:
            raise HTTPException(status_code=400, detail="Invalid model name.")

        analyzer = RiskScoreAnalyzer()
        risk_analysis = analyzer.calculate_model_risk_score(transaction.dict(), model_name)
        return risk_analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/all-models/calculate-risk")
async def calculate_all_models_risk(transaction: TransactionData):
    """Calculate risk scores for all available models"""
    try:
        analyzer = RiskScoreAnalyzer()
        models = ['logistic_regression', 'random_forest', 'xgboost']
        results = {}
        
        for model_name in models:
            try:
                risk_analysis = analyzer.calculate_model_risk_score(transaction.dict(), model_name)
                results[model_name] = risk_analysis
            except Exception as e:
                results[model_name] = {"error": str(e)}
                
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))