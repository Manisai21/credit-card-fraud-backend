# import numpy as np
# import joblib
# from sklearn.preprocessing import StandardScaler
# from typing import Dict

# class RiskScoreAnalyzer:
#     def __init__(self):
#         self.risk_weights = {
#             'model_probability': 0.6,
#             'amount_factor': 0.2,
#             'time_factor': 0.1,
#             'v_factors': 0.1
#         }
    
#     def calculate_model_risk_score(self, transaction_data: Dict, model_name: str) -> Dict:
#         """Calculate risk score for a specific model"""
#         try:
#             # Load the specified model
#             model = self._load_model(model_name)
            
#             # Prepare features
#             features = self._prepare_features(transaction_data)
            
#             # Get model's probability of fraud
#             fraud_probability = model.predict_proba(features)[0][1]
            
#             # Calculate component scores
#             model_score = fraud_probability * 100
#             amount_score = self._calculate_amount_score(transaction_data['Amount'])
#             time_score = self._calculate_time_score(transaction_data['Time'])
#             v_score = self._calculate_v_score([
#                 transaction_data['V1'],
#                 transaction_data['V2'],
#                 transaction_data['V3'],
#                 transaction_data['V4']
#             ])
            
#             # Calculate final weighted risk score
#             final_score = (
#                 self.risk_weights['model_probability'] * model_score +
#                 self.risk_weights['amount_factor'] * amount_score +
#                 self.risk_weights['time_factor'] * time_score +
#                 self.risk_weights['v_factors'] * v_score
#             )
            
#             # Normalize to 0-100 range
#             final_score = min(max(final_score, 0), 100)
#             risk_level = self._get_risk_level(final_score)
            
#             return {
#                 'model_name': model_name,
#                 'risk_score': round(final_score, 2),
#                 'risk_level': risk_level,
#                 'components': {
#                     'model_probability': round(model_score, 2),
#                     'amount_score': round(amount_score, 2),
#                     'time_score': round(time_score, 2),
#                     'v_score': round(v_score, 2)
#                 },
#                 'recommendation': self._get_recommendation(risk_level)
#             }
            
#         except Exception as e:
#             raise Exception(f"Error calculating risk score for {model_name}: {e}")

#     def _load_model(self, model_name: str):
#         """Load a specific model"""
#         try:
#             return joblib.load(f'models/{model_name}.pkl')
#         except Exception as e:
#             raise Exception(f"Could not load model {model_name}: {e}")

#     def _prepare_features(self, transaction_data: Dict) -> np.ndarray:
#         scaler = StandardScaler()
#         features = np.array([[
#             transaction_data['Time'],
#             transaction_data['V1'],
#             transaction_data['V2'],
#             transaction_data['V3'],
#             transaction_data['V4'],
#             transaction_data['Amount']
#         ]])
#         return scaler.fit_transform(features)
    
#     def _calculate_amount_score(self, amount: float) -> float:
#         if amount > 5000: return 100
#         elif amount > 1000: return 75
#         elif amount > 500: return 50
#         elif amount > 100: return 25
#         return 0
    
#     def _calculate_time_score(self, time: float) -> float:
#         hour = (time / 3600) % 24
#         if 1 <= hour <= 5: return 100
#         elif 23 <= hour or hour <= 6: return 75
#         return 0
    
#     def _calculate_v_score(self, v_values: list) -> float:
#         return sum(100 * (1 / (1 + np.exp(v))) for v in v_values) / len(v_values)
    
#     def _get_risk_level(self, score: float) -> str:
#         if score >= 75: return "HIGH"
#         elif score >= 45: return "MEDIUM"
#         return "LOW"
    
#     def _get_recommendation(self, risk_level: str) -> str:
#         recommendations = {
#             "HIGH": "Block transaction and require manual review",
#             "MEDIUM": "Request additional verification",
#             "LOW": "Approve transaction"
#         }
#         return recommendations.get(risk_level, "Unknown risk level")

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from typing import Dict

class RiskScoreAnalyzer:
    def __init__(self):
        self.risk_weights = {
            'model_probability': 0.6,
            'amount_factor': 0.2,
            'time_factor': 0.1,
            'v_factors': 0.1
        }
    
    def calculate_model_risk_score(self, transaction_data: Dict, model_name: str) -> Dict:
        """Calculate risk score for a specific model"""
        try:
            # Load the specified model
            model = joblib.load(f'models/{model_name}.pkl')
            
            # Prepare features
            features = self._prepare_features(transaction_data)
            
            # Get model's probability of fraud
            fraud_probability = model.predict_proba(features)[0][1]
            
            # Calculate component scores
            model_score = fraud_probability * 100
            amount_score = self._calculate_amount_score(transaction_data['Amount'])
            time_score = self._calculate_time_score(transaction_data['Time'])
            v_score = self._calculate_v_score([
                transaction_data['V1'],
                transaction_data['V2'],
                transaction_data['V3'],
                transaction_data['V4']
            ])
            
            # Calculate final weighted risk score
            final_score = (
                self.risk_weights['model_probability'] * model_score +
                self.risk_weights['amount_factor'] * amount_score +
                self.risk_weights['time_factor'] * time_score +
                self.risk_weights['v_factors'] * v_score
            )
            
            # Normalize to 0-100 range
            final_score = min(max(final_score, 0), 100)
            risk_level = self._get_risk_level(final_score)
            
            return {
                'model_name': model_name,
                'risk_score': round(final_score, 2),
                'risk_level': risk_level,
                'components': {
                    'model_probability': round(model_score, 2),
                    'amount_score': round(amount_score, 2),
                    'time_score': round(time_score, 2),
                    'v_score': round(v_score, 2)
                },
                'recommendation': self._get_recommendation(risk_level)
            }
            
        except Exception as e:
            raise Exception(f"Error calculating risk score for {model_name}: {e}")

    def _prepare_features(self, transaction_data: Dict) -> np.ndarray:
        scaler = StandardScaler()
        features = np.array([[
            transaction_data['Time'],
            transaction_data['V1'],
            transaction_data['V2'],
            transaction_data['V3'],
            transaction_data['V4'],
            transaction_data['Amount']
        ]])
        return scaler.fit_transform(features)
    
    def _calculate_amount_score(self, amount: float) -> float:
        if amount > 5000: return 100
        elif amount > 1000: return 75
        elif amount > 500: return 50
        elif amount > 100: return 25
        return 0
    
    def _calculate_time_score(self, time: float) -> float:
        hour = (time / 3600) % 24
        if 1 <= hour <= 5: return 100
        elif 23 <= hour or hour <= 6: return 75
        return 0
    
    def _calculate_v_score(self, v_values: list) -> float:
        return sum(100 * (1 / (1 + np.exp(v))) for v in v_values) / len(v_values)
    
    def _get_risk_level(self, score: float) -> str:
        if score >= 75: return "HIGH"
        elif score >= 45: return "MEDIUM"
        return "LOW"
    
    def _get_recommendation(self, risk_level: str) -> str:
        recommendations = {
            "HIGH": "Block transaction and require manual review",
            "MEDIUM": "Request additional verification",
            "LOW": "Approve transaction"
        }
        return recommendations.get(risk_level, "Unknown risk level")
