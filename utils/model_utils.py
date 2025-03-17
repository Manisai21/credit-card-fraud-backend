# 

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler

def train_and_save_model(df, model_name):
    try:
        # Ensure the dataset has both classes
        class_counts = df['Class'].value_counts()
        if class_counts.min() < 1:
            raise ValueError("The target variable must have at least two classes for training.")

        # Stratified sampling to ensure both classes are represented
        df_sampled = df.groupby('Class', group_keys=False).apply(lambda x: x.sample(min(len(x), 250), random_state=42))

        # Preprocess the data
        X = df_sampled.drop('Class', axis=1)
        y = df_sampled['Class']

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        # Select the model
        if model_name == 'logistic_regression':
            model = LogisticRegression(max_iter=2000, solver='saga')
        elif model_name == 'random_forest':
            model = RandomForestClassifier()
        elif model_name == 'xgboost':
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        else:
            raise Exception("Model not supported")

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy}")

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, predictions)

        # Calculate ROC curve
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

        # Feature importances
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
        else:
            feature_importances = np.zeros(X.shape[1])

        # Fraud distribution and fraud over time
        fraud_distribution = df['Class'].value_counts().to_dict()
        fraud_over_time = df.groupby('Time')['Class'].sum().to_dict()

        # Save the model
        joblib.dump(model, f'models/{model_name}.pkl')

        # Select important columns to return
        important_columns = ['Time', 'V1', 'V2', 'V3', 'V4']
        data_to_return = pd.DataFrame(X_test, columns=X.columns)[important_columns].copy()
        data_to_return['Prediction'] = predictions

        # Return accuracy, predictions, and additional data for visualizations
        return {
            "accuracy": accuracy,
            "data": data_to_return,
            "confusion_matrix": cm.tolist(),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "feature_importances": feature_importances.tolist(),
            "fraud_distribution": fraud_distribution,
            "fraud_over_time": fraud_over_time
        }
    except ValueError as ve:
        raise Exception(f"Error training model: {ve}")
    except Exception as e:
        raise Exception(f"Error training model: {e}")