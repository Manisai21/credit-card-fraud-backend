import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense

def create_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(32, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

def train_and_save_model(df, model_name):
    try:
        # Validate input data
        if df.isnull().values.any():
            raise ValueError("Input data contains missing values.")

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

        if model_name in ['logistic_regression', 'random_forest', 'xgboost']:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

            # Select the model
            if model_name == 'logistic_regression':
                model = LogisticRegression(max_iter=2000, solver='saga')
            elif model_name == 'random_forest':
                model = RandomForestClassifier()
            elif model_name == 'xgboost':
                model = XGBClassifier(eval_metric='logloss')
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
            feature_importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.zeros(X.shape[1])

            # Fraud distribution and fraud over time
            fraud_distribution = df['Class'].value_counts().to_dict()
            fraud_over_time = df.groupby('Time')['Class'].sum().to_dict()

            # Save the model
            joblib.dump(model, f'models/{model_name}.pkl')

            # Select important columns to return
            important_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25']
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

        elif model_name == 'isolation_forest':
            # Train Isolation Forest model
            model = IsolationForest(contamination=0.1)
            model.fit(X_scaled)

            # Predict anomalies
            anomaly_scores = model.decision_function(X_scaled)
            threshold = np.percentile(anomaly_scores, 10)
            predictions = (anomaly_scores < threshold).astype(int)

            # Calculate confusion matrix
            cm = confusion_matrix(y, predictions)

            # Save the model
            joblib.dump(model, 'models/isolation_forest.pkl')

            # Filter for fraud predictions (where Prediction is 1)
            df_sampled['Prediction'] = predictions
            fraud_data = df_sampled[df_sampled['Prediction'] == 1]

            return {
                "accuracy": None,
                "data": fraud_data.to_dict(orient='records'),
                "confusion_matrix": cm.tolist(),
                "anomaly_scores": anomaly_scores.tolist(),
                "fraud_distribution": df['Class'].value_counts().to_dict(),
                "fraud_over_time": df.groupby('Time')['Class'].sum().to_dict()
            }

        elif model_name == 'autoencoder':
            # Create and train the autoencoder
            autoencoder = create_autoencoder(X_scaled.shape[1])
            autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

            # Save the model
            autoencoder.save('models/autoencoder.h5')

            # Predict anomalies
            X_pred = autoencoder.predict(X_scaled)
            mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
            threshold = np.percentile(mse, 95)
            predictions = (mse > threshold).astype(int)

            # Calculate confusion matrix
            cm = confusion_matrix(y, predictions)

            # Filter for fraud predictions (where Prediction is 1)
            df_sampled['Prediction'] = predictions
            fraud_data = df_sampled[df_sampled['Prediction'] == 1]

            return {
                "accuracy": None,
                "data": fraud_data.to_dict(orient='records'),
                "confusion_matrix": cm.tolist(),
                "reconstruction_errors": mse.tolist(),
                "fraud_distribution": df['Class'].value_counts().to_dict(),
                "fraud_over_time": df.groupby('Time')['Class'].sum().to_dict()
            }

        else:
            raise Exception("Model not supported")

    except ValueError as ve:
        raise Exception(f"Error training model: {ve}")
    except Exception as e:
        raise Exception(f"Error training model: {e}")