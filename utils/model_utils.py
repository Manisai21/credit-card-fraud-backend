import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def load_model(model_name):
    try:
        return joblib.load(f'models/{model_name}.pkl')
    except Exception as e:
        raise Exception(f"Error loading model {model_name}: {e}")

def preprocess_data(data):
    try:
        # Example preprocessing: Fill missing values and ensure correct data types
        data.fillna(0, inplace=True)
        # Add more preprocessing steps as needed
        return data
    except Exception as e:
        raise Exception(f"Error in preprocessing: {e}")

def train_and_save_model(df, model_name):
    try:
        # Preprocess the data
        X = df.drop('Class', axis=1)
        y = df['Class']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Evaluate the model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy}")

        # Save the model
        joblib.dump(model, f'models/{model_name}.pkl')
        # Load the model
        model = joblib.load(f'models/{model_name}.pkl')

        # Print model details
        print(model)

        return accuracy
    except Exception as e:
        raise Exception(f"Error training model: {e}")