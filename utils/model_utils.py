# import joblib
# import pandas as pd
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split

# def load_model(model_name):
#     try:
#         return joblib.load(f'models/{model_name}.pkl')
#     except Exception as e:
#         raise Exception(f"Error loading model {model_name}: {e}")

# def preprocess_data(data):
#     try:
#         # Example preprocessing: Fill missing values and ensure correct data types
#         data.fillna(0, inplace=True)
#         # Add more preprocessing steps as needed
#         return data
#     except Exception as e:
#         raise Exception(f"Error in preprocessing: {e}")

# def train_and_save_model(df, model_name):
#     try:
#         # Preprocess the data
#         X = df.drop('Class', axis=1)
#         y = df['Class']

#         # Split the data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Train the model
#         model = LogisticRegression()
#         model.fit(X_train, y_train)

#         # Evaluate the model
#         predictions = model.predict(X_test)
#         accuracy = accuracy_score(y_test, predictions)
#         print(f"Model Accuracy: {accuracy}")

#         # Save the model
#         joblib.dump(model, f'models/{model_name}.pkl')
#         # Load the model
#         # model = joblib.load(f'models/{model_name}.pkl')

#         # Print model details
#         # print(model)

#         return accuracy
#     except Exception as e:
#         raise Exception(f"Error training model: {e}")

# import joblib
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score

# def load_model(model_name):
#     try:
#         return joblib.load(f'models/{model_name}.pkl')
#     except Exception as e:
#         raise Exception(f"Error loading model {model_name}: {e}")

# def preprocess_data(data):
#     try:
#         # Example preprocessing: Fill missing values and ensure correct data types
#         data.fillna(0, inplace=True)
#         # Add more preprocessing steps as needed
#         return data
#     except Exception as e:
#         raise Exception(f"Error in preprocessing: {e}")


# def train_and_save_model(df, model_name):
#     try:
#         # Preprocess the data
#         X = df.drop('Class', axis=1)
#         y = df['Class']

#         # Split the data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Select the model
#         if model_name == 'logistic_regression':
#             model = LogisticRegression()
#         elif model_name == 'random_forest':
#             model = RandomForestClassifier()
#         elif model_name == 'xgboost':
#             model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
#         else:
#             raise Exception("Model not supported")

#         # Train the model
#         model.fit(X_train, y_train)

#         # Evaluate the model
#         predictions = model.predict(X_test)
#         accuracy = accuracy_score(y_test, predictions)
#         print(f"Model Accuracy: {accuracy}")

#         # Save the model
#         joblib.dump(model, f'models/{model_name}.pkl')

#         # Return accuracy and predictions
#         return accuracy, predictions, X_test
#     except Exception as e:
#         raise Exception(f"Error training model: {e}")

# import joblib
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler

# def load_model(model_name):
#     try:
#         return joblib.load(f'models/{model_name}.pkl')
#     except Exception as e:
#         raise Exception(f"Error loading model {model_name}: {e}")

# def preprocess_data(data):
#     try:
#         # Example preprocessing: Fill missing values and ensure correct data types
#         data.fillna(0, inplace=True)
#         # Add more preprocessing steps as needed
#         return data
#     except Exception as e:
#         raise Exception(f"Error in preprocessing: {e}")

# def train_and_save_model(df, model_name):
#     try:
#         # Limit the dataset to the first 500 rows to reduce latency
#         df = df.head(500)

#         # Preprocess the data
#         X = df.drop('Class', axis=1)
#         y = df['Class']

#         # Check if there are at least two classes in the target variable
#         if y.nunique() < 2:
#             raise ValueError("The target variable must have at least two classes for training.")

#         # Feature scaling
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)

#         # Split the data
#         X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#         # Select the model
#         if model_name == 'logistic_regression':
#             model = LogisticRegression(max_iter=2000, solver='saga')  # Increased max_iter and changed solver
#         elif model_name == 'random_forest':
#             model = RandomForestClassifier()
#         elif model_name == 'xgboost':
#             model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
#         else:
#             raise Exception("Model not supported")

#         # Train the model
#         model.fit(X_train, y_train)

#         # Evaluate the model
#         predictions = model.predict(X_test)
#         accuracy = accuracy_score(y_test, predictions)
#         print(f"Model Accuracy: {accuracy}")

#         # Save the model
#         joblib.dump(model, f'models/{model_name}.pkl')

#         # Select important columns to return
#         important_columns = ['Time', 'V1', 'V2', 'V3', 'V4']  # Add or modify as needed
#         data_to_return = pd.DataFrame(X_test, columns=X.columns)[important_columns].copy()
#         data_to_return['Prediction'] = predictions

#         # Return accuracy and predictions
#         return accuracy, data_to_return
#     except ValueError as ve:
#         raise Exception(f"Error training model: {ve}")
#     except Exception as e:
#         raise Exception(f"Error training model: {e}")


import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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
        # Check if there are at least two classes in the target variable
        if df['Class'].nunique() < 2:
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
            model = LogisticRegression(max_iter=2000, solver='saga')  # Increased max_iter and changed solver
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

        # Save the model
        joblib.dump(model, f'models/{model_name}.pkl')

        # Select important columns to return
        important_columns = ['Time', 'V1', 'V2', 'V3', 'V4']  # Add or modify as needed
        data_to_return = pd.DataFrame(X_test, columns=X.columns)[important_columns].copy()
        data_to_return['Prediction'] = predictions

        # Return accuracy, predictions, and the model
        return accuracy, data_to_return, model
    except ValueError as ve:
        raise Exception(f"Error training model: {ve}")
    except Exception as e:
        raise Exception(f"Error training model: {e}")