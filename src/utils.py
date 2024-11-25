import os 
import sys

## Save the preprocessed data as a picker file
from src.exception import CustomException
import dill

import pickle

def save_object(file_path, obj):
    try:
        # Ensure the directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object using dill
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(f"Error in saving object: {e}", sys)

# Model Training
from sklearn.metrics import f1_score

def evaluate_models(X_train, y_train, X_test, y_test, models):
    
    try:
        report = {}

        # Iterate over the models
        for i in range(len(list(models))):
            
            model = list(models.values())[i]

            param = param[list(models.keys())]

            model.fit(X_train, y_train) # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = f1_score(y_train, y_train_pred)

            test_model_score = f1_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            report[list(models.keys())[i]] = train_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)