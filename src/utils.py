import os 
import sys
from src.exception import CustomException
from src.logger import logging
import dill

from sklearn.metrics import f1_score

def save_object(file_path, obj):
    """
    Save an object to a file path using dill.
    Args:
        file_path (str): The path where the object should be saved.
        obj (object): The object to save.
    """
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

def evaluate_models(X_train, y_train, X_test, y_test, models, param_grid = None):
    """
    Evaluates models and returns their performance based on F1 score.
    Args:
        X_train (array-like): Training feature set.
        y_train (array-like): Training labels.
        X_test (array-like): Testing feature set.
        y_test (array-like): Testing labels.
        models (dict): A dictionary of model names and model instances to evaluate.
        param_grid (dict, optional): A dictionary of hyperparameters for models. Defaults to None.
    
    Returns:
        dict: A dictionary containing model names as keys and their respective F1 scores as values.
    """
    try:
        report = {}

        # Iterate over the models
        for model_name, model in models.items():
            # Get hyperparameters for the current model if they exist in param_grid
            params = param_grid.get(model_name, {}) if param_grid else {}

            # Logging model and parameters (if any)
            logging.info(f"Evaluating model: {model_name} with parameters: {params}")
            
            # Fit the model on training data
            model.fit(X_train, y_train)
            
            # Predictions for both train and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Compute the F1 score for both training and testing sets
            train_score = f1_score(y_train, y_train_pred, average="weighted")
            test_score = f1_score(y_test, y_test_pred, average="weighted")

            # Store both scores for the model
            report[model_name] = {
                'train_score': train_score,
                'test_score': test_score
            }

            # Logging the scores
            logging.info(f"{model_name} - Train F1 Score: {train_score:.4f}, Test F1 Score: {test_score:.4f}")

        return report
    except Exception as e:
        raise CustomException(e, sys)