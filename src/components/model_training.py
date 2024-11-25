import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_config = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:

            logging.info("Splitting training and testing input data!")

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "XGBoost": XGBClassifier(),
                "LightGBM": LGBMClassifier(verbosity = -1),
                "AdaBoost": AdaBoostClassifier(algorithm = "SAMME"),
                "Bagging Classifier": BaggingClassifier(),
                "Extra Trees": ExtraTreesClassifier(),
                "Support Vector Classifier (SVC)": SVC(probability = True),
                "Naive Bayes": GaussianNB(),
                "CatBoost": CatBoostClassifier(verbose = 0)
            }

            logging.info("Evaluating all models...")

            # Call evaluate_models to get the performance of each model
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)

            logging.info("Models evaluated, calculating the best model!")

            # Get the best model
            best_model_name = max(model_report, key=lambda model: model_report[model]['test_score'])
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score['test_score'] < 0.6:
                raise CustomException("No suitable model found with sufficient accuracy!")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_config,
                obj=best_model
            )

            logging.info(f"Best Model: {best_model_name} with F1-score: {best_model_score['test_score']:.4f}")

            # Display model performance
            print("\nModel Performance:")
            for model_name, scores in model_report.items():
                print(f"{model_name}: Train F1-score: {scores['train_score']:.4f}, Test F1-score: {scores['test_score']:.4f}")

            print(f"\nBest Model: {best_model_name} with F1-score: {best_model_score['test_score']:.4f}\n")

            # Evaluate the best model on the test set
            logging.info("Evaluating the best model on the test set...")
            predicted = best_model.predict(X_test)
            test_f1 = f1_score(y_test, predicted, average = 'weighted')

            # Display classification metrics
            print("\nClassification Report:")
            print(classification_report(y_test, predicted))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, predicted))

            return test_f1
        
        except Exception as e:
            raise CustomException(e, sys)