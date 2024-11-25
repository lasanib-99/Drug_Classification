import os
import sys

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_training import ModelTrainerConfig
from src.components.model_training import ModelTrainer

@dataclass ## Decorator

class DataIngestionConfig: 
    # Any input required in data ingestion is passed through this class
    # Inputs given to the data ingestion component are:
    raw_data_path = os.path.join('artifacts', 'data.csv')
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    # Outputs stored in 'artifacts'


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig 
        # When DataIngestion class executes, above 3 paths get saved in 'ingestion_config' 

    def initiate_data_ingestion(self): # Read dataset
        logging.info("Starting data ingestion process")    

        try:
            df = pd.read_csv(os.path.join('notebook', 'data', 'drug_data.csv'))

            logging.info('Read the dataset as a dataframe')

            ## Create 'artifacts' folder to read raw, train and test sets
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)

            # To save raw data as a dataframe
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state = 1)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
# Combine data ingestion with transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _  = data_transformation.initiate_data_transformation(train_data, test_data)
# Combine with model training    
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr))       