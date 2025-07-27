# In src/pipeline/train_pipeline.py

import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

# This 'if' statement is crucial. It ensures the code runs when you execute the script.
if __name__ == "__main__":
    try:
        # 1. Start with Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # 2. Perform Data Transformation
        # The paths from step 1 are passed to step 2
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        # 3. Perform Model Training
        # The transformed data from step 2 is passed to step 3
        model_trainer = ModelTrainer()
        # The r2_score will be printed to the console
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))

    except Exception as e:
        raise CustomException(e, sys)
    