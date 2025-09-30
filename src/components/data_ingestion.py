import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

##This class will be used to take any type of input
@dataclass
class DataIngestionConfig:
    # Define the path for the training data
    train_data_path:str=os.path.join("artifacts",'train.csv')
    # Define the path for the test data
    test_data_path:str=os.path.join("artifacts",'test.csv')
    # Define the path for the raw data
    raw_data_path:str=os.path.join("artifacts",'raw.csv')
    
 
class DataIngestion:
    ## This function will initialize the data ingestion configuration
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
            
    ## This function will initiate the data ingestion form the source
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            ##Initially we are using stored data later we can change this to database or any other source
            df=pd.read_csv('../../notebook/data/stud.csv')
            logging.info("Read the dataset information")
            
            ## Create the directory for the artifacts if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            ## Save the raw data to the specified path
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Raw data added to artifacts folder")
            
            logging.info("Initialize train test split")
            ## Split the data into training and testing sets
            train_set,test_set=train_test_split(df,test_size=0.25,random_state=42)
            
            ## Save the training set to the specified path
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info("Added training data")
            
            ## Save the test set to the specified path
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Added test data")
            logging.info("Data ingestion completed")
            
            ## Return the paths to the training and testing data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.info("Error occured in ingestion")
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_array=train_arr,test_array=test_arr))