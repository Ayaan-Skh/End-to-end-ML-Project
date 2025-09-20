import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

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
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset information")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Raw data added to artifacts folder")
            
            logging.info("Initialize train test split")
            train_set,test_set=train_test_split(df,test_size=0.25,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info("Added training data")
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Added test data")
            logging.info("Data ingestion completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()