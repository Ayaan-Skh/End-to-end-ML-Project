import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import AdaBoostRegressor, RandomForestClassifier,RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from xgboost import XGBClassifier,XGBRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.metrics import r2_score
from dataclasses import dataclass
import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.neighbors import KNeighborsRegressor
from src.utils import evaluate_models,save_object

@dataclass 
class ModelTrainerConfig:
    # Define the path where the trained model will be saved
    train_model_file_path=os.path.join('artifacts','model.pkl')  
    print(train_model_file_path) 

class ModelTrainer:
    def __init__(self):
        # Initialize the ModelTrainerConfig when the ModelTrainer is instantiated
        self.model_trainer_config=ModelTrainerConfig()
        
    # This function is used to initiate the model training process
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting input data")
            # Split the training and testing data into features (X) and target (y)
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            # Define a dictionary containing various regression models
            models={
                'Random Forest':RandomForestRegressor(),
                'Decision Tree':DecisionTreeRegressor(),
                'XGBRegressor':XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                'Linear Regression':LinearRegression(),
                "AdaBoost Regressor":AdaBoostRegressor(),
                'K Neighbours Regressor':KNeighborsRegressor(),
                'Logistic Regressor':LogisticRegression()                
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                'K Neighbours Regressor':{},
                'Logistic Regressor':{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            # Evaluate the defined models using the training and testing data
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            
            # Get the best model score from the model report
            best_model_score=max(sorted(model_report.values()))
            
            # Get the name of the best model from the model report
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            
            # Retrieve the best model from the 'models' dictionary
            best_model=models[best_model_name]
            
            # Raise an exception if the best model score is below a certain threshold
            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            # Log information about the best model and its score
            logging.info(f"Best model is {best_model_name} wit score {best_model_score}")
            
            
            # Save the best model as a pickle file
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )
            
            # Make predictions using the best model
            predicted=best_model.predict(X_test)
            
            # Calculate the R-squared score to evaluate the model's performance
            r2_square=r2_score(y_test,predicted)
            return r2_square 
            
        except Exception as e:
            # Raise a custom exception if any error occurs during the process
            raise CustomException(e, sys)