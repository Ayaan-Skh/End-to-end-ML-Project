import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer ##It uses to apply different transformations to different columns
from dataclasses import dataclass
from sklearn.impute import SimpleImputer ##It will hep to handle missing values
from sklearn.pipeline import Pipeline ##It will help create pipeline of transformations
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

import os
import sys

from src.utils import save_object  ## used for saving the pickle file

## It will help to give the path or input required for data transformation
class DataTransformationConfig:
    processor_obj_file_path=os.path.join("artifacts",'processor.pkl')
    
class DataTransformation:
    def __init__(self):## This initiallizes the data transformation configuration
        self.data_transformation_config=DataTransformationConfig()
        
        ## This function will be used to get the data transformation features and transform them
    def get_data_transformation_features(self):
        '''This function is responsible for data transformation'''
        try:
            numerical_columns=['writing_score','reading_score']
            categorical_columns=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            logging.info('Numerical columns standard scaling completed')
            
            cate_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info('Categorical columns oneHotEncoding completed')
            
            preprocessor=ColumnTransformer(
                transformers=[
                ('num_pipeline',num_pipeline,numerical_columns),
                ('categorical_pipeline',cate_pipeline,categorical_columns)
                ]
            )
            logging.info('Preprocessing completed')
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
                
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Reading training and test data completed")
            
            logging.info("Obtaining processing info object")
            
            processing_obj=self.get_data_transformation_features()
            
            target_column_name='math_score'
            numerical_columns=['writing_score','reading_score']
            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on the training dataframe and testing dataframe")
            
            input_feature_train_arr=processing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=processing_obj.transform(input_feature_test_df)

            logging.info("Saved preprocessing object")

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.processor_obj_file_path,
                obj=processing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.processor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
        