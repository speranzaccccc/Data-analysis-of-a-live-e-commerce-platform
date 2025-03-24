import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple

class FeatureProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def process_features(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
        numerical_cols: List[str]
    ) -> pd.DataFrame:
        """
        Process both categorical and numerical features
        
        Args:
            df (pd.DataFrame): Input dataset
            categorical_cols (List[str]): List of categorical column names
            numerical_cols (List[str]): List of numerical column names
            
        Returns:
            pd.DataFrame: Processed dataset
        """
        df_processed = df.copy()
        
        # Process categorical features
        for col in categorical_cols:
            df_processed[col] = self._encode_categorical(df_processed[col], col)
            
        # Process numerical features
        df_processed[numerical_cols] = self.scaler.fit_transform(df_processed[numerical_cols])
        
        return df_processed
    
    def _encode_categorical(self, series: pd.Series, col_name: str) -> pd.Series:
        """
        Encode categorical variables using LabelEncoder
        
        Args:
            series (pd.Series): Input series
            col_name (str): Column name
            
        Returns:
            pd.Series: Encoded series
        """
        if col_name not in self.label_encoders:
            self.label_encoders[col_name] = LabelEncoder()
            return self.label_encoders[col_name].fit_transform(series)
        return self.label_encoders[col_name].transform(series)
    
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        interaction_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Create interaction features between specified pairs of columns
        
        Args:
            df (pd.DataFrame): Input dataset
            interaction_pairs (List[Tuple[str, str]]): List of column pairs
            
        Returns:
            pd.DataFrame: Dataset with interaction features
        """
        df_interactions = df.copy()
        
        for col1, col2 in interaction_pairs:
            feature_name = f"{col1}_{col2}_interaction"
            df_interactions[feature_name] = df[col1] * df[col2]
            
        return df_interactions 