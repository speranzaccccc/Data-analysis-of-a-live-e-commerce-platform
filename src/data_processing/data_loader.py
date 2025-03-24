import pandas as pd
import numpy as np
from typing import Optional, Tuple

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the e-commerce dataset from a CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    return pd.read_csv(file_path)

def split_features_target(
    df: pd.DataFrame,
    target_col: str = 'Churn',
    id_col: str = 'CustomerID'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataset into features and target
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Name of the target column
        id_col (str): Name of the ID column
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target
    """
    X = df.drop([target_col, id_col], axis=1)
    y = df[target_col]
    return X, y 