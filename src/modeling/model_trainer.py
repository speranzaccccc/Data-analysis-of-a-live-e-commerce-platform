import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Any
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import pickle

class ModelTrainer:
    def __init__(self):
        self.models = {
            'xgboost': xgb.XGBClassifier(random_state=42),
            'lightgbm': lgb.LGBMClassifier(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42)
        }
        self.best_model = None
        self.best_score = 0
        
    def train_evaluate_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate multiple models
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict[str, Dict[str, float]]: Performance metrics for each model
        """
        results = {}
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            results[name] = metrics
            
            # Update best model
            if metrics['roc_auc'] > self.best_score:
                self.best_score = metrics['roc_auc']
                self.best_model = model
                
        return results
    
    def save_model(self, filepath: str):
        """
        Save the best model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
            
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_model, f)
            
    @staticmethod
    def load_model(filepath: str) -> Any:
        """
        Load a saved model from disk
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            Any: Loaded model
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f) 