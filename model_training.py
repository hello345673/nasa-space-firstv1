"""
Machine Learning Model Training for Exoplanet Detection
Uses ensemble methods for high accuracy classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib
import json
from data_preprocessing import ExoplanetDataPreprocessor
import warnings
warnings.filterwarnings('ignore')


class ExoplanetClassifier:
    """Ensemble classifier for exoplanet detection"""
    
    def __init__(self, model_type='ensemble'):
        self.model_type = model_type
        self.model = None
        self.preprocessor = ExoplanetDataPreprocessor()
        self.metrics = {}
        self.feature_importance = None
        
    def create_model(self):
        """Create ensemble model with multiple classifiers"""
        if self.model_type == 'ensemble':
            # Create individual models
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            xgb_model = XGBClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
            lgbm_model = LGBMClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            
            gb_model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            )
            
            # Create voting ensemble
            self.model = VotingClassifier(
                estimators=[
                    ('rf', rf_model),
                    ('xgb', xgb_model),
                    ('lgbm', lgbm_model),
                    ('gb', gb_model)
                ],
                voting='soft',
                n_jobs=-1
            )
            
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=25,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        
        elif self.model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=300,
                max_depth=15,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, use_smote=True):
        """Train the model with optional SMOTE for class imbalance"""
        print(f"\nTraining {self.model_type} model...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Handle class imbalance with SMOTE
        if use_smote:
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE: {len(X_train_balanced)} samples")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Create and train model
        self.create_model()
        self.model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate on validation set
        print("\nEvaluating model...")
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba)
        }
        
        print("\nValidation Metrics:")
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['Non-Exoplanet', 'Exoplanet']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        
        # Feature importance (for applicable models)
        self._calculate_feature_importance()
        
        return self.metrics
    
    def _calculate_feature_importance(self):
        """Calculate feature importance from the model"""
        try:
            if self.model_type == 'ensemble':
                # Average feature importance across ensemble models
                importances = []
                for name, estimator in self.model.named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        importances.append(estimator.feature_importances_)
                
                if importances:
                    self.feature_importance = np.mean(importances, axis=0)
            elif hasattr(self.model, 'feature_importances_'):
                self.feature_importance = self.model.feature_importances_
        except Exception as e:
            print(f"Could not calculate feature importance: {e}")
    
    def predict(self, X):
        """Make predictions on new data"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)
    
    def save_model(self, model_path='models/exoplanet_model.pkl', 
                   preprocessor_path='models/preprocessor.pkl',
                   metrics_path='models/metrics.json'):
        """Save model, preprocessor, and metrics"""
        import os
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.preprocessor, preprocessor_path)
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"\nModel saved to {model_path}")
        print(f"Preprocessor saved to {preprocessor_path}")
        print(f"Metrics saved to {metrics_path}")
    
    def load_model(self, model_path='models/exoplanet_model.pkl',
                   preprocessor_path='models/preprocessor.pkl'):
        """Load saved model and preprocessor"""
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        print("Model and preprocessor loaded successfully")


def train_exoplanet_model():
    """Main training pipeline"""
    print("="*60)
    print("EXOPLANET DETECTION MODEL TRAINING")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = ExoplanetDataPreprocessor()
    
    # Load and prepare data
    kepler_path = "/Users/ronit/Downloads/cumulative_2025.10.04_07.54.21.csv"
    tess_path = "/Users/ronit/Downloads/TOI_2025.10.04_07.54.31.csv"
    k2_path = "/Users/ronit/Downloads/k2pandc_2025.10.04_07.54.41.csv"
    
    combined_df, features = preprocessor.prepare_combined_dataset(
        kepler_path, tess_path, k2_path
    )
    
    # Prepare features and labels
    X = combined_df[features].values
    y = combined_df['label'].values
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Preprocess features
    print("\nPreprocessing features...")
    X_train_scaled = preprocessor.preprocess_features(X_train)
    X_val_scaled = preprocessor.transform_features(X_val)
    X_test_scaled = preprocessor.transform_features(X_test)
    
    # Train model
    classifier = ExoplanetClassifier(model_type='ensemble')
    classifier.preprocessor = preprocessor
    
    metrics = classifier.train(X_train_scaled, y_train, X_val_scaled, y_val, use_smote=True)
    
    # Final test evaluation
    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)
    
    y_test_pred = classifier.predict(X_test_scaled)
    y_test_proba = classifier.predict_proba(X_test_scaled)[:, 1]
    
    test_metrics = {
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1_score': f1_score(y_test, y_test_pred),
        'test_roc_auc': roc_auc_score(y_test, y_test_proba)
    }
    
    print("\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Non-Exoplanet', 'Exoplanet']))
    
    # Update metrics with test results
    classifier.metrics.update(test_metrics)
    
    # Save model
    classifier.save_model()
    
    # Save feature names
    with open('models/feature_names.json', 'w') as f:
        json.dump(features, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return classifier


if __name__ == "__main__":
    classifier = train_exoplanet_model()

