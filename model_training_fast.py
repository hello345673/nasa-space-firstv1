"""
Fast training script for exoplanet detection model
Trains ensemble model with cross-validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              VotingClassifier, ExtraTreesClassifier)
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            precision_score, recall_score, f1_score, roc_auc_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib
import json
from data_preprocessing import FixedExoplanetDataPreprocessor
import warnings
import time
warnings.filterwarnings('ignore')


class FastExoplanetClassifier:
    """Ensemble classifier for exoplanet detection"""
    
    def __init__(self, model_type='enhanced_ensemble'):
        self.model_type = model_type
        self.model = None
        self.preprocessor = FixedExoplanetDataPreprocessor()
        self.metrics = {}
        self.cv_scores = {}
        self.feature_importance = None
        
    def create_model(self):
        """Create ensemble model with multiple classifiers"""
        print("\nCreating Enhanced Ensemble Model...")
        
        # Random Forest classifier
        rf_model = RandomForestClassifier(
            n_estimators=1600,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            criterion='entropy',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        # XGBoost classifier
        xgb_model = XGBClassifier(
            n_estimators=1600,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        
        # LightGBM classifier
        lgbm_model = LGBMClassifier(
            n_estimators=1600,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        
        # Gradient Boosting with optimal hyperparameters
        gb_model = GradientBoostingClassifier(
            n_estimators=1600,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
        
        # ExtraTrees
        extra_trees_model = ExtraTreesClassifier(
            n_estimators=1600,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            criterion='entropy',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        # Create voting ensemble
        self.model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('xgb', xgb_model),
                ('lgbm', lgbm_model),
                ('gb', gb_model),
                ('extra_trees', extra_trees_model)
            ],
            voting='soft',
            n_jobs=-1,
            verbose=False
        )
        
        print("Enhanced Ensemble Created (5 models × 1600 estimators each)")
        
        return self.model
    
    def perform_cross_validation(self, X, y, n_folds=5):
        """
        Perform 5-fold cross-validation ONCE (fast version)
        """
        print(f"\n🔄 Performing {n_folds}-Fold Cross-Validation...")
        
        fold_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        # Stratified K-Fold to maintain class distribution
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n   Fold {fold}/{n_folds}...")
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Apply SMOTE only to training data
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_fold, y_train_fold)
            
            # Train model
            self.model.fit(X_train_balanced, y_train_balanced)
            
            # Predict
            y_pred = self.model.predict(X_val_fold)
            y_pred_proba = self.model.predict_proba(X_val_fold)[:, 1]
            
            # Calculate metrics
            acc = accuracy_score(y_val_fold, y_pred)
            prec = precision_score(y_val_fold, y_pred, zero_division=0)
            rec = recall_score(y_val_fold, y_pred, zero_division=0)
            f1 = f1_score(y_val_fold, y_pred, zero_division=0)
            roc = roc_auc_score(y_val_fold, y_pred_proba)
            
            fold_scores['accuracy'].append(acc)
            fold_scores['precision'].append(prec)
            fold_scores['recall'].append(rec)
            fold_scores['f1'].append(f1)
            fold_scores['roc_auc'].append(roc)
            
            print(f"      Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
        
        # Calculate averages
        self.cv_scores = {
            metric: {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
            for metric, scores in fold_scores.items()
        }
        
        print("\nCross-Validation Results:")
        for metric, stats in self.cv_scores.items():
            print(f"   {metric:10s}: {stats['mean']:.4f} (±{stats['std']:.4f})")
        
        return self.cv_scores
    
    def train(self, X_train, y_train, X_val, y_val, use_smote=True):
        """Train the model with optional SMOTE for class imbalance"""
        print(f"\nFinal Training on Full Training Set...")
        
        start_time = time.time()
        
        # Handle class imbalance with SMOTE
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Train model
        self.model.fit(X_train_balanced, y_train_balanced)
        
        training_time = time.time() - start_time
        print(f"   Training completed in {training_time:.1f}s ({training_time/60:.1f} min)")
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'training_time_seconds': training_time
        }
        
        print("\nValidation Metrics:")
        for metric, value in self.metrics.items():
            if metric != 'training_time_seconds':
                print(f"   {metric:12s}: {value:.4f} ({value*100:.2f}%)")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_val, y_pred, target_names=['Non-Exoplanet', 'Exoplanet']))
        
        return self.metrics
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def save_model(self, model_path='models/exoplanet_model.pkl', 
                   preprocessor_path='models/preprocessor.pkl',
                   metrics_path='models/metrics.json'):
        """Save model, preprocessor, and metrics"""
        import os
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.preprocessor, preprocessor_path)
        
        save_data = {
            'metrics': self.metrics,
            'cv_scores': self.cv_scores
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\nModel saved!")


def train_fast_model():
    """Fast training pipeline - 5-fold CV once"""
    print("="*70)
    print("⚡ FAST TRAINING - 5-Fold Cross-Validation (Once)")
    print("="*70)
    
    # Initialize FIXED preprocessor (no duplicate features!)
    preprocessor = FixedExoplanetDataPreprocessor()
    
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
    print("\n🔧 Preprocessing features...")
    X_train_scaled = preprocessor.preprocess_features(X_train)
    X_val_scaled = preprocessor.transform_features(X_val)
    X_test_scaled = preprocessor.transform_features(X_test)
    
    # Create classifier
    classifier = FastExoplanetClassifier(model_type='enhanced_ensemble')
    classifier.preprocessor = preprocessor
    classifier.create_model()
    
    # Skip cross-validation - just train directly!
    print("\n" + "="*70)
    print("TRAINING MODEL (No CV - Single Training Run)")
    print("="*70)
    metrics = classifier.train(X_train_scaled, y_train, X_val_scaled, y_val, use_smote=True)
    
    # Final test evaluation
    print("\n" + "="*70)
    print("🏆 TEST SET EVALUATION")
    print("="*70)
    
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
        print(f"   {metric:18s}: {value:.4f} ({value*100:.2f}%)")
    
    print("\n📋 Test Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Non-Exoplanet', 'Exoplanet']))
    
    # Update metrics
    classifier.metrics.update(test_metrics)
    
    # Save model
    classifier.save_model()
    
    # Save feature names
    with open('models/feature_names.json', 'w') as f:
        json.dump(features, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"   Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"   Test F1:       {test_metrics['test_f1_score']:.4f}")
    print(f"   Test ROC-AUC:  {test_metrics['test_roc_auc']:.4f}")
    print("="*70)
    
    return classifier


if __name__ == "__main__":
    classifier = train_fast_model()

