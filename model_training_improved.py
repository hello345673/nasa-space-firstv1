"""
IMPROVED Machine Learning Model Training for Exoplanet Detection
Based on research paper: "Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification"
Improvements: ExtraTrees, 1600 estimators, 10-fold cross-validation
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
from data_preprocessing import ExoplanetDataPreprocessor
import warnings
import time
warnings.filterwarnings('ignore')


class ImprovedExoplanetClassifier:
    """Enhanced ensemble classifier with ExtraTrees and optimized hyperparameters"""
    
    def __init__(self, model_type='enhanced_ensemble'):
        self.model_type = model_type
        self.model = None
        self.preprocessor = ExoplanetDataPreprocessor()
        self.metrics = {}
        self.cv_scores = {}
        self.feature_importance = None
        
    def create_model(self):
        """Create enhanced ensemble model with 5 classifiers and optimized hyperparameters"""
        print("\n🚀 Creating Enhanced Ensemble Model...")
        print("   Based on research paper best practices")
        
        # Random Forest with optimized hyperparameters (from paper: 1600 estimators)
        rf_model = RandomForestClassifier(
            n_estimators=1600,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            criterion='entropy',  # Paper found entropy better than gini
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        # XGBoost with increased estimators
        xgb_model = XGBClassifier(
            n_estimators=1600,
            max_depth=10,
            learning_rate=0.1,  # Paper's optimal value
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        
        # LightGBM with increased estimators
        lgbm_model = LGBMClassifier(
            n_estimators=1600,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        
        # Gradient Boosting with paper's optimal hyperparameters
        gb_model = GradientBoostingClassifier(
            n_estimators=1600,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
        
        # ExtraTrees - NEW! Paper showed 82.36% accuracy
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
        
        # Create voting ensemble with 5 models
        self.model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('xgb', xgb_model),
                ('lgbm', lgbm_model),
                ('gb', gb_model),
                ('extra_trees', extra_trees_model)  # NEW!
            ],
            voting='soft',
            n_jobs=-1,
            verbose=False
        )
        
        print("✅ Enhanced Ensemble Created:")
        print("   • Random Forest (1600 estimators, entropy)")
        print("   • XGBoost (1600 estimators)")
        print("   • LightGBM (1600 estimators)")
        print("   • Gradient Boosting (1600 estimators)")
        print("   • ExtraTrees (1600 estimators) [NEW!]")
        
        return self.model
    
    def perform_cross_validation(self, X, y, n_folds=10, n_repeats=5):
        """
        Perform cross-validation as described in the research paper
        10-fold cross-validation repeated 5 times
        """
        print(f"\n🔄 Performing {n_folds}-Fold Cross-Validation (repeated {n_repeats} times)...")
        print("   This follows the paper's methodology for robust evaluation")
        
        all_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        for repeat in range(n_repeats):
            print(f"\n   Repeat {repeat + 1}/{n_repeats}...")
            
            # Stratified K-Fold to maintain class distribution
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42 + repeat)
            
            fold_scores = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'roc_auc': []
            }
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
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
                fold_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
                fold_scores['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))
                fold_scores['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
                fold_scores['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))
                fold_scores['roc_auc'].append(roc_auc_score(y_val_fold, y_pred_proba))
                
                print(f"      Fold {fold:2d}: Acc={fold_scores['accuracy'][-1]:.4f}, "
                      f"F1={fold_scores['f1'][-1]:.4f}")
            
            # Add this repeat's averages to overall scores
            for metric in all_scores.keys():
                all_scores[metric].append(np.mean(fold_scores[metric]))
        
        # Calculate final averages across all repeats
        self.cv_scores = {
            metric: {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
            for metric, scores in all_scores.items()
        }
        
        print("\n✅ Cross-Validation Results (averaged across all folds and repeats):")
        for metric, stats in self.cv_scores.items():
            print(f"   {metric:10s}: {stats['mean']:.4f} (±{stats['std']:.4f}) "
                  f"[{stats['min']:.4f} - {stats['max']:.4f}]")
        
        return self.cv_scores
    
    def train(self, X_train, y_train, X_val, y_val, use_smote=True):
        """Train the model with optional SMOTE for class imbalance"""
        print(f"\n🎯 Training Enhanced Ensemble Model...")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Validation samples: {len(X_val):,}")
        
        start_time = time.time()
        
        # Handle class imbalance with SMOTE
        if use_smote:
            print("   Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"   After SMOTE: {len(X_train_balanced):,} samples")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Train model
        print("   Training in progress (this may take several minutes with 1600 estimators)...")
        self.model.fit(X_train_balanced, y_train_balanced)
        
        training_time = time.time() - start_time
        print(f"   ✅ Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        
        # Evaluate on validation set
        print("\n📊 Evaluating model on validation set...")
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
        
        print("\n✅ Validation Metrics:")
        for metric, value in self.metrics.items():
            if metric != 'training_time_seconds':
                print(f"   {metric:12s}: {value:.4f} ({value*100:.2f}%)")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_val, y_pred, target_names=['Non-Exoplanet', 'Exoplanet']))
        
        print("\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_val, y_pred)
        print(cm)
        print(f"   True Negatives:  {cm[0,0]:,}")
        print(f"   False Positives: {cm[0,1]:,}")
        print(f"   False Negatives: {cm[1,0]:,}")
        print(f"   True Positives:  {cm[1,1]:,}")
        
        # Feature importance
        self._calculate_feature_importance()
        
        return self.metrics
    
    def _calculate_feature_importance(self):
        """Calculate feature importance from the ensemble"""
        try:
            importances = []
            for name, estimator in self.model.named_estimators_.items():
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
            
            if importances:
                self.feature_importance = np.mean(importances, axis=0)
                print(f"   ✅ Feature importance calculated from {len(importances)} models")
        except Exception as e:
            print(f"   ⚠️  Could not calculate feature importance: {e}")
    
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
        
        # Combine metrics and CV scores
        save_data = {
            'metrics': self.metrics,
            'cv_scores': self.cv_scores
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n💾 Model saved to {model_path}")
        print(f"💾 Preprocessor saved to {preprocessor_path}")
        print(f"💾 Metrics saved to {metrics_path}")


def train_improved_exoplanet_model():
    """Main training pipeline with improvements from research paper"""
    print("="*70)
    print("🌟 IMPROVED EXOPLANET DETECTION MODEL TRAINING 🌟")
    print("="*70)
    print("\n📚 Improvements based on research paper:")
    print("   1. ExtraTrees added to ensemble (5 models total)")
    print("   2. Increased estimators from 200 to 1600")
    print("   3. 10-fold cross-validation repeated 5 times")
    print("   4. Optimized hyperparameters (entropy, learning_rate=0.1)")
    print("="*70)
    
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
    
    # Split data (70/30 as in the paper)
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
    classifier = ImprovedExoplanetClassifier(model_type='enhanced_ensemble')
    classifier.preprocessor = preprocessor
    classifier.create_model()
    
    # Perform cross-validation (paper's methodology)
    cv_scores = classifier.perform_cross_validation(X_train_scaled, y_train, n_folds=10, n_repeats=5)
    
    # Train on full training set
    print("\n" + "="*70)
    print("🎯 TRAINING ON FULL TRAINING SET")
    print("="*70)
    metrics = classifier.train(X_train_scaled, y_train, X_val_scaled, y_val, use_smote=True)
    
    # Final test evaluation
    print("\n" + "="*70)
    print("🏆 FINAL TEST SET EVALUATION")
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
    
    print("\n✅ Test Set Metrics:")
    for metric, value in test_metrics.items():
        print(f"   {metric:18s}: {value:.4f} ({value*100:.2f}%)")
    
    print("\n📋 Test Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Non-Exoplanet', 'Exoplanet']))
    
    print("\n📊 Test Confusion Matrix:")
    cm_test = confusion_matrix(y_test, y_test_pred)
    print(cm_test)
    
    # Update metrics with test results
    classifier.metrics.update(test_metrics)
    
    # Save model
    classifier.save_model()
    
    # Save feature names
    with open('models/feature_names.json', 'w') as f:
        json.dump(features, f, indent=2)
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE! 🎉")
    print("="*70)
    print("\n📈 Summary:")
    print(f"   Cross-Validation Accuracy: {cv_scores['accuracy']['mean']:.4f} (±{cv_scores['accuracy']['std']:.4f})")
    print(f"   Test Set Accuracy:         {test_metrics['test_accuracy']:.4f}")
    print(f"   Test Set F1 Score:         {test_metrics['test_f1_score']:.4f}")
    print(f"   Test Set ROC AUC:          {test_metrics['test_roc_auc']:.4f}")
    print("\n🏆 Model ready for NASA Space Apps Challenge submission!")
    print("="*70)
    
    return classifier


if __name__ == "__main__":
    classifier = train_improved_exoplanet_model()

