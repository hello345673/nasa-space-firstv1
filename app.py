"""
Flask Web Application for Exoplanet Detection
Interactive interface for researchers and enthusiasts
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import plotly
import plotly.graph_objs as go
import plotly.express as px
from feature_labels import get_label, get_description
from data_preprocessing_fixed import FixedExoplanetDataPreprocessor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model and preprocessor
model = None
preprocessor = None
feature_names = []
metrics = {}


def load_model_and_data():
    """Load the trained model, preprocessor, and metadata"""
    global model, preprocessor, feature_names, metrics
    
    try:
        model = joblib.load('models/exoplanet_model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        
        with open('models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        with open('models/metrics.json', 'r') as f:
            metrics = json.load(f)
        
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page - single entry or batch upload"""
    if request.method == 'GET':
        # Create features with labels
        features_with_labels = [
            {'name': f, 'label': get_label(f), 'description': get_description(f)}
            for f in feature_names
        ]
        return render_template('predict.html', features=features_with_labels)
    
    try:
        # Check if it's a file upload or form data
        if 'file' in request.files and request.files['file'].filename:
            # Batch prediction from CSV
            file = request.files['file']
            df = pd.read_csv(file)
            
            # Extract features (fill missing with median)
            X = []
            for feature in feature_names:
                if feature in df.columns:
                    X.append(df[feature].fillna(df[feature].median()).values)
                else:
                    X.append(np.zeros(len(df)))
            
            X = np.array(X).T
            
            # Preprocess and predict
            X_scaled = preprocessor.transform_features(X)
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)
            
            # Create results dataframe
            results = []
            for i in range(len(predictions)):
                results.append({
                    'index': i,
                    'prediction': 'Exoplanet' if predictions[i] == 1 else 'Non-Exoplanet',
                    'confidence': float(probabilities[i][predictions[i]]) * 100
                })
            
            return jsonify({
                'success': True,
                'batch': True,
                'results': results,
                'total': len(results),
                'exoplanets': int(np.sum(predictions))
            })
        
        else:
            # Single prediction from form
            features_dict = request.form.to_dict()
            
            # Create feature vector - handle empty/missing values
            X = []
            for feature in feature_names:
                value = features_dict.get(feature, '')
                try:
                    # If empty or invalid, use NaN (will be imputed by preprocessor)
                    if value == '' or value is None:
                        X.append(np.nan)
                    else:
                        X.append(float(value))
                except:
                    X.append(np.nan)
            
            X = np.array(X).reshape(1, -1)
            
            # Preprocess and predict
            X_scaled = preprocessor.transform_features(X)
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0]
            
            result = {
                'prediction': 'Exoplanet' if prediction == 1 else 'Non-Exoplanet',
                'confidence': float(probability[prediction]) * 100,
                'probability_exoplanet': float(probability[1]) * 100,
                'probability_non_exoplanet': float(probability[0]) * 100
            }
            
            return jsonify({'success': True, 'result': result})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/statistics')
def statistics():
    """Model statistics and performance dashboard"""
    
    # Create performance metrics visualization
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Validation': [
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0),
            metrics.get('roc_auc', 0)
        ],
        'Test': [
            metrics.get('test_accuracy', 0),
            metrics.get('test_precision', 0),
            metrics.get('test_recall', 0),
            metrics.get('test_f1_score', 0),
            metrics.get('test_roc_auc', 0)
        ]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(name='Validation', x=df_metrics['Metric'], y=df_metrics['Validation']),
        go.Bar(name='Test', x=df_metrics['Metric'], y=df_metrics['Test'])
    ])
    
    fig.update_layout(
        title='Model Performance Metrics',
        xaxis_title='Metric',
        yaxis_title='Score',
        barmode='group',
        template='plotly_dark',
        height=500
    )
    
    metrics_plot = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Feature importance visualization (if available)
    feature_importance_plot = None
    if hasattr(preprocessor, 'feature_importance') and preprocessor.feature_importance is not None:
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': preprocessor.feature_importance
        }).sort_values('Importance', ascending=False).head(15)
        
        fig2 = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                      title='Top 15 Feature Importances',
                      template='plotly_dark')
        feature_importance_plot = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('statistics.html',
                         metrics=metrics,
                         metrics_plot=metrics_plot,
                         feature_importance_plot=feature_importance_plot)


@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')


@app.route('/api/model-info')
def model_info():
    """API endpoint for model information"""
    return jsonify({
        'model_loaded': model is not None,
        'features_count': len(feature_names),
        'features': feature_names,
        'metrics': metrics,
        'model_type': 'Ensemble (Random Forest + XGBoost + LightGBM + Gradient Boosting)'
    })


if __name__ == '__main__':
    print("Loading model...")
    if load_model_and_data():
        print("Starting Flask application...")
        print("Navigate to http://localhost:8000 in your browser")
        app.run(debug=True, host='0.0.0.0', port=8000)
    else:
        print("Error: Could not load model. Please train the model first using model_training.py")

