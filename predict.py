"""
Simple Prediction Script - Test Your Exoplanet Model
Just edit the values below and run: python3 predict.py
"""

import joblib
import numpy as np
import json

# ============================================================================
# EDIT THESE VALUES TO TEST DIFFERENT PLANETS
# ============================================================================

# Test with HD 209458b (confirmed exoplanet - "Osiris")
planet_data = {
    # HD 209458b Parameters (Real Exoplanet)
    'pl_orbper': 3.52,         # days — known orbital period
    'pl_trandur': 3.0,         # hours — typical transit duration
    'pl_trandep': 150,         # ppm — reasonable depth for hot Jupiter
    'pl_rade': 15.1,           # Earth radii — hot Jupiter size
    'pl_eqt': 1450,            # K — hot Jupiter temperature
    'pl_insol': 1000,          # Earth flux — high insolation
    'pl_imppar': 0.5,          # moderate impact parameter
    
    # Star Parameters (HD 209458)
    'st_teff': 6071,           # K — Sun-like star
    'st_rad': 1.2,             # R_sun — slightly larger than Sun
    'st_logg': 4.3,            # Star Surface Gravity (log g)
    
    # Position
    'ra': 22.0,                # Right Ascension (degrees)
    'dec': 18.9                # Declination (degrees)
}

# ============================================================================
# MODEL PREDICTION (Don't edit below this line)
# ============================================================================

def predict_exoplanet(data):
    """Make prediction using the trained model"""
    
    # Load model and preprocessor
    print("Loading model...")
    model = joblib.load('models/exoplanet_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
    
    # Load feature names
    with open('models/feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    print(f"Model uses {len(feature_names)} features\n")
    
    # Create feature array in correct order
    X = []
    print("Input Features:")
    print("-" * 70)
    for feature in feature_names:
        value = data.get(feature, np.nan)
        X.append(value)
        if np.isnan(value):
            print(f"  {feature:20s} = Not provided (will use median)")
        else:
            print(f"  {feature:20s} = {value}")
    
    X = np.array(X).reshape(1, -1)
    
    # Preprocess and predict
    X_scaled = preprocessor.transform_features(X)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    if prediction == 1:
        print(f"\n🌍 EXOPLANET")
        print(f"   Confidence: {probability[1]*100:.2f}%")
    else:
        print(f"\n❌ NOT AN EXOPLANET")
        print(f"   Confidence: {probability[0]*100:.2f}%")
    
    print(f"\n📊 Probabilities:")
    print(f"   Exoplanet:     {probability[1]*100:.2f}%")
    print(f"   Non-Exoplanet: {probability[0]*100:.2f}%")
    
    print("\n" + "="*70)
    
    return prediction, probability


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXOPLANET PREDICTION")
    print("="*70 + "\n")
    
    try:
        prediction, probability = predict_exoplanet(planet_data)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nMake sure:")
        print("  1. Model is trained (models/exoplanet_model.pkl exists)")
        print("  2. You're in the correct directory")
        print("  3. All dependencies are installed")

