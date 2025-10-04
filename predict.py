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

planet_data = {
    # Planet Transit Parameters
    'pl_orbper': 9.73,         # Orbital Period (days)
    'pl_trandur': 4.1,         # Transit Duration (hours)
    'pl_trandep': 457,         # Transit Depth (ppm) - ≈ (2.1 R_earth / 0.9 R_sun)^2
    'pl_rade': 2.1,            # Planet Radius (Earth radii)
    'pl_eqt': 680,             # Planet Temperature (Kelvin)
    'pl_insol': 45.3,          # Insolation Flux (Earth flux)
    'pl_imppar': 0.36,         # Impact Parameter (0-1)
    
    # Star Parameters
    'st_teff': 5250,           # Star Temperature (Kelvin) - K-type star
    'st_rad': 0.9,             # Star Radius (Solar radii)
    'st_logg': 4.5,            # Star Surface Gravity (log g)
    
    # Position
    'ra': 150.123,             # Right Ascension (degrees)
    'dec': -22.45              # Declination (degrees)
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

