"""
Prediction script for exoplanet detection
Edit the values below and run: python3 predict.py
"""

import joblib
import numpy as np
import json

# Edit these values to test different planets

# Test with GJ 1289 stellar parameters
planet_data = {
    # Planet parameters (unknown - using typical values for testing)
    'pl_orbper': 10.0,         # days - typical orbital period
    'pl_trandur': 2.0,         # hours - typical transit duration
    'pl_trandep': 50,          # ppm - typical transit depth
    'pl_rade': 1.5,             # Earth radii - Earth-like planet
    'pl_eqt': 250,             # K - habitable zone temperature
    'pl_insol': 1.0,           # Earth flux - similar to Earth
    'pl_imppar': 0.3,          # impact parameter
    
    # GJ 1289 Star Parameters (from your data)
    'st_teff': 3175,           # K - effective temperature
    'st_rad': 0.245,           # R_sun - stellar radius
    'st_logg': 4.8,            # log g - estimated for M4.5V star
    
    # Position (approximate)
    'ra': 0.0,                 # degrees - placeholder
    'dec': 0.0                 # degrees - placeholder
}

# Model prediction (don't edit below this line)

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
    # Handle the preprocessor transform method
    try:
        X_scaled = preprocessor.transform_features(X)
    except AttributeError:
        # Fallback: use transform method directly
        X_scaled = preprocessor.transform(X)
    
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    if prediction == 1:
        print(f"\nEXOPLANET")
        print(f"   Confidence: {probability[1]*100:.2f}%")
    else:
        print(f"\nNOT AN EXOPLANET")
        print(f"   Confidence: {probability[0]*100:.2f}%")
    
    print(f"\nProbabilities:")
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
        print(f"\nERROR: {e}")
        print("\nMake sure:")
        print("  1. Model is trained (models/exoplanet_model.pkl exists)")
        print("  2. You're in the correct directory")
        print("  3. All dependencies are installed")

