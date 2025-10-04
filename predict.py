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
    'pl_orbper': 14.30,        # Orbital Period (days) - relatively few transits in short missions
    'pl_trandur': 2.3,         # Transit Duration (hours) - fairly short duration
    'pl_trandep': 85,          # Transit Depth (ppm) - very shallow (≈ 1.2 R_earth / 1.2 R_sun)^2
    'pl_rade': 1.20,           # Planet Radius (Earth radii) - small planet
    'pl_eqt': 560,             # Planet Temperature (Kelvin) - warm temperate-ish
    'pl_insol': 18.5,          # Insolation Flux (Earth flux) - modest
    'pl_imppar': 0.92,         # Impact Parameter (0-1) - very grazing transit (near-limb) → V-shaped, shallow
    
    # Star Parameters
    'st_teff': 6100,           # Star Temperature (Kelvin) - slightly hotter than Sun
    'st_rad': 1.20,            # Star Radius (Solar radii) - larger star reduces depth
    'st_logg': 4.30,           # Star Surface Gravity (log g)
    
    # Position
    'ra': 187.321,             # Right Ascension (degrees)
    'dec': 11.452              # Declination (degrees)
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

