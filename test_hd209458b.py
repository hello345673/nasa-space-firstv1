"""
Test model with HD 209458b - Famous Hot Jupiter Exoplanet
This is a CONFIRMED exoplanet, so the model should predict it as an exoplanet!
"""
import joblib
import json
import numpy as np

# Load model and preprocessor
print("="*70)
print("TESTING MODEL WITH HD 209458b (Famous Hot Jupiter)")
print("="*70)

model = joblib.load('models/exoplanet_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Load feature names
with open('models/feature_names.json', 'r') as f:
    feature_names = json.load(f)

print(f"\nModel uses {len(feature_names)} features:")
print(f"Features: {feature_names}")

# HD 209458b parameters (from user input)
hd209458b_data = {
    'st_rad': 1.20,                    # Star Radius (Solar radii)
    'koi_time0bk': 2452826.82625,      # Transit Epoch (BJD)
    'pl_insol': 600,                   # Insolation Flux (Earth flux)
    'koi_insol': 600,                  # Insolation Flux (duplicate name)
    'dec': 18.884,                     # Declination (degrees)
    'st_logg': 4.45,                   # Star Surface Gravity (log g)
    'koi_slogg': 4.45,                 # Star Surface Gravity (duplicate)
    'pl_tranmid': 2452826.82625,       # Transit Midpoint Time (BJD)
    'pl_eqt': 1448,                    # Planet Temperature (Kelvin)
    'koi_teq': 1448,                   # Planet Temperature (duplicate)
    'pl_imppar': 0.50,                 # Impact Parameter (0-1)
    'koi_impact': 0.50,                # Impact Parameter (duplicate)
    'pl_orbper': 3.5247486,            # Orbital Period (days)
    'koi_period': 3.5247486,           # Orbital Period (duplicate)
    'pl_trandur': 3.1,                 # Transit Duration (hours)
    'koi_duration': 3.1,               # Transit Duration (duplicate)
    'pl_rade': 15.1,                   # Planet Radius (Earth radii)
    'koi_prad': 15.1,                  # Planet Radius (duplicate)
    'st_teff': 6026,                   # Star Temperature (Kelvin)
    'koi_steff': 6026,                 # Star Temperature (duplicate)
    'ra': 330.793,                     # Right Ascension (degrees)
    'pl_trandep': 14600,               # Transit Depth (ppm)
    'koi_depth': 14600,                # Transit Depth (duplicate)
    'st_tmag': 7.65,                   # TESS Magnitude
    'koi_srad': 1.20,                  # Star Radius (duplicate)
}

print("\n" + "="*70)
print("HD 209458b PARAMETERS:")
print("="*70)
print(f"Planet Type: Hot Jupiter (CONFIRMED EXOPLANET)")
print(f"Orbital Period: {hd209458b_data['pl_orbper']} days")
print(f"Planet Radius: {hd209458b_data['pl_rade']} Earth radii (1.35 Jupiter radii)")
print(f"Planet Temp: {hd209458b_data['pl_eqt']} K (Very hot!)")
print(f"Star Type: G0V (Sun-like)")
print(f"Star Temp: {hd209458b_data['st_teff']} K")
print(f"Transit Depth: {hd209458b_data['pl_trandep']} ppm (1.46% dip)")

# Create feature vector
test_data = np.zeros((1, len(feature_names)))

print("\n" + "="*70)
print("MAPPING PARAMETERS TO MODEL FEATURES:")
print("="*70)

filled_count = 0
for i, feature in enumerate(feature_names):
    if feature in hd209458b_data:
        test_data[0, i] = hd209458b_data[feature]
        print(f"✅ {feature:20s} = {hd209458b_data[feature]}")
        filled_count += 1
    else:
        print(f"❌ {feature:20s} = (will use median)")

print(f"\nFilled {filled_count}/{len(feature_names)} features")

# Preprocess and predict
print("\n" + "="*70)
print("MAKING PREDICTION...")
print("="*70)

test_scaled = preprocessor.transform_features(test_data)
prediction = model.predict(test_scaled)[0]
probability = model.predict_proba(test_scaled)[0]

print("\n" + "🔮 " + "="*66)
if prediction == 1:
    print("PREDICTION: ✅ EXOPLANET")
else:
    print("PREDICTION: ❌ NOT AN EXOPLANET")
print("="*70)

print(f"\n📊 CONFIDENCE SCORES:")
print(f"   Exoplanet Probability:     {probability[1]*100:.2f}%")
print(f"   Non-Exoplanet Probability: {probability[0]*100:.2f}%")

print("\n" + "="*70)
print("EXPECTED RESULT: EXOPLANET (This is HD 209458b, a confirmed exoplanet!)")
print("="*70)

if prediction == 1:
    print("✅ SUCCESS! Model correctly identified this as an exoplanet!")
else:
    print("❌ ERROR! Model failed to identify this known exoplanet.")
    print("   This might be due to missing features or data preprocessing issues.")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)

