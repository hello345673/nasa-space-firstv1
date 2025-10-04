"""
Test HD 209458b WITHOUT time-based features
"""
import joblib
import json
import numpy as np

model = joblib.load('models/exoplanet_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

with open('models/feature_names.json', 'r') as f:
    feature_names = json.load(f)

# HD 209458b - ONLY physically meaningful features
hd209458b_data = {
    'st_rad': 1.20,           # Star Radius
    'dec': 18.884,            # Sky position
    'st_logg': 4.45,          # Star gravity
    'koi_slogg': 4.45,        
    'koi_impact': 0.50,       # Impact parameter
    'pl_orbper': 3.5247486,   # Orbital period - KEY!
    'koi_period': 3.5247486,
    'koi_duration': 3.1,      # Transit duration - KEY!
    'pl_eqt': 1448,           # Planet temp - KEY!
    'koi_teq': 1448,
    'koi_prad': 15.1,         # Planet radius - KEY!
    'pl_rade': 15.1,
    'koi_steff': 6026,        # Star temp
    'st_teff': 6026,
    'ra': 330.793,            # Sky position
    'pl_trandep': 14600,      # Transit depth - KEY!
    'koi_depth': 14600,
    'st_tmag': 7.65,          # Star brightness
    'koi_srad': 1.20,
    'pl_insol': 600,          # Insolation
    'koi_insol': 600,
}

print("="*70)
print("HD 209458b TEST (Without problematic time features)")
print("="*70)

test_data = np.zeros((1, len(feature_names)))

for i, feature in enumerate(feature_names):
    if feature in hd209458b_data:
        test_data[0, i] = hd209458b_data[feature]

# Set time features to median (not out-of-range values)
for i, feature in enumerate(feature_names):
    if 'time' in feature.lower() or 'tranmid' in feature:
        test_data[0, i] = 0  # Will be filled with median by preprocessor
        
test_scaled = preprocessor.transform_features(test_data)
prediction = model.predict(test_scaled)[0]
probability = model.predict_proba(test_scaled)[0]

print(f"\n{'='*70}")
if prediction == 1:
    print("PREDICTION: ✅ EXOPLANET")
else:
    print("PREDICTION: ❌ NOT AN EXOPLANET")
print(f"{'='*70}")

print(f"\nExoplanet Probability:     {probability[1]*100:.2f}%")
print(f"Non-Exoplanet Probability: {probability[0]*100:.2f}%")

if prediction == 1:
    print("\n✅ SUCCESS!")
else:
    print("\n❌ Still failed - deeper model issues")

