"""
Test HD 209458b with the FIXED model
"""
import joblib
import json
import numpy as np

print("="*70)
print("TESTING HD 209458b (Famous Hot Jupiter Exoplanet)")
print("="*70)

# Load new model
model = joblib.load('models/exoplanet_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

with open('models/feature_names.json', 'r') as f:
    feature_names = json.load(f)

print(f"\nNew model uses {len(feature_names)} features (was 25, now should be ~20-22)")
print(f"Features: {feature_names}")

# HD 209458b parameters - ONLY physically meaningful features (no timestamps!)
hd209458b_data = {
    # Planet properties - KEY FEATURES
    'pl_orbper': 3.5247486,       # Orbital period (days)
    'pl_trandur': 3.1,            # Transit duration (hours)
    'pl_trandep': 14600,          # Transit depth (ppm) - 1.46% dip
    'pl_rade': 15.1,              # Planet radius (15.1 Earth radii = 1.35 Jupiter)
    'pl_radj': 1.35,              # Planet radius (Jupiter radii)
    'pl_eqt': 1448,               # Planet temperature (Kelvin)
    'pl_insol': 600,              # Insolation flux
    'pl_imppar': 0.50,            # Impact parameter
    
    # Star properties
    'st_teff': 6026,              # Star temperature (G0V star)
    'st_logg': 4.45,              # Star surface gravity
    'st_rad': 1.20,               # Star radius (Solar radii)
    'st_mag': 7.65,               # TESS magnitude
    'st_tmag': 7.65,              # TESS magnitude
    
    # Position
    'ra': 330.793,                # Right ascension
    'dec': 18.884,                # Declination
}

print("\n" + "="*70)
print("HD 209458b CHARACTERISTICS:")
print("="*70)
print(f"Type: Hot Jupiter (CONFIRMED EXOPLANET)")
print(f"Orbital Period: {hd209458b_data['pl_orbper']} days (3.5 days!)")
print(f"Planet Radius: {hd209458b_data['pl_rade']} Earth radii")
print(f"Planet Temp: {hd209458b_data['pl_eqt']} K (very hot!)")
print(f"Transit Depth: {hd209458b_data['pl_trandep']} ppm")
print(f"Star: G0V (Sun-like), {hd209458b_data['st_teff']} K")

# Create feature vector
test_data = np.zeros((1, len(feature_names)))

print("\n" + "="*70)
print("FILLING FEATURES:")
print("="*70)

filled = 0
for i, feature in enumerate(feature_names):
    if feature in hd209458b_data:
        test_data[0, i] = hd209458b_data[feature]
        filled += 1
        print(f"✅ {feature:20s} = {hd209458b_data[feature]}")
    else:
        # Use NaN for missing (will be imputed)
        test_data[0, i] = np.nan
        print(f"⚠️  {feature:20s} = NaN (will use median)")

print(f"\nFilled {filled}/{len(feature_names)} features")

# Preprocess and predict
print("\n" + "="*70)
print("MAKING PREDICTION...")
print("="*70)

test_scaled = preprocessor.transform_features(test_data)
prediction = model.predict(test_scaled)[0]
probability = model.predict_proba(test_scaled)[0]

print("\n" + "🔮 " + "="*66)
if prediction == 1:
    print("✅ PREDICTION: EXOPLANET")
    print("="*70)
    print(f"\n🎉 SUCCESS! Model correctly identified HD 209458b!")
else:
    print("❌ PREDICTION: NOT AN EXOPLANET")
    print("="*70)
    print(f"\n😞 Failed: Model should identify this as an exoplanet!")

print(f"\n📊 CONFIDENCE:")
print(f"   Exoplanet:     {probability[1]*100:.2f}%")
print(f"   Non-Exoplanet: {probability[0]*100:.2f}%")

print("\n" + "="*70)
if prediction == 1 and probability[1] > 0.7:
    print("✅✅✅ PERFECT! High confidence correct prediction!")
elif prediction == 1:
    print("✅ Correct but low confidence")
else:
    print("❌ Model needs more work")
print("="*70)

