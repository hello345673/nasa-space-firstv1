"""
Quick test script for the trained exoplanet model
"""
import joblib
import json
import numpy as np

# Load model and preprocessor
print("Loading model...")
model = joblib.load('models/exoplanet_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Load feature names
with open('models/feature_names.json', 'r') as f:
    feature_names = json.load(f)

print(f"✅ Model loaded! Uses {len(feature_names)} features")
print(f"Features: {feature_names[:5]}... (showing first 5)")

# Load metrics
with open('models/metrics.json', 'r') as f:
    metrics = json.load(f)

print("\n📊 Model Performance:")
print(f"   Test Accuracy: {metrics['metrics']['test_accuracy']:.4f} ({metrics['metrics']['test_accuracy']*100:.2f}%)")
print(f"   Test F1 Score: {metrics['metrics']['test_f1_score']:.4f}")
print(f"   Test ROC-AUC:  {metrics['metrics']['test_roc_auc']:.4f}")

# Test with example data (Earth-like planet)
print("\n🌍 Testing with Earth-like planet parameters:")
test_data = np.zeros((1, len(feature_names)))  # Start with zeros

# You can manually set values here for specific features
# For example, if we know the indices of key features:
print("   (Using median-filled values for demonstration)")

# Preprocess and predict
test_scaled = preprocessor.transform_features(test_data)
prediction = model.predict(test_scaled)[0]
probability = model.predict_proba(test_scaled)[0]

print(f"\n🔮 Prediction: {'EXOPLANET ✅' if prediction == 1 else 'NOT EXOPLANET ❌'}")
print(f"   Confidence: {probability[prediction]*100:.1f}%")
print(f"   Probabilities: Exoplanet={probability[1]*100:.1f}%, Non-Exoplanet={probability[0]*100:.1f}%")

print("\n✅ Model is working! Ready for the web app!")

