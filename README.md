# Exoplanet Detection System

A machine learning-powered web application for classifying planetary candidates using NASA mission data from Kepler, K2, and TESS missions.

## Overview

This project implements an ensemble machine learning model to automatically detect and classify exoplanets from transit photometry data. The system combines multiple algorithms to achieve high accuracy in distinguishing between confirmed exoplanets and false positives.

## Features

- **Advanced Ensemble Model**: Combines Random Forest, XGBoost, LightGBM, Gradient Boosting, and ExtraTrees classifiers
- **High Accuracy**: Achieves 81.96% test accuracy with 89.92% ROC-AUC score
- **Interactive Web Interface**: User-friendly Flask web application for real-time predictions
- **NASA Data Integration**: Trained on 21,000+ samples from Kepler, K2, and TESS missions
- **Real-time Classification**: Input stellar and planetary parameters for instant predictions
- **Professional UI**: Modern, responsive design optimized for hackathon presentation

## Model Performance

### Test Set Results
- **Accuracy**: 81.96%
- **Precision**: 79.23%
- **Recall**: 85.59%
- **F1 Score**: 82.29%
- **ROC-AUC**: 89.92%

### Validation Results
- **Accuracy**: 80.95%
- **Precision**: 77.33%
- **Recall**: 86.43%
- **F1 Score**: 81.63%
- **ROC-AUC**: 88.18%

## Technical Architecture

### Machine Learning Stack
- **scikit-learn**: Core ML algorithms and preprocessing
- **XGBoost**: Extreme gradient boosting
- **LightGBM**: Microsoft's gradient boosting framework
- **imblearn**: SMOTE for class balancing
- **pandas/numpy**: Data manipulation and numerical computing

### Web Application Stack
- **Flask**: Python web framework
- **HTML5/CSS3**: Frontend markup and styling
- **JavaScript**: Client-side interactivity
- **Font Awesome**: Icon framework

### Data Processing
- **Feature Engineering**: 12 core parameters for classification
- **Data Harmonization**: Standardized column names across missions
- **Missing Value Handling**: Median imputation for robust predictions
- **Feature Scaling**: StandardScaler for optimal model performance

## Dataset Information

### Data Sources
- **Kepler Mission**: Primary exoplanet discovery data
- **K2 Mission**: Extended Kepler mission data
- **TESS Mission**: Transiting Exoplanet Survey Satellite data

### Features Used
1. **Orbital Period** (pl_orbper): Planet's orbital period in days
2. **Transit Duration** (pl_trandur): Transit duration in hours
3. **Transit Depth** (pl_trandep): Transit depth in parts per million
4. **Planet Radius** (pl_rade): Planet radius in Earth radii
5. **Planet Temperature** (pl_eqt): Equilibrium temperature in Kelvin
6. **Insolation Flux** (pl_insol): Stellar flux relative to Earth
7. **Impact Parameter** (pl_imppar): Transit impact parameter
8. **Star Temperature** (st_teff): Stellar effective temperature
9. **Star Radius** (st_rad): Stellar radius in solar radii
10. **Star Surface Gravity** (st_logg): Stellar surface gravity
11. **Right Ascension** (ra): Celestial longitude
12. **Declination** (dec): Celestial latitude

## Installation

### Prerequisites
- Python 3.9+
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd exoplanet-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if not already trained)
   ```bash
   python model_training_fast.py
   ```

4. **Start the web application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Open your browser to `http://localhost:8001`
   - Navigate to the Predict page to test the model

## Usage

### Web Interface
1. Navigate to the Predict page
2. Enter known stellar and planetary parameters
3. Leave unknown parameters blank (will use median values)
4. Click "Classify" to get prediction results
5. View confidence scores and probability breakdown

### Standalone Prediction
```python
python predict.py
```
Edit the `planet_data` dictionary in `predict.py` with your parameters.

## Project Structure

```
exoplanet-detection/
├── app.py                          # Flask web application
├── model_training_fast.py          # Model training script
├── data_preprocessing.py           # Data preprocessing pipeline
├── predict.py                      # Standalone prediction script
├── feature_labels.py               # Feature descriptions
├── requirements.txt                # Python dependencies
├── models/                         # Trained model files
│   ├── exoplanet_model.pkl        # Ensemble model
│   ├── preprocessor.pkl           # Data preprocessor
│   ├── feature_names.json         # Feature list
│   └── metrics.json               # Performance metrics
├── templates/                      # HTML templates
│   ├── base.html                  # Base template
│   ├── index.html                 # Home page
│   └── predict.html               # Prediction page
├── static/                        # Static assets
│   ├── css/style.css              # Stylesheet
│   ├── js/main.js                 # Main JavaScript
│   ├── js/predict.js              # Prediction JavaScript
│   └── images/hero-bg.jpg         # Background image
└── README.md                      # This file
```

## Model Training Details

### Ensemble Configuration
- **Random Forest**: 1600 estimators, max_depth=20
- **XGBoost**: 1600 estimators, max_depth=10, learning_rate=0.1
- **LightGBM**: 1600 estimators, max_depth=10, learning_rate=0.1
- **Gradient Boosting**: 1600 estimators, max_depth=10, learning_rate=0.1
- **ExtraTrees**: 1600 estimators, max_depth=20
- **Voting**: Soft voting for probability-based predictions

### Training Process
1. **Data Loading**: Load and harmonize Kepler, K2, and TESS datasets
2. **Preprocessing**: Standardize features and handle missing values
3. **Feature Selection**: Select 12 core parameters for classification
4. **Class Balancing**: Apply SMOTE to handle class imbalance
5. **Cross-Validation**: 5-fold stratified cross-validation
6. **Ensemble Training**: Train all models with optimized hyperparameters
7. **Model Evaluation**: Comprehensive performance assessment

## Performance Analysis

### Strengths
- High recall (85.59%) ensures most exoplanets are detected
- Strong ROC-AUC (89.92%) indicates excellent discrimination ability
- Ensemble approach provides robust predictions
- Handles missing data gracefully with imputation

### Limitations
- Requires specific stellar and planetary parameters
- Performance may vary for unusual planetary systems
- Training data limited to transit-detected planets
- Model accuracy depends on data quality

## Future Improvements

- Integration of additional NASA mission data
- Real-time data pipeline for continuous learning
- Advanced feature engineering with domain expertise
- Ensemble model optimization with hyperparameter tuning
- Mobile-responsive design enhancements

## Contributing

This project was developed for the NASA Space Apps Challenge 2025. Contributions and improvements are welcome.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- NASA for providing open-source exoplanet datasets
- Kepler, K2, and TESS mission teams for data collection
- Scikit-learn, XGBoost, and LightGBM communities for ML frameworks
- Flask and web development communities for application framework

## Contact

For questions or collaboration opportunities, please contact the development team.
