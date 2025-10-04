"""
Data Preprocessing Module for Exoplanet Detection
Handles Kepler, K2, and TESS datasets
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class ExoplanetDataPreprocessor:
    """Preprocessor for exoplanet datasets from Kepler, K2, and TESS missions"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        
    def load_kepler_data(self, filepath):
        """Load and preprocess Kepler KOI dataset"""
        print("Loading Kepler dataset...")
        df = pd.read_csv(filepath, comment='#')
        
        # Target column: 'koi_disposition' (CONFIRMED, FALSE POSITIVE, CANDIDATE)
        if 'koi_disposition' in df.columns:
            # Map to binary: 1 for CONFIRMED, 0 for FALSE POSITIVE/CANDIDATE
            df['label'] = df['koi_disposition'].apply(
                lambda x: 1 if x == 'CONFIRMED' else 0
            )
        
        # Select relevant features
        feature_cols = [
            'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
            'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol',
            'koi_model_snr', 'koi_steff', 'koi_slogg', 'koi_srad',
            'ra', 'dec', 'koi_kepmag'
        ]
        
        # Filter available columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        if 'label' in df.columns:
            df_clean = df[available_features + ['label']].copy()
        else:
            df_clean = df[available_features].copy()
        
        df_clean['source'] = 'Kepler'
        print(f"Kepler data loaded: {len(df_clean)} samples")
        return df_clean
    
    def load_tess_data(self, filepath):
        """Load and preprocess TESS TOI dataset"""
        print("Loading TESS dataset...")
        df = pd.read_csv(filepath, comment='#')
        
        # Target column: 'tfopwg_disp' (PC, FP, CP, etc.)
        if 'tfopwg_disp' in df.columns:
            df['label'] = df['tfopwg_disp'].apply(
                lambda x: 1 if str(x).upper() in ['CP', 'PC'] else 0
            )
        
        # Select relevant features
        feature_cols = [
            'pl_orbper', 'pl_trandur', 'pl_tranmid', 'pl_imppar',
            'pl_trandep', 'pl_rade', 'pl_eqt', 'pl_insol',
            'st_tmag', 'st_teff', 'st_logg', 'st_rad',
            'ra', 'dec'
        ]
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        if 'label' in df.columns:
            df_clean = df[available_features + ['label']].copy()
        else:
            df_clean = df[available_features].copy()
            
        df_clean['source'] = 'TESS'
        print(f"TESS data loaded: {len(df_clean)} samples")
        return df_clean
    
    def load_k2_data(self, filepath):
        """Load and preprocess K2 dataset"""
        print("Loading K2 dataset...")
        df = pd.read_csv(filepath, comment='#')
        
        # Target column might vary, look for disposition column
        disp_cols = [col for col in df.columns if 'disp' in col.lower()]
        if disp_cols:
            target_col = disp_cols[0]
            df['label'] = df[target_col].apply(
                lambda x: 1 if 'CONFIRM' in str(x).upper() else 0
            )
        
        # Select relevant features (similar to Kepler)
        feature_cols = [
            'pl_orbper', 'pl_trandur', 'pl_tranmid', 'pl_imppar',
            'pl_trandep', 'pl_rade', 'pl_eqt', 'pl_insol',
            'st_teff', 'st_logg', 'st_rad', 'ra', 'dec'
        ]
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        if 'label' in df.columns:
            df_clean = df[available_features + ['label']].copy()
        else:
            df_clean = df[available_features].copy()
            
        df_clean['source'] = 'K2'
        print(f"K2 data loaded: {len(df_clean)} samples")
        return df_clean
    
    def harmonize_datasets(self, datasets):
        """Harmonize multiple datasets to common feature set"""
        print("\nHarmonizing datasets...")
        
        # Find common columns (excluding 'label' and 'source')
        common_cols = set(datasets[0].columns)
        for df in datasets[1:]:
            common_cols = common_cols.intersection(set(df.columns))
        
        # Remove label and source from common cols
        common_cols = [col for col in common_cols if col not in ['label', 'source']]
        
        # If few common columns, use all available and fill missing
        if len(common_cols) < 5:
            all_cols = set()
            for df in datasets:
                all_cols.update([col for col in df.columns if col not in ['label', 'source']])
            common_cols = list(all_cols)
        
        # Reindex all datasets to have same columns
        harmonized = []
        for df in datasets:
            cols_to_keep = [col for col in common_cols if col in df.columns]
            if 'label' in df.columns:
                cols_to_keep += ['label']
            if 'source' in df.columns:
                cols_to_keep += ['source']
            
            df_harm = df[cols_to_keep].copy()
            
            # Add missing columns with NaN
            for col in common_cols:
                if col not in df_harm.columns:
                    df_harm[col] = np.nan
            
            harmonized.append(df_harm)
        
        print(f"Harmonized to {len(common_cols)} common features")
        return harmonized, common_cols
    
    def preprocess_features(self, X):
        """Preprocess features: impute missing values and scale"""
        # Impute missing values
        X_imputed = self.imputer.fit_transform(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        return X_scaled
    
    def transform_features(self, X):
        """Transform new features using fitted preprocessor"""
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled
    
    def prepare_combined_dataset(self, kepler_path, tess_path, k2_path):
        """Load and combine all three datasets"""
        datasets = []
        
        try:
            kepler_df = self.load_kepler_data(kepler_path)
            datasets.append(kepler_df)
        except Exception as e:
            print(f"Warning: Could not load Kepler data: {e}")
        
        try:
            tess_df = self.load_tess_data(tess_path)
            datasets.append(tess_df)
        except Exception as e:
            print(f"Warning: Could not load TESS data: {e}")
        
        try:
            k2_df = self.load_k2_data(k2_path)
            datasets.append(k2_df)
        except Exception as e:
            print(f"Warning: Could not load K2 data: {e}")
        
        if not datasets:
            raise ValueError("No datasets could be loaded!")
        
        # Harmonize datasets
        harmonized_datasets, common_features = self.harmonize_datasets(datasets)
        
        # Combine all datasets
        combined_df = pd.concat(harmonized_datasets, ignore_index=True)
        
        # Remove rows with too many missing values
        threshold = len(common_features) * 0.5
        combined_df = combined_df.dropna(thresh=threshold)
        
        print(f"\nCombined dataset: {len(combined_df)} samples")
        print(f"Features: {len(common_features)}")
        
        if 'label' in combined_df.columns:
            print(f"Confirmed exoplanets: {combined_df['label'].sum()}")
            print(f"False positives/candidates: {len(combined_df) - combined_df['label'].sum()}")
        
        self.feature_columns = common_features
        
        return combined_df, common_features


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = ExoplanetDataPreprocessor()
    
    kepler_path = "/Users/ronit/Downloads/cumulative_2025.10.04_07.54.21.csv"
    tess_path = "/Users/ronit/Downloads/TOI_2025.10.04_07.54.31.csv"
    k2_path = "/Users/ronit/Downloads/k2pandc_2025.10.04_07.54.41.csv"
    
    combined_df, features = preprocessor.prepare_combined_dataset(
        kepler_path, tess_path, k2_path
    )
    
    print("\nDataset Info:")
    print(combined_df.info())
    print("\nFeature columns:", features)

