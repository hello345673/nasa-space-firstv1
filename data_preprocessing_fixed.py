"""
FIXED Data Preprocessing Module for Exoplanet Detection
Properly maps column names across Kepler, K2, and TESS datasets
NO DUPLICATE FEATURES!
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class FixedExoplanetDataPreprocessor:
    """Fixed preprocessor that properly harmonizes column names across missions"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        
        # Column name mappings: Kepler names → Standard names
        self.kepler_to_standard = {
            'koi_period': 'pl_orbper',
            'koi_time0bk': 'pl_tranmid',
            'koi_impact': 'pl_imppar',
            'koi_duration': 'pl_trandur',
            'koi_depth': 'pl_trandep',
            'koi_prad': 'pl_rade',
            'koi_teq': 'pl_eqt',
            'koi_insol': 'pl_insol',
            'koi_steff': 'st_teff',
            'koi_slogg': 'st_logg',
            'koi_srad': 'st_rad',
            'koi_kepmag': 'st_mag',
            'koi_model_snr': 'pl_snr'
        }
        
        # CORE FEATURES - Only universally available features for high confidence predictions
        # Removed mission-specific and often-missing features
        self.standard_features = [
            # CRITICAL Planet Transit Parameters (always measured)
            'pl_orbper',      # ⭐ Orbital period (days) - MUST HAVE
            'pl_trandur',     # ⭐ Transit duration (hours) - MUST HAVE
            'pl_trandep',     # ⭐ Transit depth (ppm) - MUST HAVE
            'pl_rade',        # ⭐ Planet radius (Earth radii) - MUST HAVE
            'pl_eqt',         # ⭐ Planet temperature (Kelvin) - MUST HAVE
            'pl_insol',       # Insolation flux (Earth flux) - Common
            'pl_imppar',      # Impact parameter (0-1) - Common
            
            # CRITICAL Stellar Parameters (always measured)
            'st_teff',        # ⭐ Star temperature (Kelvin) - MUST HAVE
            'st_rad',         # ⭐ Star radius (Solar radii) - MUST HAVE
            'st_logg',        # Star surface gravity (log g) - Common
            
            # Position (always available)
            'ra',             # Right ascension (degrees)
            'dec'             # Declination (degrees)
            
            # REMOVED for better generalization:
            # - koi_fpflag_* (Kepler-only, not universal)
            # - pl_snr (often missing for external data)
            # - pl_radj (redundant with pl_rade)
            # - st_mag, st_dist, st_pmra, st_pmdec (often missing)
        ]
        
    def load_kepler_data(self, filepath):
        """Load and standardize Kepler KOI dataset"""
        print("Loading Kepler dataset...")
        df = pd.read_csv(filepath, comment='#')
        
        # Target: 1 for CONFIRMED, 0 for others
        if 'koi_disposition' in df.columns:
            df['label'] = df['koi_disposition'].apply(
                lambda x: 1 if x == 'CONFIRMED' else 0
            )
        
        # Rename Kepler columns to standard names
        df_renamed = df.rename(columns=self.kepler_to_standard)
        
        # Select only columns that exist
        available_features = [col for col in self.standard_features if col in df_renamed.columns]
        
        if 'label' in df.columns:
            df_clean = df_renamed[available_features + ['label']].copy()
        else:
            df_clean = df_renamed[available_features].copy()
        
        df_clean['source'] = 'Kepler'
        print(f"   Kepler data loaded: {len(df_clean)} samples, {len(available_features)} features")
        return df_clean
    
    def load_tess_data(self, filepath):
        """Load and standardize TESS TOI dataset"""
        print("Loading TESS dataset...")
        df = pd.read_csv(filepath, comment='#')
        
        # Target: 1 for CP/PC, 0 for others
        if 'tfopwg_disp' in df.columns:
            df['label'] = df['tfopwg_disp'].apply(
                lambda x: 1 if str(x).upper() in ['CP', 'PC'] else 0
            )
        
        # TESS already uses standard pl_* and st_* naming
        # Handle TESS-specific columns
        if 'st_tmag' in df.columns:
            df['st_mag'] = df['st_tmag']
        
        # Handle duration - TESS uses pl_trandurh (hours)
        if 'pl_trandurh' in df.columns:
            df['pl_trandur'] = df['pl_trandurh']
        
        # Proper motion and distance are already in standard format (st_pmra, st_pmdec, st_dist)
        
        # Select only columns that exist
        available_features = [col for col in self.standard_features if col in df.columns]
        
        if 'label' in df.columns:
            df_clean = df[available_features + ['label']].copy()
        else:
            df_clean = df[available_features].copy()
        
        df_clean['source'] = 'TESS'
        print(f"   TESS data loaded: {len(df_clean)} samples, {len(available_features)} features")
        return df_clean
    
    def load_k2_data(self, filepath):
        """Load and standardize K2 dataset"""
        print("Loading K2 dataset...")
        df = pd.read_csv(filepath, comment='#')
        
        # Target: 1 for CONFIRMED, 0 for others
        disp_cols = [col for col in df.columns if 'disp' in col.lower()]
        if disp_cols:
            target_col = disp_cols[0]
            df['label'] = df[target_col].apply(
                lambda x: 1 if 'CONFIRM' in str(x).upper() else 0
            )
        
        # K2 already uses standard pl_* and st_* naming
        # Handle K2-specific column names
        if 'sy_kepmag' in df.columns:
            df['st_mag'] = df['sy_kepmag']
        if 'sy_kmag' in df.columns and 'st_mag' not in df.columns:
            df['st_mag'] = df['sy_kmag']
        if 'sy_dist' in df.columns:
            df['st_dist'] = df['sy_dist']
        
        # Select only columns that exist
        available_features = [col for col in self.standard_features if col in df.columns]
        
        if 'label' in df.columns:
            df_clean = df[available_features + ['label']].copy()
        else:
            df_clean = df[available_features].copy()
        
        df_clean['source'] = 'K2'
        print(f"   K2 data loaded: {len(df_clean)} samples, {len(available_features)} features")
        return df_clean
    
    def harmonize_datasets(self, datasets):
        """Combine datasets - they now all use the same column names!"""
        print("\n✅ Harmonizing datasets (all use standard column names)...")
        
        # Find columns that exist in at least one dataset
        all_features = set()
        for df in datasets:
            all_features.update([col for col in df.columns if col not in ['label', 'source']])
        
        common_features = sorted([f for f in self.standard_features if f in all_features])
        
        # Ensure all datasets have the same columns (fill missing with NaN)
        harmonized = []
        for df in datasets:
            # Add missing columns as NaN
            for feature in common_features:
                if feature not in df.columns:
                    df[feature] = np.nan
            
            # Select features in consistent order
            cols = common_features + ['label', 'source']
            cols = [c for c in cols if c in df.columns]
            harmonized.append(df[cols].copy())
        
        print(f"   ✅ Standardized to {len(common_features)} unified features (NO DUPLICATES!)")
        return harmonized, common_features
    
    def preprocess_features(self, X):
        """Preprocess features: impute missing values and scale"""
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        return X_scaled
    
    def transform_features(self, X):
        """Transform new features using fitted preprocessor"""
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled
    
    def prepare_combined_dataset(self, kepler_path, tess_path, k2_path):
        """Load and combine all three datasets with proper column mapping"""
        print("="*70)
        print("🔧 FIXED DATA PREPROCESSING - Proper Column Mapping")
        print("="*70)
        
        datasets = []
        
        try:
            kepler_df = self.load_kepler_data(kepler_path)
            datasets.append(kepler_df)
        except Exception as e:
            print(f"⚠️  Warning: Could not load Kepler data: {e}")
        
        try:
            tess_df = self.load_tess_data(tess_path)
            datasets.append(tess_df)
        except Exception as e:
            print(f"⚠️  Warning: Could not load TESS data: {e}")
        
        try:
            k2_df = self.load_k2_data(k2_path)
            datasets.append(k2_df)
        except Exception as e:
            print(f"⚠️  Warning: Could not load K2 data: {e}")
        
        if not datasets:
            raise ValueError("No datasets could be loaded!")
        
        # Harmonize datasets
        harmonized_datasets, common_features = self.harmonize_datasets(datasets)
        
        # Combine all datasets
        combined_df = pd.concat(harmonized_datasets, ignore_index=True)
        
        # Remove rows with too many missing values (>50% missing)
        threshold = len(common_features) * 0.5
        combined_df = combined_df.dropna(thresh=threshold)
        
        print(f"\n📊 Combined Dataset Summary:")
        print(f"   Total samples: {len(combined_df):,}")
        print(f"   Unified features: {len(common_features)}")
        print(f"   Features: {common_features}")
        
        if 'label' in combined_df.columns:
            print(f"\n🎯 Target Distribution:")
            print(f"   Confirmed exoplanets: {combined_df['label'].sum():,}")
            print(f"   Non-exoplanets: {(len(combined_df) - combined_df['label'].sum()):,}")
        
        # Check missing data percentage
        missing_pct = (combined_df[common_features].isna().sum() / len(combined_df) * 100).round(2)
        print(f"\n📉 Missing Data by Feature:")
        for feat, pct in missing_pct.items():
            if pct > 0:
                print(f"   {feat}: {pct}% missing")
        
        self.feature_columns = common_features
        print("="*70)
        
        return combined_df, common_features


if __name__ == "__main__":
    # Test the fixed preprocessor
    preprocessor = FixedExoplanetDataPreprocessor()
    
    kepler_path = "/Users/ronit/Downloads/cumulative_2025.10.04_07.54.21.csv"
    tess_path = "/Users/ronit/Downloads/TOI_2025.10.04_07.54.31.csv"
    k2_path = "/Users/ronit/Downloads/k2pandc_2025.10.04_07.54.41.csv"
    
    combined_df, features = preprocessor.prepare_combined_dataset(
        kepler_path, tess_path, k2_path
    )
    
    print("\n✅ SUCCESS! Dataset properly harmonized with NO DUPLICATES!")
    print(f"   Features ({len(features)}): {features}")

