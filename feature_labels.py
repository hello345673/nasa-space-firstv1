"""
User-friendly labels for exoplanet features
Maps technical column names to readable labels with units
"""

FEATURE_LABELS = {
    # Planet Orbital Parameters
    'pl_orbper': 'Orbital Period (days)',
    'koi_period': 'Orbital Period (days)',
    
    # Transit Timing
    'pl_tranmid': 'Transit Midpoint Time (BJD)',
    'koi_time0bk': 'Transit Epoch (BJD)',
    
    # Transit Characteristics
    'pl_trandur': 'Transit Duration (hours)',
    'koi_duration': 'Transit Duration (hours)',
    'pl_trandep': 'Transit Depth (ppm)',
    'koi_depth': 'Transit Depth (ppm)',
    'pl_imppar': 'Impact Parameter (0-1)',
    'koi_impact': 'Impact Parameter (0-1)',
    
    # Planet Size
    'pl_rade': 'Planet Radius (Earth radii)',
    'koi_prad': 'Planet Radius (Earth radii)',
    'pl_radj': 'Planet Radius (Jupiter radii)',
    
    # Planet Temperature & Flux
    'pl_eqt': 'Planet Temperature (Kelvin)',
    'koi_teq': 'Planet Temperature (Kelvin)',
    'pl_insol': 'Insolation Flux (Earth flux)',
    'koi_insol': 'Insolation Flux (Earth flux)',
    
    # Signal Quality
    'pl_snr': 'Signal-to-Noise Ratio',
    'koi_model_snr': 'Signal-to-Noise Ratio',
    
    # Stellar Properties
    'st_teff': 'Star Temperature (Kelvin)',
    'koi_steff': 'Star Temperature (Kelvin)',
    'st_logg': 'Star Surface Gravity (log g)',
    'koi_slogg': 'Star Surface Gravity (log g)',
    'st_rad': 'Star Radius (Solar radii)',
    'koi_srad': 'Star Radius (Solar radii)',
    'st_mass': 'Star Mass (Solar masses)',
    'st_met': 'Star Metallicity [Fe/H]',
    'st_mag': 'Star Brightness (magnitude)',
    'koi_kepmag': 'Kepler Magnitude',
    'st_tmag': 'TESS Magnitude',
    'st_dist': 'Distance to Star (parsecs)',
    'st_pmra': 'Star Proper Motion RA (mas/yr)',
    'st_pmdec': 'Star Proper Motion Dec (mas/yr)',
    
    # Orbital Parameters
    'pl_orbsmax': 'Semi-Major Axis (AU)',
    'pl_orbeccen': 'Orbital Eccentricity (0-1)',
    'pl_bmasse': 'Planet Mass (Earth masses)',
    'pl_bmassj': 'Planet Mass (Jupiter masses)',
    
    # False Positive Flags
    'koi_fpflag_nt': 'Not Transit-Like Flag (0/1)',
    'koi_fpflag_ss': 'Stellar Eclipse Flag (0/1)',
    'koi_fpflag_co': 'Centroid Offset Flag (0/1)',
    'koi_fpflag_ec': 'Ephemeris Match Flag (0/1)',
    
    # Other
    'ttv_flag': 'Transit Timing Variation (0/1)',
    'ra': 'Right Ascension (degrees)',
    'dec': 'Declination (degrees)'
}

FEATURE_DESCRIPTIONS = {
    'pl_orbper': 'How long the planet takes to orbit its star',
    'pl_trandur': 'How long the transit lasts when planet crosses star',
    'pl_trandep': 'How much the star dims during transit',
    'pl_rade': 'Size of planet compared to Earth (1.0 = Earth-sized)',
    'pl_eqt': 'Expected surface temperature of the planet',
    'st_teff': 'Surface temperature of the host star',
    'st_rad': 'Size of star compared to Sun (1.0 = Sun-sized)',
    'ra': 'Position in sky (longitude)',
    'dec': 'Position in sky (latitude)'
}

def get_label(feature_name):
    """Get user-friendly label for a feature"""
    return FEATURE_LABELS.get(feature_name, feature_name)

def get_description(feature_name):
    """Get description for a feature"""
    return FEATURE_DESCRIPTIONS.get(feature_name, '')

