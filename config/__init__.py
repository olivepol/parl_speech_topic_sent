"""
Configuration module for centralized parameters.

All model hyperparameters, data paths, and mappings are defined here
to ensure reproducibility and avoid magic numbers in notebooks.

Example:
    from config import LDA_PARAMS, PARTY_ID_MAP, TOPIC_LABEL_MAP
    from config.model_params import RANDOM_SEED
"""
from .model_params import *
