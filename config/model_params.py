"""
Configuration parameters for NLP models.

This module centralizes all hyperparameters and configuration settings
to ensure reproducibility and avoid magic numbers in notebooks.

Usage:
    from config.model_params import LDA_PARAMS, RANDOM_SEED
    model = LatentDirichletAllocation(**LDA_PARAMS)

Sections:
    - Text Preprocessing: Character/word limits
    - TF-IDF Vectorization: Feature extraction params
    - LDA Topic Modeling: Topic model hyperparameters
    - BERT Classification: Pre-trained model identifiers
    - Sentiment Analysis: Batch sizes, checkpoints
    - Party Mappings: factionId → party name
    - Topic Labels: topic ID → German label
"""

from typing import Dict, Any, List

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

#: Minimum character length for a speech to be included
MIN_SPEECH_LENGTH: int = 200

#: Maximum words per text segment (for optimal sentiment model performance)
MAX_WORDS_PER_PARAGRAPH: int = 300

# ============================================================================
# TF-IDF VECTORIZATION
# ============================================================================

TFIDF_PARAMS: Dict[str, Any] = {
    'max_features': 1000,      # Vocabulary size limit
    'min_df': 2,               # Ignore terms appearing in < 2 docs
    'max_df': 0.8,             # Ignore terms appearing in > 80% of docs
    'ngram_range': (1, 2),     # Unigrams and bigrams
    'sublinear_tf': True,      # Apply log(1 + tf) scaling
    'norm': 'l2'               # L2 normalization
}

# ============================================================================
# LDA TOPIC MODELING
# ============================================================================

LDA_PARAMS: Dict[str, Any] = {
    'n_topics': 30,            # Number of latent topics
    'random_state': 42,        # Reproducibility seed
    'max_iter': 30,            # EM algorithm iterations
    'learning_method': 'online',  # 'online' for large datasets
    'n_jobs': -1               # Use all CPU cores
}

COUNT_VECTORIZER_PARAMS: Dict[str, Any] = {
    'max_features': 1000,
    'min_df': 10,
    'max_df': 0.90,
    'ngram_range': (1, 2)
}

# ============================================================================
# BERT TOPIC CLASSIFICATION
# ============================================================================

#: HuggingFace model for German parliamentary topic classification
BERT_TOPIC_MODEL: str = "chkla/parlbert-topic-german"

# ============================================================================
# SENTIMENT ANALYSIS
# ============================================================================

#: Batch sizes (GPU: 256, CPU: 128)
SENTIMENT_BATCH_SIZE_GPU: int = 256
SENTIMENT_BATCH_SIZE_CPU: int = 128

#: Save checkpoint every N batches
CHECKPOINT_INTERVAL: int = 50

# ============================================================================
# DATA PATHS (relative to project root)
# ============================================================================

DATA_PATHS: Dict[str, str] = {
    'raw': 'data/raw',
    'interim': 'data/interim',
    'processed': 'data/processed'
}

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

RANDOM_SEED: int = 42
SAMPLE_SIZE: int = 100
SAMPLE_FRACTION: float = 0.01  # 1% sample for sentiment analysis

# ============================================================================
# PARTY MAPPINGS
# ============================================================================

#: factionId → party name
PARTY_ID_MAP: Dict[int, str] = {
    -1: 'Non-MP',
    3: 'Greens',
    4: 'CDU',
    6: 'Left',
    7: 'DP',
    13: 'FDP',
    14: 'Zentrum',
    23: 'SPD'
}

#: Official party colors (hex)
PARTY_COLORS: Dict[str, str] = {
    'CDU': '#000000',
    'SPD': '#E3000F',
    'FDP': '#FFED00',
    'Greens': '#64A12D',
    'Left': '#BE3075'
}

# ============================================================================
# TOPIC LABELS (LDA topic ID → German label)
# ============================================================================

TOPIC_LABEL_MAP: Dict[str, str] = {
    '1': 'Wirtschaft & Arbeitsmarkt',
    '3': 'Staat, Verwaltung & öff. Leistungen',
    '4': 'Gesetzgebung & Verfassungsfragen',
    '6': 'Bildung, Forschung & Zukunftspolitik',
    '8': 'Sozial-, Familien- & Gesellschaftspolitik',
    '9': 'Europapolitik, Energie & Klima',
    '11': 'Außen-, Sicherheits- & Menschenrechtspolitik',
    '13': 'Haushalt & Finanzpolitik',
    '14': 'Gesetzgebung & Verfassungsfragen'  # Merged with topic 4
}

#: Control/procedural topics to exclude from analysis
CONTROL_TOPICS: List[str] = ['0', '2', '5', '7', '10', '12']
