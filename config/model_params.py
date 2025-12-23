"""
Configuration parameters for NLP models.
Centralizes all hyperparameters to avoid magic numbers in notebooks.
"""

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
MIN_SPEECH_LENGTH = 200  # Minimum characters for a speech to be included
MAX_WORDS_PER_PARAGRAPH = 300  # Maximum words before trimming

# ============================================================================
# TF-IDF VECTORIZATION
# ============================================================================
TFIDF_PARAMS = {
    'max_features': 1000,
    'min_df': 2,
    'max_df': 0.8,
    'ngram_range': (1, 2),
    'sublinear_tf': True,
    'norm': 'l2'
}

# ============================================================================
# LDA TOPIC MODELING
# ============================================================================
LDA_PARAMS = {
    'n_topics': 30,
    'random_state': 42,
    'max_iter': 30,
    'learning_method': 'online',
    'n_jobs': -1
}

COUNT_VECTORIZER_PARAMS = {
    'max_features': 1000,
    'min_df': 10,
    'max_df': 0.90,
    'ngram_range': (1, 2)
}

# ============================================================================
# BERT TOPIC CLASSIFICATION
# ============================================================================
BERT_TOPIC_MODEL = "chkla/parlbert-topic-german"

# ============================================================================
# SENTIMENT ANALYSIS
# ============================================================================
# germansentiment model is loaded without explicit params

# ============================================================================
# DATA PATHS (relative to project root)
# ============================================================================
DATA_PATHS = {
    'raw': 'data/raw',
    'interim': 'data/interim',
    'processed': 'data/processed'
}

# ============================================================================
# RANDOM SEED FOR REPRODUCIBILITY
# ============================================================================
RANDOM_SEED = 42
SAMPLE_SIZE = 100
