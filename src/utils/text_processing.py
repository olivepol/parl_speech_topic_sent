"""
Text processing utilities for parliamentary speech analysis.

This module provides high-performance text processing functions using
native Polars expressions. Functions are optimized for large datasets
and avoid slow Python UDFs.

Performance Notes:
    - Native Polars operations are 10-100x faster than map_elements
    - Use batch processing for ML model inference
    - Prefer vectorized operations over row-by-row processing

Functions:
    trim_to_max_words_native: Trim text to maximum word count
    add_word_count: Add word count column
    add_char_count: Add character count column

Example:
    >>> import polars as pl
    >>> from src.utils.text_processing import trim_to_max_words_native
    >>> df = pl.DataFrame({'text': ['This is a very long text...']})
    >>> df_trimmed = trim_to_max_words_native(df, 'text', max_words=50)
"""

import polars as pl
from typing import List, Tuple


def trim_to_max_words_native(
    df: pl.DataFrame, 
    col: str, 
    max_words: int = 300
) -> pl.DataFrame:
    """
    Trim text column to maximum number of words using native Polars operations.
    
    This implementation uses Polars' built-in string operations which are
    significantly faster than Python UDFs (10-100x speedup).
    
    Args:
        df: Input Polars DataFrame
        col: Name of the text column to trim
        max_words: Maximum number of words to keep (default: 300)
        
    Returns:
        DataFrame with the specified column trimmed to max_words
        
    Example:
        >>> df = pl.DataFrame({'speech': ['word ' * 500]})
        >>> df_trimmed = trim_to_max_words_native(df, 'speech', max_words=100)
        >>> len(df_trimmed['speech'][0].split())
        100
    """
    return df.with_columns(
        pl.col(col)
        .str.split(' ')           # Split into list of words
        .list.head(max_words)     # Take first max_words elements
        .list.join(' ')           # Join back into string
        .alias(col)
    )


def add_word_count(
    df: pl.DataFrame, 
    text_col: str, 
    count_col: str = 'word_count'
) -> pl.DataFrame:
    """
    Add a column with word counts for each row.
    
    Args:
        df: Input Polars DataFrame
        text_col: Name of the text column to count words in
        count_col: Name of the output count column (default: 'word_count')
        
    Returns:
        DataFrame with an additional column containing word counts
        
    Example:
        >>> df = pl.DataFrame({'text': ['hello world', 'one two three']})
        >>> df_with_counts = add_word_count(df, 'text')
        >>> df_with_counts['word_count'].to_list()
        [2, 3]
    """
    return df.with_columns(
        pl.col(text_col).str.split(' ').list.len().alias(count_col)
    )


def add_char_count(
    df: pl.DataFrame, 
    text_col: str, 
    count_col: str = 'char_count'
) -> pl.DataFrame:
    """
    Add character count column using native Polars expressions.
    
    Args:
        df: Polars DataFrame
        text_col: Name of the text column
        count_col: Name of the output count column
        
    Returns:
        DataFrame with character count column added
    """
    return df.with_columns(
        pl.col(text_col).str.len_chars().alias(count_col)
    )


def split_text_to_rows(df: pl.DataFrame, text_col: str, delimiter: str = '\n\n') -> pl.DataFrame:
    """
    Split text column by delimiter and explode into rows.
    Much faster than iterating in Python.
    
    Args:
        df: Polars DataFrame
        text_col: Name of the text column to split
        delimiter: String to split on (default: double newline)
        
    Returns:
        DataFrame with one row per split segment
    """
    return (
        df
        .with_columns(
            pl.col(text_col).str.split(delimiter).alias('_segments')
        )
        .explode('_segments')
        .with_columns(
            pl.col('_segments').str.strip_chars().alias(text_col)
        )
        .drop('_segments')
        .filter(pl.col(text_col).str.len_chars() > 0)  # Remove empty rows
    )


def batch_process_sentiment(texts: list, model, batch_size: int = 32, show_progress: bool = True):
    """
    Process sentiment in batches for better performance.
    
    Args:
        texts: List of text strings
        model: SentimentModel instance
        batch_size: Number of texts to process at once
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (sentiments list, probabilities list)
    """
    from tqdm import tqdm
    
    sentiments = []
    probabilities_list = []
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    # Create iterator with optional progress bar
    batch_iterator = range(0, len(texts), batch_size)
    if show_progress:
        batch_iterator = tqdm(batch_iterator, total=total_batches, desc="Sentiment analysis")
    
    for i in batch_iterator:
        batch = texts[i:i + batch_size]
        
        # Process batch
        classes, probs = model.predict_sentiment(batch, output_probabilities=True)
        sentiments.extend(classes)
        probabilities_list.extend(probs)
    
    if show_progress:
        print(f"\n✓ Processed {len(texts)} texts in {total_batches} batches")
    
    return sentiments, probabilities_list
