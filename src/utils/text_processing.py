"""
Utility functions for text processing with Polars.
Uses native Polars expressions for better performance.
"""

import polars as pl


def trim_to_max_words_native(df: pl.DataFrame, col: str, max_words: int = 300) -> pl.DataFrame:
    """
    Trim text column to maximum number of words using native Polars operations.
    
    This is ~10-100x faster than using map_elements with a Python function.
    
    Args:
        df: Polars DataFrame
        col: Name of the text column to trim
        max_words: Maximum number of words to keep
        
    Returns:
        DataFrame with trimmed text column
    """
    return df.with_columns(
        pl.col(col)
        .str.split(' ')  # Split into words
        .list.head(max_words)  # Take first max_words
        .list.join(' ')  # Join back
        .alias(col)
    )


def add_word_count(df: pl.DataFrame, text_col: str, count_col: str = 'word_count') -> pl.DataFrame:
    """
    Add word count column using native Polars expressions.
    
    Args:
        df: Polars DataFrame
        text_col: Name of the text column
        count_col: Name of the output count column
        
    Returns:
        DataFrame with word count column added
    """
    return df.with_columns(
        pl.col(text_col).str.split(' ').list.len().alias(count_col)
    )


def add_char_count(df: pl.DataFrame, text_col: str, count_col: str = 'char_count') -> pl.DataFrame:
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
        show_progress: Whether to show progress
        
    Returns:
        Tuple of (sentiments list, probabilities list)
    """
    sentiments = []
    probabilities_list = []
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        if show_progress:
            batch_num = i // batch_size + 1
            print(f"  Processing batch {batch_num}/{total_batches}...", end='\r')
        
        # Process batch
        classes, probs = model.predict_sentiment(batch, output_probabilities=True)
        sentiments.extend(classes)
        probabilities_list.extend(probs)
    
    if show_progress:
        print(f"  Processed {len(texts)} texts in {total_batches} batches")
    
    return sentiments, probabilities_list
