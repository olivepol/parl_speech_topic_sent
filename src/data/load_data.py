"""
Data loading utilities for the parliamentary speeches project.

This module provides functions for loading speech data from various formats
(Parquet, CSV) and creating filtered samples for analysis.

Functions:
    get_data_dir: Get the project data directory path
    load_speeches: Load the full speeches dataset
    load_sample: Load the pre-created sample dataset
    load_processed: Load any processed data file

Example:
    >>> from src.data.load_data import load_speeches, load_sample
    >>> df_full = load_speeches()  # Full dataset
    >>> df_sample = load_sample()  # Pre-filtered sample
"""

import polars as pl
from pathlib import Path
from typing import Optional


def get_data_dir() -> Path:
    """
    Get the data directory path, creating it if it doesn't exist.
    
    Returns:
        Path to the data/ directory relative to project root
    """
    data_dir = Path(__file__).parent.parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    return data_dir


def load_speeches(source: str = 'parquet') -> pl.DataFrame:
    """
    Load the full speeches dataset.
    
    Args:
        source: Either 'parquet' or 'csv' to specify the file format
        
    Returns:
        Polars DataFrame containing the speeches data
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If source is not 'parquet' or 'csv'
        
    Example:
        >>> df = load_speeches('parquet')
        >>> print(f"Loaded {df.shape[0]} speeches")
    """
    if source not in ['parquet', 'csv']:
        raise ValueError(f"source must be 'parquet' or 'csv', got {source}")
    
    data_dir = get_data_dir()
    
    if source == 'parquet':
        file_path = data_dir / 'raw' / 'speeches.parquet'
    else:
        file_path = data_dir / 'raw' / 'speeches.csv'
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    if source == 'parquet':
        return pl.read_parquet(file_path)
    else:
        return pl.read_csv(file_path)


def load_sample(source: str = 'csv') -> pl.DataFrame:
    """
    Load the sample dataset (100 rows) from data/raw/df_sample.csv.
    
    Args:
        source: Either 'csv' for the sample file
        
    Returns:
        Polars DataFrame containing the sample data
        
    Raises:
        FileNotFoundError: If the sample file doesn't exist
    """
    data_dir = get_data_dir()
    file_path = data_dir / 'raw' / 'df_sample.csv'
    
    if not file_path.exists():
        raise FileNotFoundError(f"Sample file not found: {file_path}")
    
    return pl.read_csv(file_path)


def load_cleaned() -> pl.DataFrame:
    """
    Load the cleaned and processed dataset from data/processed/df_sample_cleaned.csv.
    
    Returns:
        Polars DataFrame containing the cleaned data
        
    Raises:
        FileNotFoundError: If the cleaned file doesn't exist
    """
    data_dir = get_data_dir()
    file_path = data_dir / 'processed' / 'df_sample_cleaned.csv'
    
    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {file_path}. Please run the data cleaning notebook first.")
    
    return pl.read_csv(file_path)


def load_data(use_sample: bool = False, source: str = 'auto') -> pl.DataFrame:
    """
    Convenience function to load data.
    
    Args:
        use_sample: If True, load sample data. If False (default), load full dataset.
        source: 'auto' (default parquet for full, csv for sample), 
                'parquet', or 'csv'
        
    Returns:
        Polars DataFrame
    """
    if use_sample:
        return load_sample()
    else:
        if source == 'auto':
            source = 'parquet'
        return load_speeches(source=source)


if __name__ == '__main__':
    # Example usage
    print("Loading sample data...")
    df_sample = load_sample()
    print(f"Sample shape: {df_sample.shape}")
    print(df_sample.head())
    
    print("\nLoading full dataset...")
    df_full = load_speeches()
    print(f"Full dataset shape: {df_full.shape}")
