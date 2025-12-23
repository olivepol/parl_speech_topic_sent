"""
Download German Parliament Speeches dataset from HuggingFace.

This script downloads the full parliamentary speeches dataset and saves it
as a Parquet file for efficient loading in downstream analysis.

Prerequisites:
    - HuggingFace CLI login: `huggingface-cli login`
    - Required packages: polars

Usage:
    python scripts/import_data.py

Output:
    data/raw/speeches.parquet - Full dataset (~2GB)

Dataset:
    https://huggingface.co/datasets/emilpartow/german-parliament-speeches
"""

import polars as pl
from pathlib import Path

# Create data directory if it doesn't exist
data_dir = Path('data/raw')
data_dir.mkdir(parents=True, exist_ok=True)

# Login using e.g. `huggingface-cli login` to access this dataset
print("Loading dataset from HuggingFace (this may take a few minutes)...")
df = pl.read_csv('hf://datasets/emilpartow/german-parliament-speeches/speeches.csv')

# Save the dataframe
output_path = data_dir / 'speeches.parquet'
df.write_parquet(output_path)

print(f"✓ Data saved to {output_path}")
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns}")
print(df.head())

