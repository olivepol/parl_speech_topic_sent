"""
Create stratified sample of parliamentary speeches for analysis.

This script filters the full dataset to CDU and SPD speeches from 2000 onwards,
then creates a 50% random sample for computational efficiency while maintaining
representative distributions.

Prerequisites:
    - Run import_data.py first to download the full dataset

Usage:
    python scripts/sample_data.py

Input:
    data/raw/speeches.parquet - Full dataset from HuggingFace

Output:
    data/raw/df_sample.csv - 50% stratified sample (CDU/SPD, 2000+)

Filtering Criteria:
    - Date >= 2000-01-01
    - factionId in [4 (CDU), 23 (SPD)]
    - Random 50% sample with seed=42 for reproducibility
"""

import polars as pl
from pathlib import Path

# Load the full dataset - use absolute path from script location
script_dir = Path(__file__).parent
speeches_path = script_dir.parent / 'data' / 'raw' / 'speeches.parquet'

print(f"Loading data from: {speeches_path}")
df = pl.read_parquet(speeches_path)

print(f"Total rows in dataset: {df.shape[0]:,}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Filter for SPD and CDU/CSU speeches (factionId 4 and 23)
df_filtered = df.filter(
    (pl.col('date') >= '2000-01-01') & 
    ((pl.col('factionId') == 23) | (pl.col('factionId') == 4))
)

print(f"\n✓ Filtered speeches (CDU + SPD, 2000+): {df_filtered.shape[0]:,}")

# Sample 50% of the filtered data
df_sample = df_filtered.sample(fraction=0.5, seed=42)

print(f"✓ After 50% sampling: {df_sample.shape[0]:,}")
print(f"  Date range: {df_sample['date'].min()} to {df_sample['date'].max()}")

# Save the sample to raw folder (correct location for raw samples)
raw_dir = script_dir.parent / 'data' / 'raw'
raw_dir.mkdir(exist_ok=True)
output_path = raw_dir / 'df_sample.csv'
df_sample.write_csv(output_path)

print(f"\n✓ Sample saved to: {output_path}")
print(f"  Shape: {df_sample.shape}")
print(f"\nFirst few rows:")
print(df_sample.head())
