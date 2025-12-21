import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt

# Load the sample data from raw folder
data_dir = Path('data')
sample_path = data_dir / 'raw' / 'df_sample.csv'

df = pl.read_csv(sample_path)

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# 1. Dataset shape and basic info
print("\n1. DATASET SHAPE AND INFO")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\nColumn names and types:")
print(df.schema)

# 2. Missing values
print("\n2. MISSING VALUES")
missing_counts = df.null_count()
print(missing_counts)

# 3. Basic statistics
print("\n3. BASIC STATISTICS")
print(df.describe())

# 4. First few rows
print("\n4. FIRST FEW ROWS")
print(df.head())

# 5. Column information
print("\n5. COLUMN INFORMATION")
for col in df.columns:
    col_type = df.schema[col]
    unique_count = df.select(col).n_unique()
    print(f"  {col}: {col_type}, Unique values: {unique_count}")

# 6. Text length analysis (if text column exists)
text_columns = [col for col in df.columns if df.schema[col] == pl.Utf8]
if text_columns:
    print(f"\n6. TEXT COLUMN ANALYSIS")
    for col in text_columns:
        df = df.with_columns(
            pl.col(col).str.len_chars().alias(f"{col}_length")
        )
        lengths = df.select(f"{col}_length")
        print(f"\n  {col}:")
        print(f"    Average length: {lengths.mean().item():.0f} chars")
        print(f"    Min length: {lengths.min().item()} chars")
        print(f"    Max length: {lengths.max().item()} chars")

print("\n" + "=" * 80)
print("END OF EXPLORATORY DATA ANALYSIS")
print("=" * 80)
