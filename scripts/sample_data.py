import polars as pl
from pathlib import Path

# Load the full dataset - use absolute path from script location
script_dir = Path(__file__).parent
speeches_path = script_dir.parent / 'data' / 'raw' / 'speeches.parquet'

df = pl.read_parquet(speeches_path)

# Sample 100 random rows
df_sample = df.sample(n=1000, seed=42)

# Save the sample to raw folder (correct location for raw samples)
raw_dir = script_dir.parent / 'data' / 'raw'
raw_dir.mkdir(exist_ok=True)
output_path = raw_dir / 'df_sample.csv'
df_sample.write_csv(output_path)

print(f"Sample of 100 rows saved to {output_path}")
print(f"Sample shape: {df_sample.shape}")
print(df_sample.head())
