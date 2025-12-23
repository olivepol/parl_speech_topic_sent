import polars as pl
from pathlib import Path

# Load the full dataset - use absolute path from script location
script_dir = Path(__file__).parent
speeches_path = script_dir.parent / 'data' / 'raw' / 'speeches.parquet'

df = pl.read_parquet(speeches_path)

print(f"Total rows in dataset: {df.shape[0]}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Filter for SPD and CDU/CSU speeches (factionId 4 and 23)
df_filtered = df.filter(
    (pl.col('date') >= '2000-01-01') & 
    ((pl.col('factionId') == 23) | (pl.col('factionId') == 4))
)

print(f"\nFiltered speeches (factionId 4 & 23) from 2000 onwards: {df_filtered.shape[0]}")

# Sample 50% of the filtered data
df_sample = df_filtered.sample(fraction=0.5, seed=42)

print(f"After 50% sampling: {df_sample.shape[0]}")
print(f"Filtered date range: {df_sample['date'].min()} to {df_sample['date'].max()}")

# Save the sample to raw folder (correct location for raw samples)
raw_dir = script_dir.parent / 'data' / 'raw'
raw_dir.mkdir(exist_ok=True)
output_path = raw_dir / 'df_sample.csv'
df_sample.write_csv(output_path)

print(f"\nSample saved to {output_path}")
print(f"Sample shape: {df_sample.shape}")
print(f"\nFirst few rows:")
print(df_sample.head())
