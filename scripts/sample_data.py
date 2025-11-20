import polars as pl
from pathlib import Path

# Load the full dataset
data_dir = Path('data')
speeches_path = data_dir / 'speeches.parquet'

df = pl.read_parquet(speeches_path)

# Sample 100 random rows
df_sample = df.sample(n=100, seed=42)

# Save the sample to CSV
output_path = data_dir / 'df_sample.csv'
df_sample.write_csv(output_path)

print(f"Sample of 100 rows saved to {output_path}")
print(f"Sample shape: {df_sample.shape}")
print(df_sample.head())
