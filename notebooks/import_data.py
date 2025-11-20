import polars as pl
from pathlib import Path

# Create data directory if it doesn't exist
data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

# Login using e.g. `huggingface-cli login` to access this dataset
df = pl.read_csv('hf://datasets/emilpartow/german-parliament-speeches/speeches.csv')


# Save the dataframe
output_path = data_dir / 'speeches.parquet'
df.write_parquet(output_path)

print(f"Data saved to {output_path}")
print(df.head())

