import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

data_scorce = "data/verl_countdown.jsonl"

with open(data_scorce, 'r') as f:
    dataset = [json.loads(line) for line in f]
print(f"Total entries in dataset: {len(dataset)}")

# generate validation set dataset[200:700]
val_data = dataset[200:700]
df = pd.DataFrame(val_data)
df.to_parquet(f"data/dataset_sizes/test.parquet", engine='pyarrow', compression='snappy')

# generate training set with different sizes
dataset_sizes = [2,5,10,20, 50]
larger_dataset_sizes = [200, 1000]    
for size in dataset_sizes:
    train_data = dataset[:size]
    df = pd.DataFrame(train_data)
    df.to_parquet(f"data/dataset_sizes/train_{size}.parquet", engine='pyarrow', compression='snappy')

# for larger sizes, take first 200, and skip the validation set, then take the rest
for size in larger_dataset_sizes:
    train_data = dataset[:200] + dataset[700:700 + (size - 200)]
    df = pd.DataFrame(train_data)
    df.to_parquet(f"data/dataset_sizes/train_{size}.parquet", engine='pyarrow', compression='snappy')

print("Datasets generated successfully.")