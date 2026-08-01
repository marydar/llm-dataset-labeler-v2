from datasets import load_from_disk
import pandas as pd

dataset = load_from_disk("output/labeled_dataset")

df = dataset.to_pandas()

print(df.head(10))

# import datasets

# print(datasets.__file__)