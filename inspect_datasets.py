from datasets import load_from_disk
import pandas as pd

dataset = load_from_disk("output/labeled_dataset")

df = dataset.to_pandas()

print(df)

# import datasets

# print(datasets.__file__)