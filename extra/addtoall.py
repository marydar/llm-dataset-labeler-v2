from datasets import load_dataset, concatenate_datasets

# --------------------------------------------------
# Load datasets
# --------------------------------------------------

ds1 = load_dataset("maryamdar/topic_classification_dataset_gen")
ds2 = load_dataset("maryamdar/prompt-batch-test")

ds1 = ds1["train"]
ds2 = ds2["train"]


# --------------------------------------------------
# Select second_ generation groups
# --------------------------------------------------

second_ds = ds2.filter(
    lambda x: x["generation_group"].startswith("second_")
)

print(f"Selected {len(second_ds)} rows from ds2")


# --------------------------------------------------
# Fix columns for the new rows
# --------------------------------------------------

second_ds = second_ds.map(
    lambda x: {
        "old_label": None,
        "old_parent_label": None,
        "use_for_train": True,
    }
)


# --------------------------------------------------
# Make sure both datasets have the same columns
# --------------------------------------------------

all_columns = list(dict.fromkeys(
    ds1.column_names + second_ds.column_names
))

for col in all_columns:
    if col not in ds1.column_names:
        ds1 = ds1.add_column(col, [None] * len(ds1))

    if col not in second_ds.column_names:
        second_ds = second_ds.add_column(col, [None] * len(second_ds))


# --------------------------------------------------
# Same column order
# --------------------------------------------------

ds1 = ds1.select_columns(all_columns)
second_ds = second_ds.select_columns(all_columns)


# --------------------------------------------------
# Combine
# --------------------------------------------------

final_ds = concatenate_datasets([
    ds1,
    second_ds
])

print(f"ds1:       {len(ds1)}")
print(f"second_ds: {len(second_ds)}")
print(f"final:     {len(final_ds)}")


# --------------------------------------------------
# Push back to ds1's HF repository
# --------------------------------------------------

final_ds.push_to_hub(
    "maryamdar/topic_classification_dataset_gen"
)