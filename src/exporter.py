from datasets import Dataset



def save_dataset(
        data,
        path
):

    dataset = Dataset.from_list(
        data
    )


    dataset.save_to_disk(
        path
    )
    


# def push_to_hub(
#         results,
#         repo_id
# ):
#     final_results = [
#     {
#         "text": x["text"],
#         "label": x["label"]
#     }
#     for x in results
#     ]

#     dataset = Dataset.from_list(
#         final_results
#     )

#     dataset.push_to_hub(
#         repo_id
#     )

#     print(
#         f"Uploaded to {repo_id}"
#     )


#     return dataset

from datasets import (
    Dataset,
    load_dataset,
    concatenate_datasets
)

def push_to_hub(
        results,
        repo_id
):
    """
    Downloads the existing HF dataset (if it exists),
    merges the new samples,
    removes duplicates,
    and uploads the updated dataset.
    """

    # Keep only training columns
    final_results = [
        {
            "text": item["text"],
            "label": item["label"]
        }
        for item in results
    ]

    new_dataset = Dataset.from_list(
        final_results
    )

    # -------------------------
    # Download existing dataset
    # -------------------------

    try:

        print("Downloading existing dataset...")

        old_dataset = load_dataset(
            repo_id,
            split="train"
        )

        print(
            f"Existing samples: {len(old_dataset)}"
        )

        merged = concatenate_datasets(
            [
                old_dataset,
                new_dataset
            ]
        )

    except Exception:

        print(
            "Dataset does not exist yet. Creating a new one."
        )

        merged = new_dataset

    # -------------------------
    # Remove duplicates
    # -------------------------

    df = merged.to_pandas()

    before = len(df)

    df = df.drop_duplicates(
        subset="text"
    )

    after = len(df)

    print(
        f"Removed {before-after} duplicates."
    )

    merged = Dataset.from_pandas(
        df,
        preserve_index=False
    )

    # -------------------------
    # Upload
    # -------------------------

    merged.push_to_hub(
        repo_id
    )

    print(
        f"Uploaded {len(merged)} samples to {repo_id}"
    )

    return merged