from datasets import (
    Dataset,
    load_dataset,
    concatenate_datasets
)


FINAL_COLUMNS = [
    "text",
    "label",
    "parent_label",
    "generator_model",
    "source"
]


def prepare_for_export(results):
    """
    Remove temporary fields before saving.
    """

    return [
        {
            key: item[key]
            for key in FINAL_COLUMNS
            if key in item
        }
        for item in results
    ]


def save_dataset(
        results,
        path
):

    results = prepare_for_export(results)

    dataset = Dataset.from_list(
        results
    )

    dataset.save_to_disk(path)

    print(
        f"Saved {len(dataset)} samples."
    )



def push_to_hub(
        results,
        repo_id
):

    results = prepare_for_export(results)

    new_dataset = Dataset.from_list(
        results
    )


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


    # Remove duplicate texts
    df = merged.to_pandas()

    before = len(df)

    df = df.drop_duplicates(
        subset=["text"],
        keep="first"
    ).reset_index(drop=True)

    after = len(df)

    print(
        f"Removed {before-after} duplicates."
    )


    merged = Dataset.from_pandas(
        df,
        preserve_index=False
    )


    merged.push_to_hub(
        repo_id
    )


    print(
        f"Uploaded {len(merged)} samples to {repo_id}"
    )

    return merged