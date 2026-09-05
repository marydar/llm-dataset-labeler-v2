from datasets import (
    Dataset,
    load_dataset,
    concatenate_datasets,
    Features,
    Value
)


FINAL_COLUMNS = [
    "text",
    "label",
    "parent_label",
    "generator_model",
    "source",
    "confidence_score",
    "source_dataset",
]


FEATURES = Features({
    "text": Value("string"),
    "label": Value("string"),
    "parent_label": Value("string"),
    "generator_model": Value("string"),
    "source": Value("string"),
    "confidence_score": Value("float32"),
    "source_dataset": Value("string"),
})



def prepare_for_export(results):

    return [
        {
            **{
                key: item[key]
                for key in FINAL_COLUMNS
                if key in item
            },
            "confidence_score": round(
                float(item["confidence_score"]),
                2
            )
            if "confidence_score" in item and item["confidence_score"] is not None
            else None
        }
        for item in results
    ]

def save_dataset(
        results,
        path
):

    results = prepare_for_export(results)

    dataset = Dataset.from_list(
        results,
        features=FEATURES
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
        results,
        features=FEATURES
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

        # Make sure the old dataset also uses float32
        old_dataset = old_dataset.cast(FEATURES)

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
        features=FEATURES,
        preserve_index=False
    )


    merged.push_to_hub(
        repo_id
    )


    print(
        f"Uploaded {len(merged)} samples to {repo_id}"
    )

    return merged