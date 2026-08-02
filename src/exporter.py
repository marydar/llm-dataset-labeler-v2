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
    


def push_to_hub(
        results,
        repo_id
):
    final_results = [
    {
        "text": x["text"],
        "label": x["label"]
    }
    for x in results
    ]

    dataset = Dataset.from_list(
        final_results
    )

    dataset.push_to_hub(
        repo_id
    )

    print(
        f"Uploaded to {repo_id}"
    )


    return dataset