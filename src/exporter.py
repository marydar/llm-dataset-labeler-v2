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


    return dataset