from datasets import load_dataset


def load_text_dataset(
        dataset_name,
        split="train",
        text_column="text"
):

    dataset = load_dataset(
        dataset_name,
        split=split
    )


    texts = []


    for item in dataset:

        text = item.get(text_column)


        if text and len(text.strip()) > 5:
            texts.append(text.strip())


    return texts