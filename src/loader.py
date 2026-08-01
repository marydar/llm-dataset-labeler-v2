from datasets import load_dataset
from tqdm import tqdm


def load_text_dataset(
        dataset_name,
        split="train",
        conversation_column="conversation",
        language_column="language"
):
    
    print("Loading dataset")

    dataset = load_dataset(
        dataset_name,
        split=split
    )

    print("Loaded dataset")
    print(f"Total samples: {len(dataset)}")


    texts = []


    for item in tqdm(
        dataset,
        desc="Extracting English user prompts",
        total=len(dataset)
    ):

        # Keep only English
        language = item.get(language_column)

        if language != "English":
            continue


        conversation = item.get(
            conversation_column
        )


        if not conversation:
            continue


        # Extract user message
        for message in conversation:

            if message.get("role") == "user":

                text = message.get(
                    "content"
                )

                if text and len(text.strip()) > 5:
                    texts.append(
                        text.strip()
                    )

                break


    print("Loaded English texts")
    print(f"Number of prompts: {len(texts)}")


    return texts