
from datasets import load_dataset, Dataset
from tqdm import tqdm

from config import MODEL


# -----------------------------
# Labels
# -----------------------------

VALID_LABELS = {
    "Desktop & Mobile & Web Development",
    "Cybersecurity",
    "AI / Machine Learning / Data Science",
    "Infrastructure (DevOps, Cloud, Databases, Networking)",
    "Clinical Diagnosis, Treatment & Surgery",
    "Medication & Pharmacology",
    "Mental Health (Clinical)",
    "Healthcare Organizations, System, Hospitals",
    "Nutrition",
    "Payments & Personal Budgeting",
    "Banking",
    "Investment, Markets & Cryptocurrency",
    "Corporate Accounting",
    "Physics & Mathematics",
    "Chemistry",
    "Biology",
    "Team Sports",
    "Individual Sports",
    "Fitness & Training",
    "Civil, Structural & Architecture",
    "Mechanical & Electrical Engineering",
    "Family & Relationships",
    "Personal Growth & Reflection",
    "Travel",
    "Marketing & Sales",
    "Entrepreneurship & Startups",
    "Management & Strategy & Human Resources",
    "Criminal & Civil Law",
    "Labor, Family & Contract Law",
    "Corporate, Regulatory & International Law",
    "General Law (misc.)",
    "Game",
    "Film",
    "Music",
    "Literature",
    "Painting",
}


HIERARCHY = {
    "Technology & Programming": [
        "Desktop & Mobile & Web Development",
        "Cybersecurity",
        "AI / Machine Learning / Data Science",
        "Infrastructure (DevOps, Cloud, Databases, Networking)"
    ],

    "Medical": [
        "Clinical Diagnosis, Treatment & Surgery",
        "Medication & Pharmacology",
        "Mental Health (Clinical)",
        "Healthcare Organizations, System, Hospitals",
        "Nutrition"
    ],

    "Finance": [
        "Payments & Personal Budgeting",
        "Banking",
        "Investment, Markets & Cryptocurrency",
        "Corporate Accounting"
    ],

    "Science": [
        "Physics & Mathematics",
        "Chemistry",
        "Biology"
    ],

    "Sports": [
        "Team Sports",
        "Individual Sports",
        "Fitness & Training"
    ],

    "Engineering": [
        "Civil, Structural & Architecture",
        "Mechanical & Electrical Engineering"
    ],

    "Personal": [
        "Family & Relationships",
        "Personal Growth & Reflection",
        "Travel"
    ],

    "Business": [
        "Marketing & Sales",
        "Entrepreneurship & Startups",
        "Management & Strategy & Human Resources"
    ],

    "Law": [
        "Criminal & Civil Law",
        "Labor, Family & Contract Law",
        "Corporate, Regulatory & International Law",
        "General Law (misc.)"
    ],

    "Art": [
        "Game",
        "Film",
        "Music",
        "Literature",
        "Painting"
    ]
}


CHILD_TO_PARENT = {
    child: parent
    for parent, children in HIERARCHY.items()
    for child in children
}



def validate_and_add_metadata(dataset):

    cleaned = []

    stats = {
        "empty_text": 0,
        "empty_label": 0,
        "invalid_label": 0,
    }


    seen_texts = set()
    duplicate_count = 0


    for row in tqdm(
        dataset,
        desc="Validating"
    ):

        text = row.get("text")
        label = row.get("label")


        # Check text
        if not isinstance(text, str) or not text.strip():

            stats["empty_text"] += 1
            continue


        text = text.strip()


        # Check label
        if not isinstance(label, str) or not label.strip():

            stats["empty_label"] += 1
            continue


        label = label.strip()


        # Check label validity
        if label not in VALID_LABELS:

            stats["invalid_label"] += 1
            continue


        # Duplicate check
        if text in seen_texts:

            duplicate_count += 1
            continue


        seen_texts.add(text)


        cleaned.append(
            {
                "text": text,
                "label": label,
                "parent_label": CHILD_TO_PARENT[label],
                "generator_model": MODEL,
                "source": "real",
            }
        )


    print("\nValidation report")
    print("----------------------")
    print(f"Empty texts removed: {stats['empty_text']}")
    print(f"Empty labels removed: {stats['empty_label']}")
    print(f"Invalid labels removed: {stats['invalid_label']}")
    print(f"Duplicate texts removed: {duplicate_count}")
    print(f"Final samples: {len(cleaned)}")


    return cleaned



def main():

    OLD_DATASET = "maryamdar/llm-chats-labeled"

    NEW_DATASET_PATH = "maryamdar/llm-chats-labeled-v2"

    NEW_HF_REPO = "maryamdar/llm-chats-labeled-v2"
    # Example:
    # NEW_HF_REPO = "your_username/prompt-classification-clean"


    print("Loading dataset...")

    dataset = load_dataset(
        OLD_DATASET,
        split="train"
    )


    print(dataset)


    cleaned_rows = validate_and_add_metadata(
        dataset
    )


    new_dataset = Dataset.from_list(
        cleaned_rows
    )


    print(new_dataset)


    # Save locally
    new_dataset.save_to_disk(
        NEW_DATASET_PATH
    )

    print(
        f"Saved locally: {NEW_DATASET_PATH}"
    )


    # Optional upload
    if NEW_HF_REPO:

        new_dataset.push_to_hub(
            NEW_HF_REPO
        )

        print(
            f"Uploaded: {NEW_HF_REPO}"
        )



if __name__ == "__main__":
    main()