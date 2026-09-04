import json
from datasets import load_dataset

# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "maryamdar/topic_classification_dataset_gen"
JSON_PATH = "classification_results_old_to_new.json"

VALID_LABELS = [
    "Desktop & Mobile & Web Development",
    "Cybersecurity",
    "AI / Machine Learning / Data Science",
    "Infrastructure (DevOps, Cloud, Databases, Networking)",
    "Clinical Diagnosis, Treatment & Surgery",
    "Medication & Pharmacology",
    "Mental Health",
    "Healthcare Organizations, System, Hospitals",
    "Nutrition",
    "Payments & Personal Budgeting",
    "Investment, Markets & Cryptocurrency",
    "Corporate",
    "Physics, Mathematics",
    "Chemistry",
    "Biology",
    "Team Sports",
    "Individual Sports",
    "Fitness",
    "Civil, Structural & Architecture",
    "Mechanical & Electrical Engineering",
    "Family & Relationships",
    "Personal",
    "Travel",
    "Marketing & Sales",
    "Entrepreneurship & Startups",
    "Management & Strategy & Human Resources",
    "Criminal Law",
    "Family Law",
    "Corporate Law",
    "Civil Law" ,
    "Game",
    "Film",
    "Music",
    "Literature",
    "Painting",
]
# Parent-to-Child Mapping Hierarchy
HIERARCHY = {
    "Programming / Technology": [
        "Desktop & Mobile & Web Development",
        "Cybersecurity",
        "AI / Machine Learning / Data Science",
        "Infrastructure (DevOps, Cloud, Databases, Networking)"
    ],
    "Medical": [
        "Clinical Diagnosis, Treatment & Surgery",
        "Medication & Pharmacology",
        "Mental Health",
        "Healthcare Organizations, System, Hospitals",
        "Nutrition"
    ],
    "Finance": [
        "Payments & Personal Budgeting",
        "Investment, Markets & Cryptocurrency",
        "Corporate"
    ],
    "Science": [
        "Physics, Mathematics",
        "Chemistry",
        "Biology"
    ],
    "Sports": [
        "Team Sports",
        "Individual Sports",
        "Fitness"
    ],
    "Engineering": [
        "Civil, Structural & Architecture",
        "Mechanical & Electrical Engineering",
    ],
    "Personal": [
        "Family & Relationships",
        "Personal",
        "Travel"
    ],
    "Business": [
        "Marketing & Sales",
        "Entrepreneurship & Startups",
        "Management & Strategy & Human Resources"
    ],
     "Law": [
        "Criminal Law",
        "Family Law",
        "Corporate Law",
        "Civil Law" 
    ],
     "Art": [
        "Game",
        "Film",
        "Music",
        "Literature",
        "Painting"
    ],


}

# ============================================================
# LOAD DATA
# ============================================================

dataset = load_dataset(DATASET_NAME)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    json_data = json.load(f)

print(f"JSON samples: {len(json_data)}")


# ============================================================
# CREATE LABEL -> PARENT MAPPING
# ============================================================

label_to_parent = {}

for parent, labels in HIERARCHY.items():
    for label in labels:
        if label in label_to_parent:
            print(f"WARNING: label appears under multiple parents: {label}")

        label_to_parent[label] = parent


# ============================================================
# VALID LABEL CHECK
# ============================================================

valid_labels_set = set(VALID_LABELS)

print(f"Valid labels: {len(valid_labels_set)}")

# Check that every valid label has a parent
missing_parents = valid_labels_set - set(label_to_parent.keys())

if missing_parents:
    print("\nWARNING: These VALID_LABELS are missing from HIERARCHY:")
    for label in sorted(missing_parents):
        print(" -", label)


# ============================================================
# PROCESS EACH SPLIT
# ============================================================

total_updated = 0

not_found = []
duplicate_matches = []
invalid_predictions = []
parent_not_found = []

for split in dataset:

    ds = dataset[split]

    texts = ds["text"]
    old_labels = ds["old_label"]

    new_labels = list(ds["label"])
    new_parent_labels = list(ds["parent_label"])

    # --------------------------------------------------------
    # Create lookup:
    # (text, old_label) -> dataset index(es)
    # --------------------------------------------------------

    lookup = {}

    for i, (text, old_label) in enumerate(zip(texts, old_labels)):

        key = (text, old_label)

        if key not in lookup:
            lookup[key] = []

        lookup[key].append(i)

    # --------------------------------------------------------
    # Process JSON
    # --------------------------------------------------------

    for item in json_data:

        text = item["text"]
        old_label = item["label"]
        llm_prediction = item["llm_prediction"]

        # ----------------------------------------------------
        # 1. Validate LLM prediction FIRST
        # ----------------------------------------------------

        if llm_prediction not in valid_labels_set:

            invalid_predictions.append({
                "text": text,
                "old_label": old_label,
                "llm_prediction": llm_prediction,
                "split": split
            })

            continue

        # ----------------------------------------------------
        # 2. Find matching dataset row
        # ----------------------------------------------------

        key = (text, old_label)

        matches = lookup.get(key, [])

        # No match
        if len(matches) == 0:

            not_found.append({
                "text": text,
                "old_label": old_label,
                "llm_prediction": llm_prediction,
                "split": split
            })

            continue

        # Multiple matches
        if len(matches) > 1:

            duplicate_matches.append({
                "text": text,
                "old_label": old_label,
                "matches": matches,
                "llm_prediction": llm_prediction,
                "split": split
            })

            continue

        # ----------------------------------------------------
        # 3. Exactly one match
        # ----------------------------------------------------

        idx = matches[0]

        # Set new label
        new_labels[idx] = llm_prediction

        # ----------------------------------------------------
        # 4. Find parent
        # ----------------------------------------------------

        if llm_prediction in label_to_parent:

            new_parent_labels[idx] = label_to_parent[llm_prediction]

        else:

            new_parent_labels[idx] = None

            parent_not_found.append({
                "new_label": llm_prediction,
                "text": text,
                "split": split
            })

        total_updated += 1

    # --------------------------------------------------------
    # Replace columns
    # --------------------------------------------------------

    dataset[split] = ds.remove_columns(
        ["label", "parent_label"]
    )

    dataset[split] = dataset[split].add_column(
        "label",
        new_labels
    )

    dataset[split] = dataset[split].add_column(
        "parent_label",
        new_parent_labels
    )


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"Updated rows:        {total_updated}")
print(f"Not found:           {len(not_found)}")
print(f"Duplicate matches:   {len(duplicate_matches)}")
print(f"Invalid predictions: {len(invalid_predictions)}")
print(f"Missing parents:     {len(parent_not_found)}")


# ============================================================
# SHOW INVALID PREDICTIONS
# ============================================================

if invalid_predictions:

    print("\n" + "=" * 60)
    print("INVALID LLM PREDICTIONS")
    print("=" * 60)

    # Show unique invalid labels first
    invalid_labels = sorted(
        set(x["llm_prediction"] for x in invalid_predictions)
    )

    print(f"\nUnique invalid labels: {len(invalid_labels)}")

    for label in invalid_labels:
        count = sum(
            x["llm_prediction"] == label
            for x in invalid_predictions
        )

        print(f"  {count:5d}  {repr(label)}")

    print("\nFirst 10 invalid samples:")

    for x in invalid_predictions[:10]:
        print("\nText:", x["text"][:200])
        print("Old label:", x["old_label"])
        print("Prediction:", repr(x["llm_prediction"]))


# ============================================================
# SHOW NOT FOUND
# ============================================================

if not_found:

    print("\n" + "=" * 60)
    print("NOT FOUND")
    print("=" * 60)

    for x in not_found[:10]:
        print("\nText:", x["text"][:200])
        print("Old label:", x["old_label"])


# ============================================================
# SHOW DUPLICATES
# ============================================================

if duplicate_matches:

    print("\n" + "=" * 60)
    print("DUPLICATE MATCHES")
    print("=" * 60)

    for x in duplicate_matches[:10]:
        print("\nText:", x["text"][:200])
        print("Old label:", x["old_label"])
        print("Dataset indices:", x["matches"])


# ============================================================
# PUSH TO HUB
# ============================================================

dataset.push_to_hub(DATASET_NAME)

print("\nDataset successfully pushed to Hugging Face.")