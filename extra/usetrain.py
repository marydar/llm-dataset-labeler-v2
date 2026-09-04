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


label_mapping = {
    "Desktop & Mobile & Web Development":    "Desktop & Mobile & Web Development",
    "Cybersecurity":    "Cybersecurity",
    "AI / Machine Learning / Data Science":    "AI / Machine Learning / Data Science",
    "Infrastructure (DevOps, Cloud, Databases, Networking)":    "Infrastructure (DevOps, Cloud, Databases, Networking)",
    "Clinical Diagnosis, Treatment & Surgery":    "Clinical Diagnosis, Treatment & Surgery",
    "Medication & Pharmacology":    "Medication & Pharmacology",
    "Mental Health (Clinical)":    "Mental Health",
    "Healthcare Organizations, System, Hospitals":    "Healthcare Organizations, System, Hospitals",
    "Nutrition":    "Nutrition",
    "Payments & Personal Budgeting":    "Payments & Personal Budgeting",
    "Banking":None,
    "Investment, Markets & Cryptocurrency":    "Investment, Markets & Cryptocurrency",
    "Corporate Accounting":    "Corporate",
    "Physics & Mathematics":    "Physics, Mathematics",
    "Chemistry":    "Chemistry",
    "Biology":    "Biology",
    "Team Sports":    "Team Sports",
    "Individual Sports":    "Individual Sports",
    "Fitness & Training":    "Fitness",
    "Civil, Structural & Architecture":    "Civil, Structural & Architecture",
    "Mechanical & Electrical Engineering":    "Mechanical & Electrical Engineering",
    "Family & Relationships":    "Family & Relationships",
    "Personal Growth & Reflection":    "Personal",
    "Travel":    "Travel",
    "Marketing & Sales":    "Marketing & Sales",
    "Entrepreneurship & Startups":    "Entrepreneurship & Startups",
    "Management & Strategy & Human Resources":    "Management & Strategy & Human Resources",
    "Criminal & Civil Law":None,
    "Labor, Family & Contract Law":None,
    "Corporate, Regulatory & International Law":None,
    "General Law (misc.)":None,
    "Game": "Game",
    "Film":"Film",
    "Music":  "Music",
    "Literature":"Literature",
    "Painting":"Painting"
}
from datasets import load_dataset

# Load dataset
dataset = load_dataset("maryamdar/topic_classification_dataset_gen")

for split in dataset:
    ds = dataset[split]

    use_for_train = []

    for old_label, label in zip(ds["old_label"], ds["label"]):
        expected_label = label_mapping.get(old_label)

        if expected_label is None:
            use_for_train.append(False)
        else:
            use_for_train.append(expected_label == label)

    dataset[split] = ds.remove_columns("use_for_train")
    dataset[split] = dataset[split].add_column(
        "use_for_train",
        use_for_train
    )
    
for split in dataset:
    ds = dataset[split]

    total = len(ds)
    train_count = sum(ds["use_for_train"])
    excluded = total - train_count

    print(f"\n{split}:")
    print(f"Total:         {total}")
    print(f"Use for train: {train_count}")
    print(f"Excluded:      {excluded}")
    print(f"Train %:       {train_count / total * 100:.2f}%")
    
dataset.push_to_hub("maryamdar/topic_classification_dataset_gen")