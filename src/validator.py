from config import MODEL
from config import REASONING

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
    ],
    "Not Related": [
        "Not Related"
    ]
}

CHILD_TO_PARENT = {
    child: parent
    for parent, children in HIERARCHY.items()
    for child in children
}


def validate_results(results):
    """
    Remove invalid rows and enrich valid ones.
    """

    cleaned = []

    for item in results:

        # Must be a dictionary
        if not isinstance(item, dict):
            continue

        # Skip empty dictionaries: {}
        if not item:
            continue

        text = item.get("text", "")
        label = item.get("label", "")
        confidence_score = item.get("confidence_score", 0.0)

        # text and label must exist
        if not isinstance(text, str):
            continue

        if not isinstance(label, str):
            continue

        # Remove empty strings
        text = text.strip()
        label = label.strip()

        if text == "":
            continue

        if label == "":
            continue

        # Remove invalid labels
        if label not in VALID_LABELS:
            print(f"Skipping {label}")
            if label != "Not Related":
                continue

        cleaned.append({
            "text": text,
            "label": label,
            "parent_label": CHILD_TO_PARENT[label],
            "generator_model": f"{MODEL}/Reasoning = {REASONING}",
            "source": "real",
            "confidence_score": confidence_score,
        })

    return cleaned