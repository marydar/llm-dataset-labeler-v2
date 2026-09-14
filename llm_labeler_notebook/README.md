# LLM-Powered Topic Classification — Notebook Overview

## Purpose

This notebook implements an automated topic classification pipeline that uses a Large Language Model (LLM) to assign one of 35 predefined topic labels to text samples from a dataset. The classified results are saved as JSON files and evaluated through confusion matrix visualizations. The entire workflow is designed for iterative, resumable batch processing of a large dataset.

---

## 1. Dataset and Label Taxonomy

The notebook operates on a dataset hosted on Hugging Face (`maryamdar/topic_classification_dataset_gen`) containing 9,102 text samples, each with a ground-truth label. The dataset is loaded, shuffled once with a fixed random seed (42) for reproducibility, and saved locally as a permanent shuffled copy so that every subsequent run processes the data in the same order.

A taxonomy of 35 topic labels is defined, spanning domains such as technology (e.g., Web Development, Cybersecurity, AI/ML), healthcare (e.g., Clinical Diagnosis, Mental Health, Nutrition), finance (e.g., Payments, Investment, Corporate), science (e.g., Physics, Chemistry, Biology), sports (e.g., Team Sports, Individual Sports, Fitness), engineering (e.g., Civil Engineering, Mechanical Engineering), personal/lifestyle (e.g., Family, Travel, Personal), business (e.g., Marketing, Entrepreneurship, Management), law (e.g., Criminal Law, Family Law, Corporate Law, Civil Law), and arts/entertainment (e.g., Game, Film, Music, Literature, Painting). Each label has a detailed textual description specifying what it covers and what it excludes, serving as the semantic foundation for the LLM's classification decisions.

An older version of the label set (referred to as "old labels") is also retained for migration analysis — comparing how the dataset's original labels map onto the current 35-label taxonomy.

---

## 2. LLM Configuration

The notebook uses NVIDIA's Nemotron 3 Ultra 550B model (`nvidia/nemotron-3-ultra-550b-a55b`) accessed through NVIDIA's API endpoint. A dedicated `llm_call` function sends prompts to the model with specific generation parameters: temperature of 1, top-p of 0.95, max tokens of 4096, and a fixed seed of 42 for reproducibility. Reasoning/thinking mode is explicitly disabled.

---

## 3. Classification Prompt Design

Each batch of texts is sent to the LLM with a carefully structured prompt that includes:

- All 35 label descriptions, **shuffled in random order** on each run to mitigate positional bias (the model favoring labels that appear earlier in the list).
- Strict labeling rules: each text must receive exactly one label; predictions must be the exact label name with no modifications, no numbering, no explanations, and no confidence scores.
- A required JSON output format with a `"predictions"` key containing an ordered list of label strings matching the order of input texts.

This design ensures the model's output is machine-parseable and directly comparable to ground truth.

---

## 4. Batch Processing and Results Management

Texts are classified in configurable batches (default batch size: 10). The `classify_batch` function handles the full cycle:

1. Constructs the prompt with shuffled labels and the batch of texts.
2. Sends the request to the LLM.
3. Parses the JSON response.
4. Validates each prediction against the allowed label set using a multi-layered validation function that handles: exact matches, cleaned matches (stripping markdown formatting and trailing periods), case-insensitive matches, and substring extraction (e.g., extracting "Cybersecurity" from a response like "The answer is Cybersecurity").
5. If any validation step fails, the request is retried after a brief sleep.

Results are saved incrementally to a JSON file after every batch, making the process resumable. Before starting, the notebook checks which samples have already been classified and skips them, allowing interrupted runs to continue from where they left off.

Rate limiting is enforced by sleeping for 30 seconds after every 30 API requests to avoid hitting API limits.

---

## 5. Evaluation and Visualization

After classification, the notebook produces several types of confusion matrices:

### 5.1 Standard Confusion Matrix (New Labels)
A standard square confusion matrix comparing the ground-truth labels against the LLM's predicted labels across all 35 categories. This gives an overall picture of classification accuracy and shows which labels are most frequently confused with each other.

### 5.2 Old-to-New Label Migration Matrix (Counts)
A non-square matrix where rows represent the old/previous label taxonomy and columns represent the new 35-label taxonomy. This matrix uses raw counts and shows how texts originally labeled under the old system map to the new labels after LLM reclassification. It is useful for understanding the effect of label consolidation or splitting (e.g., when "Banking" and "Corporate Accounting" from the old set were merged into or replaced by different labels in the new set).

### 5.3 Old-to-New Label Migration Matrix (Percentages)
The same migration matrix as above but row-normalized to percentages. Each row sums to 100%, making it easy to see what proportion of texts from each old label were assigned to each new label. This is particularly useful for comparing label distributions across different old categories.

### 5.4 Misclassification Analysis
The notebook calculates the misclassification percentage per true label, identifying which topics the LLM struggles with most.

### 5.5 Generic Confusion Matrix Plotter
A flexible utility function that can generate a confusion matrix from any arbitrary JSON results file, enabling analysis of specific subsets of the data or different experimental runs.

All visualizations are saved as high-resolution PNG images (250 DPI) for inclusion in reports and presentations.

---

## 6. Key Design Decisions

- **Shuffling labels in the prompt**: Prevents the model from exhibiting positional bias toward labels that appear first or last in the list.
- **Incremental saving**: Every batch's results are written to disk immediately, preventing data loss from crashes or interruptions.
- **Resumable execution**: The notebook tracks which samples have been classified by their shuffled index and skips them on subsequent runs.
- **Multi-layered validation**: The prediction validator handles common LLM output quirks (extra formatting, casing variations, verbose responses) to maximize the yield of valid predictions.
- **Fixed random seeds**: Both the dataset shuffle and label shuffle use seed 42, ensuring full reproducibility across runs.
- **Old vs. new label comparison**: The migration matrices provide a structured way to evaluate how well the LLM handles label taxonomy changes, which is valuable when refining a classification scheme over time.

---

## 7. Output Files

| File | Description |
|---|---|
| `shuffled_dataset_excluded_none.json` | The permanently shuffled version of the full dataset with added shuffled indices |
| `classification_results_excluded_none.json` | All classification results (each entry contains the text, ground-truth label, and LLM prediction) |
| `confusion_matrix_all_labeled_data.png` | Standard confusion matrix across all 35 labels |
| `confusion_matrix_old_to_new_labels.png` | Old-to-new label migration matrix (raw counts) |
| `confusion_matrix_old_to_new_percentages.png` | Old-to-new label migration matrix (row-normalized percentages) |
