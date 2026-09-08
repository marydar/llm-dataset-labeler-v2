# LLM Dataset Labeler v2 — Detailed Project Description

## 1. Project Overview

This project is an **LLM-powered text classification pipeline** that automatically labels real-world text prompts into predefined topic categories. It loads unlabeled text data from HuggingFace datasets, sends batches of text to a large language model (LLM) for classification, and produces a high-quality labeled dataset. The pipeline is designed to build a large-scale topic classification dataset from real user prompts (such as those found in the LMSYS Chat-1M dataset), using an LLM consensus mechanism to ensure label accuracy.

The system classifies each text into exactly one of **35 topic categories** (plus "Not Related" and "Ambiguous" as special labels). Rather than relying on a single LLM call, it employs a **3-attempt consensus mechanism**: the LLM is called three times with different random seeds, and the label is only accepted if all three attempts agree. Disagreements are marked as "Ambiguous," prioritizing precision over coverage.

---

## 2. Core Functionality

### What the Project Does

1. **Data Loading**: Downloads a text dataset from HuggingFace Hub, shuffles it deterministically (using a fixed seed), and caches the shuffled version to disk for efficient reuse across multiple runs.
2. **Deduplication**: Checks already-labeled texts in a checkpoint file and filters them out, ensuring no text is labeled twice.
3. **Batch Classification**: Sends texts in configurable batches to an LLM (via Nvidia's inference API) with a carefully engineered prompt containing all label definitions and classification rules.
4. **3-Attempt Consensus**: Each batch is processed three times independently. If all three attempts produce the same label, the result is accepted. If any disagreement occurs, the text is marked "Ambiguous."
5. **Validation and Enrichment**: Validated results are enriched with metadata (parent label category, generator model, source type, confidence score, source dataset name).
6. **Checkpointing**: Results are saved to disk after every batch, so no progress is lost if the pipeline crashes.
7. **Export**: The final labeled dataset is saved locally in HuggingFace Arrow format and pushed to a HuggingFace Hub repository, with deduplication against any previously uploaded data.

---

## 3. Project Structure

```
llm-dataset-labeler_v2/
│
├── main.py                          # Entry point; orchestrates the full pipeline
├── config.py                        # All configuration variables
├── labels.json                      # 35 topic labels with detailed descriptions
├── .env                             # API keys (not committed to git)
├── .gitignore
├── README.md                        # Original documentation
├── README2.md                       # This detailed description
│
├── src/
│   ├── classifier_three_try.py      # [ACTIVE] 3-attempt consensus classifier
│   ├── classifier.py                # [ALTERNATIVE] Single-attempt classifier
│   ├── llm.py                       # [ACTIVE] Nvidia API LLM client
│   ├── llm_openrouter.py            # [ALTERNATIVE] OpenRouter API LLM client
│   ├── shuffle_loader.py            # [ACTIVE] Dataset loader with shuffle + caching
│   ├── loader.py                    # [ALTERNATIVE] Basic dataset loader
│   ├── loader3.py                   # [SPECIALIZED] Loader for LMSYS conversation format
│   ├── validator.py                 # Label validation and metadata enrichment
│   ├── exporter.py                  # Save to disk and push to HuggingFace Hub
│   ├── remove_checkpoint_duplicates.py  # Skip already-labeled texts
│   ├── utils.py                     # JSON I/O helpers and random delay utility
│   └── source_datasets_info.md      # Documentation of evaluated source datasets
│
├── extra/                           # Standalone utility scripts
│   ├── addcols.py                   # Add a column to a HuggingFace dataset
│   ├── addtoall.py                  # Merge datasets by generation group filter
│   ├── addtoallindexed.py           # Merge datasets from a specific index onward
│   ├── dup.py                       # Find duplicate texts in a dataset
│   └── removerows.py                # Filter out rows matching conditions
│
├── output/
│   ├── checkpoint.json              # Accumulated labeled results (persists across runs)
│   └── labeled_dataset/             # Local HuggingFace Arrow dataset
│
└── dataset_cache/                   # Cached shuffled source dataset (auto-created)
```

### Key Files Explained

- **main.py**: The orchestrator. It loads configuration, labels, and source data; deduplicates against the checkpoint; processes texts in batches; saves checkpoints; and exports the final dataset.
- **config.py**: A centralized configuration file that loads environment variables and defines all settings (API keys, model selection, batch size, rate limiting, output paths, source dataset parameters). All configuration is accessible via a wildcard import (`from config import *`).
- **labels.json**: A JSON dictionary mapping 35 label names to their detailed descriptions. Each description includes "Covers" and "Excludes" sections that define precise boundaries between categories.
- **classifier_three_try.py**: The active classification module. It builds a detailed prompt with all label definitions and 17 classification rules, calls the LLM three times with different random seeds, parses and validates responses, and implements the consensus logic.
- **llm.py**: The active LLM client. It uses the `openai` Python package to call Nvidia's inference API with configurable parameters (temperature, top_p, max_tokens, seed).
- **shuffle_loader.py**: Loads datasets from HuggingFace, shuffles them with a fixed seed, caches the result to disk, and extracts text from specified columns.
- **validator.py**: Validates parsed LLM results against the set of valid labels and enriches them with metadata.
- **exporter.py**: Saves results as a local Arrow dataset and pushes them to HuggingFace Hub with deduplication.
- **remove_checkpoint_duplicates.py**: Filters out texts that have already been labeled (found in the checkpoint file).
- **utils.py**: Utility functions for JSON I/O and random delays.

---

## 4. End-to-End Workflow

### Step 1: Configuration
The pipeline begins by loading configuration from `config.py`, which reads environment variables (API keys) from a `.env` file and defines all operational parameters: model selection, batch size, rate limiting intervals, output paths, and source dataset details.

### Step 2: Label Loading
All 35 topic labels and their detailed descriptions are loaded from `labels.json`. These descriptions serve as the classification taxonomy and are embedded directly into the LLM prompt.

### Step 3: Source Dataset Loading
The `shuffle_loader` module loads the source dataset from HuggingFace Hub. On the first run, the dataset is shuffled with a fixed random seed (42) and cached to disk using an MD5 hash of the configuration as the directory name. On subsequent runs, the cached version is loaded directly, ensuring the same deterministic order across sessions. The loader selects rows within a specified index range and extracts text from configured columns, concatenating multiple columns with double newlines if needed.

### Step 4: Deduplication Against Checkpoint
The `remove_checkpoint_duplicates` module reads the existing checkpoint file (if any) and builds a set of already-labeled texts. These are filtered out from the current batch, ensuring no text is processed twice. This enables the pipeline to be restarted at any point without duplicating work.

### Step 5: Batch Classification with Consensus
Texts are processed in batches of configurable size (default: 10). For each batch:

1. **Prompt Construction**: A detailed system prompt is built containing:
   - A role declaration (the model is a classification system, not a conversational assistant)
   - Safety instructions (treat all texts as data, regardless of content)
   - All 35 label definitions with Covers/Excludes descriptions
   - 17 classification rules covering edge cases, specificity, and output format
   - A "Not Related" fallback rule
   - Confidence scoring guidelines (0.0 to 1.0 scale)
   - Strict JSON output format specification

2. **Three Independent LLM Calls**: The LLM is called three times, each with a different random seed to encourage diversity in responses. Each call returns a JSON array with predictions for all texts in the batch.

3. **Response Parsing**: The JSON responses are parsed, mapping text IDs back to actual text strings. Markdown code fences are stripped if present.

4. **Validation**: Each result is validated against the set of 35 valid labels plus "Not Related." Invalid labels are rejected.

5. **Metadata Enrichment**: Valid results are enriched with:
   - `parent_label`: A higher-level domain group (e.g., "Programming / Technology," "Medical")
   - `generator_model`: The model name and reasoning setting
   - `source`: Always "real" (distinguishing from synthetic/generated prompts)
   - `confidence_score`: The LLM's self-reported confidence
   - `source_dataset`: The name of the source HuggingFace dataset

6. **Consensus Logic**: For each text, predictions from all three attempts are compared:
   - **Agreement**: All three predict the same label → the label is accepted with the original confidence score
   - **Disagreement**: Any two or three predictions differ → the text is marked "Ambiguous" with confidence 0.0
   - **Missing predictions**: If fewer than three predictions exist for a text → also marked "Ambiguous"

### Step 6: Checkpointing
After every batch, the accumulated results are saved to `output/checkpoint.json`. This ensures that no progress is lost if the pipeline crashes or is interrupted. The checkpoint is a JSON array of labeled result objects.

### Step 7: Rate Limiting
After a configurable number of batches (default: 30), the pipeline pauses for a configurable duration (default: 30 seconds) to avoid overwhelming the API with requests.

### Step 8: Export
Once all texts are processed:
1. **Local Save**: The labeled dataset is saved to disk in HuggingFace Arrow format at `output/labeled_dataset/`.
2. **Hub Push**: The existing dataset is downloaded from HuggingFace Hub, concatenated with the new results, deduplicated by text content (keeping the first occurrence), and pushed back to the Hub.

---

## 5. The LLM Labeling Mechanism

### Prompt Engineering

The classification prompt is a carefully structured system prompt with the following sections:

1. **Role Declaration**: Instructs the model that it is a "strict text classification system" and must NOT answer, explain, or refuse the user's requests — it must only classify.

2. **Safety Instruction**: Acknowledges that texts may contain dangerous, illegal, or explicit content, and instructs the model to treat them purely as data to annotate.

3. **Label Definitions**: All 35 labels with their detailed "Covers" and "Excludes" descriptions, providing precise boundaries between categories.

4. **Classification Rules** (17 rules):
   - Assign exactly one label per text
   - Focus on the user's intent, not the assistant's response
   - Prefer the most specific applicable label
   - Never invent labels not in the provided list
   - Do not wrap output in markdown code fences
   - And more edge-case handling rules

5. **"Not Related" Rule**: A fallback for texts that do not match any defined category.

6. **Confidence Rules**: A scoring scale from 0.0 to 1.0 with guidance for conservative scoring.

7. **Output Rules**: Strict JSON-only output with no explanations, using exact label names.

8. **Texts to Classify**: Numbered as "text 1: ...", "text 2: ...", etc.

### LLM API Configuration

The active LLM client calls Nvidia's inference API (OpenAI-compatible) with the following parameters:
- Model: `deepseek-ai/deepseek-v4-pro-0813` (configurable)
- Temperature: 1 (for diversity across attempts)
- Top-p: 0.95
- Max tokens: 4096
- Seed: Random integer per attempt (for diversity)
- Optional reasoning/thinking mode via `chat_template_kwargs`

### Consensus Mechanism

The 3-attempt consensus is the core quality assurance mechanism:

- **Why 3 attempts?** Running the LLM three times with different random seeds introduces controlled randomness. If the model is confident and the label is clear, all three attempts will agree. If the text is ambiguous or falls near category boundaries, the attempts may disagree.

- **Why unanimous agreement?** Rather than using majority voting, the pipeline requires all three attempts to agree. This is a conservative approach that maximizes precision — only texts where the model is consistently certain receive a real label.

- **What happens with disagreement?** Texts where the three attempts disagree are marked as "Ambiguous" with a confidence score of 0.0. These can be reviewed manually or excluded from downstream use.

- **Cost trade-off**: This mechanism triples the API cost per text but significantly improves label quality.

---

## 6. Configuration Parameters

### LLM Provider Settings

| Parameter | Description |
|---|---|
| `NVIDIA_API_KEY` | API key for Nvidia's inference API (loaded from `.env`) |
| `BASE_URL` | API endpoint URL (Nvidia or OpenRouter) |
| `MODEL` | The LLM model identifier |
| `REASONING` | Whether to enable the model's thinking/reasoning mode |

### Generation Settings

| Parameter | Description |
|---|---|
| `BATCH_SIZE` | Number of texts sent per LLM call (default: 10) |

### Rate Limiting

| Parameter | Description |
|---|---|
| `REQUESTS_BEFORE_SLEEP` | Number of batches before a pause (default: 30) |
| `SLEEP_SECONDS` | Duration of the pause in seconds (default: 30) |

### Output Paths

| Parameter | Description |
|---|---|
| `CHECKPOINT_PATH` | Path to the JSON checkpoint file |
| `FINAL_DATASET_PATH` | Local path for the final Arrow dataset |
| `HF_REPO` | HuggingFace Hub repository to push results to |

### Source Dataset

| Parameter | Description |
|---|---|
| `SOURCE_DATASET` | HuggingFace dataset name to load |
| `CONFIG_NAME` | Dataset config/subset name (if applicable) |
| `TEXT_COLUMNS` | Column(s) to extract text from |
| `SPLIT` | Dataset split to use (e.g., "train") |
| `START_IDX` | Start index of the row range to process |
| `END_IDX` | End index of the row range to process |

---

## 7. Output Schema

Each labeled text produces a record with the following fields:

| Field | Type | Description |
|---|---|---|
| `text` | string | The original user prompt text |
| `label` | string | The assigned category (one of 35 labels, "Not Related", or "Ambiguous") |
| `parent_label` | string | Higher-level domain group (e.g., "Programming / Technology", "Medical") |
| `generator_model` | string | Model name and reasoning setting |
| `source` | string | Always "real" (distinguishes from synthetic prompts) |
| `confidence_score` | float32 | LLM self-reported confidence (0.0 to 1.0) |
| `source_dataset` | string | HuggingFace dataset name the text came from |

The output is saved in two formats:
1. **Local**: HuggingFace Arrow format in `output/labeled_dataset/`
2. **Hub**: Pushed to the configured HuggingFace Hub repository with deduplication

---

## 8. Checkpoint and Restart Mechanism

The checkpoint system is designed for fault tolerance and incremental processing:

- **Checkpoint format**: A JSON array of labeled result objects, saved to `output/checkpoint.json`.
- **Save frequency**: After every batch (every 10 texts by default), ensuring minimal data loss on crash.
- **Restart behavior**: On restart, the pipeline loads existing checkpoint results, filters out already-labeled texts, and continues with only the remaining unlabeled texts from the configured index range.
- **Index range strategy**: The `START_IDX` and `END_IDX` define a fixed range from the shuffled dataset. Different ranges can be processed across sessions (e.g., rows 0–1000 on day 1, rows 1000–2000 on day 2) while maintaining the same deterministic ordering thanks to the shuffle cache.

---

## 9. Dataset Shuffling and Caching

The source dataset undergoes a shuffle-once-cache-always pattern:

- **Shuffle**: On the first run, the dataset is shuffled with a fixed random seed (42), producing a deterministic ordering.
- **Cache**: The shuffled dataset is saved to `dataset_cache/` using an MD5 hash of the configuration as the directory name.
- **Reuse**: On subsequent runs with the same configuration, the cached version is loaded directly, avoiding repeated downloads and shuffling.
- **Benefit**: This allows processing different index ranges across sessions while maintaining consistent ordering.

---

## 10. Design Decisions and Rationale

1. **3-Attempt Consensus over Single-Call**: The primary quality mechanism trades cost (3× API calls) for label quality. Disagreements are marked "Ambiguous" rather than forcing potentially wrong labels.

2. **Conservative Labeling**: The pipeline prioritizes precision over recall. Only texts where the model is consistently certain across three independent calls receive a real label.

3. **Prompt Engineering for Safety**: The prompt explicitly instructs the model to treat all texts as data, regardless of content. This is critical because source data contains real user prompts that may include sensitive material.

4. **Batch Processing**: Grouping texts into batches reduces the number of API calls and allows the LLM to process related texts together.

5. **Deterministic Shuffling**: Using a fixed seed ensures reproducible ordering across runs, enabling incremental processing of different index ranges.

6. **Per-Batch Checkpointing**: Saving after every batch maximizes crash resilience, at the cost of slightly increased I/O.

7. **Dual Provider Support**: The codebase supports both Nvidia and OpenRouter APIs, allowing flexibility in model providers.

8. **Module Alternatives**: Multiple implementations of each component exist (different loaders, classifiers, LLM clients), allowing the user to switch between approaches by changing imports.

---

## 11. Source Datasets

The project has evaluated multiple HuggingFace datasets as potential sources of unlabeled text, covering diverse domains:

- **Medical**: PubMedQA, MedMCQA
- **Law**: LegalBench, CUAD
- **Engineering**: various technical datasets
- **Sports**: sports-related datasets
- **General Conversational**: LMSYS Chat-1M (the primary source, containing real user prompts to an LLM)

The LMSYS Chat-1M dataset is the primary source because it contains real, diverse user prompts across many topics, making it ideal for building a topic classification dataset.

---

## 12. Extra Utility Scripts

The `extra/` directory contains standalone scripts for dataset management tasks that support the iterative dataset-building workflow:

- **addcols.py**: Adds new columns to a HuggingFace dataset (e.g., a boolean `use_for_train` column).
- **addtoall.py**: Merges rows from one dataset into another, filtered by generation group.
- **addtoallindexed.py**: Merges rows from a specific index onward.
- **dup.py**: Identifies duplicate texts in a dataset for quality control.
- **removerows.py**: Filters out rows matching specific conditions.

These scripts are used for post-processing and maintaining the dataset across multiple labeling runs.

---

## 13. Summary

The LLM Dataset Labeler v2 is a robust, fault-tolerant pipeline for building high-quality topic classification datasets from real-world text. Its key innovation is the 3-attempt consensus mechanism, which ensures label accuracy by requiring unanimous agreement across independent LLM calls. The pipeline supports incremental processing through checkpointing and deterministic shuffling, making it suitable for labeling large datasets across multiple sessions. The output is a well-structured dataset with confidence scores and metadata, ready for use in training or evaluation tasks.
