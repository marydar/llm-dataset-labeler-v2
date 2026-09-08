# LLM Dataset Labeler

An LLM-powered pipeline that takes real-world text prompts from HuggingFace datasets, classifies them into one of **35 topic categories** using an LLM with a 3-attempt consensus mechanism, and produces a labeled dataset uploaded to HuggingFace Hub.

---

## Quick Start

### 1. Install dependencies

```bash
pip install openai datasets python-dotenv tqdm
```

### 2. Set up your API keys

Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
OPENROUTER_API_KEY=your_openrouter_key_here  # optional, only if using OpenRouter
```

### 3. Configure

Edit `config.py` (see [Configuration Reference](#configuration-reference) below).

### 4. Run

```bash
python main.py
```

The pipeline will load the source dataset, classify texts in batches, save checkpoints as it goes, and push the final labeled dataset to HuggingFace Hub.

---

## Project Structure

```
llm-dataset-labeler_v2/
├── main.py                 # Entry point - orchestrates the full pipeline
├── config.py               # All configuration in one place
├── labels.json             # 35 topic labels with detailed descriptions
├── .env                    # API keys (not committed to git)
│
├── src/
│   ├── classifier_three_try.py   # 3-attempt consensus classifier (ACTIVE)
│   ├── classifier.py             # Single-attempt classifier (alternative)
│   ├── llm.py                    # LLM client - Nvidia API (ACTIVE)
│   ├── llm_openrouter.py         # LLM client - OpenRouter (alternative)
│   ├── shuffle_loader.py         # Dataset loader with shuffling + caching (ACTIVE)
│   ├── loader.py                 # Basic dataset loader (alternative)
│   ├── loader3.py                # Specialized loader for LMSYS conversations
│   ├── validator.py              # Label validation + metadata enrichment
│   ├── exporter.py               # Save to disk + push to HuggingFace Hub
│   ├── remove_checkpoint_duplicates.py  # Skip already-labeled texts
│   └── utils.py                  # JSON I/O helpers + random delay
│
├── extra/                  # Standalone utility scripts (one-off operations)
│   ├── addcols.py          # Add a column to a HuggingFace dataset
│   ├── addtoall.py         # Merge datasets from two HF repos
│   ├── addtoallindexed.py  # Merge datasets from a specific index onward
│   ├── dup.py              # Find duplicate texts in a dataset
│   └── removerows.py       # Filter out specific rows from a dataset
│
├── output/
│   ├── checkpoint.json     # Accumulated labeled results (survives restarts)
│   └── labeled_dataset/    # HuggingFace-format dataset saved to disk
│
└── dataset_cache/          # Cached shuffled source dataset (auto-created)
```

---

## How It Works (Pipeline Flow)

```
1. CONFIGURE (config.py)
   Loads API keys, sets LLM model, batch size, rate limits, source/output paths.
        |
2. LOAD LABELS (labels.json)
   35 topic labels + "Not Related", each with detailed scope and exclusion rules.
        |
3. LOAD SOURCE DATASET (src/shuffle_loader.py)
   Downloads from HuggingFace, shuffles (seed=42), caches to disk for reuse.
   Selects the configured row range and extracts text from specified columns.
        |
4. DEDUPLICATE (src/remove_checkpoint_duplicates.py)
   Removes any texts already present in checkpoint.json (avoids re-labeling).
        |
5. CLASSIFY IN BATCHES (main.py loop)
   For each batch of texts:
     a. Build prompt with label definitions + 17 classification rules
     b. Run LLM 3 times with different random seeds
     c. If all 3 agree -> keep label. If any disagree -> mark "Ambiguous"
     d. Save checkpoint after every batch
     e. Sleep periodically for rate limiting
        |
6. EXPORT (src/exporter.py)
   Save to disk as Arrow dataset, then push to HuggingFace Hub
   (merges with existing data, deduplicates by text).
```

### Output Schema

Each labeled row looks like:

```json
{
  "text": "the original user prompt",
  "label": "Desktop & Mobile & Web Development",
  "parent_label": "Programming / Technology",
  "generator_model": "deepseek-ai/deepseek-v4-pro-0813/Reasoning = False",
  "source": "real",
  "confidence_score": 0.9,
  "source_dataset": "lmsys/lmsys-chat-1m"
}
```

---

## Configuration Reference

All settings live in `config.py`. Here's what each one does:

### LLM Provider

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | from `.env` | API key for Nvidia (used when `BASE_URL` points to Nvidia) |
| `BASE_URL` | `https://integrate.api.nvidia.com/v1` | API endpoint. Change to `https://openrouter.ai/api/v1` for OpenRouter |
| `MODEL` | `deepseek-ai/deepseek-v4-pro-0813` | The LLM model to use. Other options: `nvidia/nemotron-3-ultra-550b-a55b`, `moonshotai/kimi-k3` |
| `REASONING` | `False` | Enable reasoning/thinking mode (if the model supports it) |

### Generation

| Variable | Default | Description |
|---|---|---|
| `BATCH_SIZE` | `10` | Number of texts sent to the LLM in each call |
| `MAX_TOKENS` | `1500` | Maximum tokens the LLM can generate per response |
| `TEMPERATURE` | `0.2` | Controls randomness. Lower = more deterministic |

### Rate Limiting

| Variable | Default | Description |
|---|---|---|
| `REQUESTS_BEFORE_SLEEP` | `30` | Pause after this many batches |
| `SLEEP_SECONDS` | `30` | How long to sleep during the pause |
| `REQUEST_DELAY_MIN` | `4` | Minimum random delay between requests (seconds) |
| `REQUEST_DELAY_MAX` | `8` | Maximum random delay between requests (seconds) |

### Output

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT_PATH` | `"output/checkpoint.json"` | Where checkpoint data is saved. Allows resuming interrupted runs |
| `FINAL_DATASET_PATH` | `"output/labeled_dataset"` | Where the final HuggingFace Arrow dataset is saved locally |
| `HF_REPO` | `"maryamdar/topic-classification-dataset-real-labeled"` | HuggingFace Hub repo to push the final dataset to |

### Source Dataset

| Variable | Default | Description |
|---|---|---|
| `SOURCE_DATASET` | `"lmsys/lmsys-chat-1m"` | HuggingFace dataset to load from. Can be any public/private HF dataset |
| `CONFIG_NAME` | `None` | Dataset config/subset name (if the dataset has multiple configs) |
| `TEXT_COLUMNS` | `["text"]` | Column(s) containing the text to classify. Multiple columns are concatenated |
| `SPLIT` | `"train"` | Which split of the dataset to use (`train`, `test`, `validation`) |
| `START_IDX` | `100` | Start index (inclusive) of the dataset range to process |
| `END_IDX` | `1100` | End index (exclusive) of the dataset range to process |

---

## How to Customize

### Use a different source dataset

1. Find a dataset on HuggingFace Hub
2. In `config.py`, update:
   ```python
   SOURCE_DATASET = "username/dataset-name"
   TEXT_COLUMNS = ["column_name"]  # check the dataset's column names
   SPLIT = "train"
   START_IDX = 0
   END_IDX = 500  # how many texts to label
   ```
3. In `main.py`, make sure the `load_text_dataset()` call uses `SOURCE_DATASET` (not a hardcoded name)

### Use OpenRouter instead of Nvidia

1. In `config.py`, comment out the Nvidia section and uncomment the OpenRouter section:
   ```python
   # Nvidia (comment out)
   # NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
   # BASE_URL = "https://integrate.api.nvidia.com/v1"
   # MODEL = "deepseek-ai/deepseek-v4-pro-0813"

   # OpenRouter (uncomment)
   OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
   BASE_URL = "https://openrouter.ai/api/v1"
   MODEL = "google/gemini-2.5-flash"
   ```
2. In `src/classifier_three_try.py`, change the import from `llm` to `llm_openrouter`
3. Update your `.env` with the correct API key

### Add or modify labels

Edit `labels.json`. Each entry is:

```json
{
  "Label Name": "Covers: what this label includes.\nExcludes: what it does not include."
}
```

The labels here are passed to the LLM as classification categories.

### Resume an interrupted run

Just run `python main.py` again. The pipeline reads `output/checkpoint.json` and skips already-labeled texts automatically.

---

## The 35 Labels

| Domain | Labels |
|---|---|
| **Technology** | Desktop & Mobile & Web Development, Cybersecurity, AI / Machine Learning / Data Science, Infrastructure (DevOps, Cloud, Databases, Networking) |
| **Medical** | Clinical Diagnosis Treatment & Surgery, Medication & Pharmacology, Mental Health, Healthcare Organizations System Hospitals, Nutrition |
| **Finance** | Payments & Personal Budgeting, Investment Markets & Cryptocurrency, Corporate |
| **Science** | Physics Mathematics, Chemistry, Biology |
| **Sports** | Team Sports, Individual Sports, Fitness |
| **Engineering** | Civil Structural & Architecture, Mechanical & Electrical Engineering |
| **Personal** | Family & Relationships, Personal, Travel |
| **Business** | Marketing & Sales, Entrepreneurship & Startups, Management Strategy & Human Resources |
| **Law** | Criminal Law, Family Law, Corporate Law, Civil Law |
| **Art & Entertainment** | Game, Film, Music, Literature, Painting |

---

## Extra Scripts

The `extra/` folder contains one-off utility scripts. These are standalone and not part of the main pipeline:

| Script | What it does |
|---|---|
| `addcols.py` | Adds a new column to a HuggingFace dataset |
| `addtoall.py` | Merges rows from one HF dataset into another |
| `addtoallindexed.py` | Same as above, but starts from a specific index |
| `dup.py` | Finds and reports duplicate texts in a dataset |
| `removerows.py` | Removes rows matching specific conditions from a dataset |

---
