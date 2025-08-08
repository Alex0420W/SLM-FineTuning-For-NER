import csv
import os
import warnings
from typing import List, Dict

from datasets import load_dataset, Dataset, DatasetDict, DownloadMode

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
MAX_ROWS = 200                     # rows per dataset – change if you want a different number

# NOTE: every entry below is known to expose a `tokens` column and a tag column
# (`ner_tags` or `labels`).  If you add a new dataset, make sure it follows this
# convention, otherwise the script will skip it and warn you.
DATASETS = [
    # ------------------------------- classic benchmarks -------------------------------
    {"hf_name": "conll2003",                    "split": "train"},
    {"hf_name": "wnut_17",                      "split": "train"},
    {"hf_name": "wikiann",                      "split": "train", "config": "en"},
    # ------------------------------- additional public corpora -----------------------
    {"hf_name": "csebuetnlp/ner-wikiner-fr",    "split": "train"},
    {"hf_name": "tweebank-ner",                 "split": "train"},
    {"hf_name": "DFKI-SLT/cross_ner",           "split": "train"},
    {"hf_name": "bigbio/bc5cdr",                "split": "train"},
    {"hf_name": "tner/ontonotes5",              "split": "train"},
]

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------
def generate_instruction(entity_types: List[str], text: str) -> str:
    """Create the instruction string that will be written to the CSV."""
    entity_list = "\n".join([f"- {e}" for e in entity_types])
    return f"# Entities\n{entity_list}\n\n# Text\n{text}"


def find_tag_key(example: dict) -> str:
    """Return the first key that contains the integer‑ID tag list."""
    for key in ("ner_tags", "labels", "ner", "entity"):
        if key in example:
            return key
    raise KeyError("No recognised tag column (ner_tags/labels/ner/entity) found.")


def get_label_names(ds: Dataset) -> List[str]:
    """Extract the list of human‑readable label names from a dataset."""
    if "ner_tags" in ds.column_names:
        return ds.features["ner_tags"].feature.names
    if "labels" in ds.column_names:
        return ds.features["labels"].names
    # Fallback – inspect the first example's feature object
    first_example = ds[0]
    tag_key = find_tag_key(first_example)
    return ds.features[tag_key].feature.names


def clean_dataset_cache(hf_name: str) -> None:
    """
    Remove the local cache for a single dataset.
    Useful when you see “utf‑8 codec can't decode byte 0x8b” errors,
    which usually mean a corrupted gzip file in the cache.
    """
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    # The folder name is the HF repo with slashes replaced by three underscores
    broken_path = os.path.join(cache_dir, hf_name.replace("/", "___"))
    if os.path.isdir(broken_path):
        import shutil
        shutil.rmtree(broken_path)


def process_one_dataset(name: str, split: str, config: str | None = None) -> List[Dict]:
    """
    Load a dataset, sample ≤MAX_ROWS rows, and transform each row into the
    dict that will later become a CSV line.
    """
    # --------------------------------------------------------------
    # 1️⃣  Load (force a fresh download to avoid corrupted caches)
    # --------------------------------------------------------------
    try:
        # In the very rare case the cached files are corrupted we delete them
        # before the first download attempt.
        clean_dataset_cache(name)

        if config is not None:
            raw = load_dataset(
                name,
                config,
                split=split,
                trust_remote_code=True,
                download_mode=DownloadMode.FORCE_REDOWNLOAD,
            )
        else:
            raw = load_dataset(
                name,
                split=split,
                trust_remote_code=True,
                download_mode=DownloadMode.FORCE_REDOWNLOAD,
            )
    except Exception as exc:
        raise RuntimeError(f"Could not load dataset {name} (split={split}): {exc}")

    # Occasionally HF returns a DatasetDict even when we ask for a specific split;
    # we just pick the first dataset it contains.
    if isinstance(raw, DatasetDict):
        raw = list(raw.values())[0]

    # --------------------------------------------------------------
    # 2️⃣  Sanity‑check required columns
    # --------------------------------------------------------------
    if "tokens" not in raw.column_names:
        raise RuntimeError(f"`tokens` column missing in {name}")

    tag_key = find_tag_key(raw[0])                 # raises if none found
    label_names = get_label_names(raw)             # e.g. ["O","B-PER","I-PER",...]

    # --------------------------------------------------------------
    # 3️⃣  Sample (shuffle + slice) – deterministic because of the seed
    # --------------------------------------------------------------
    if len(raw) > MAX_ROWS:
        raw = raw.shuffle(seed=42).select(range(MAX_ROWS))
    else:
        raw = raw.shuffle(seed=42)                 # keep everything, still shuffled

    # --------------------------------------------------------------
    # 4️⃣  Build CSV rows
    # --------------------------------------------------------------
    rows = []
    for item in raw:
        tokens = item["tokens"]
        tag_ids = item[tag_key]

        # Build a dict: entity_type → list of tokens
        entities: Dict[str, List[str]] = {}
        for token, tag_id in zip(tokens, tag_ids):
            tag_name = label_names[tag_id]
            if tag_name != "O":                     # ignore the outside label
                entities.setdefault(tag_name, []).append(token)

        # Drop empty entity types (should not happen, but safe)
        entities = {k: v for k, v in entities.items() if v}

        rows.append(
            {
                "instruction": generate_instruction(list(entities.keys()), " ".join(tokens)),
                "expected_output": str(entities),               # kept as string for compatibility
                "source": name.upper(),
            }
        )
    return rows


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main() -> None:
    csv_path = "generated_sample.csv"
    all_rows: List[Dict] = []

    for ds in DATASETS:
        name = ds["hf_name"]
        split = ds["split"]
        config = ds.get("config")
        try:
            all_rows.extend(process_one_dataset(name, split, config))
        except Exception as exc:
            warnings.warn(
                f"❌  Failed to process dataset {name}"
                f"{' (config=' + config + ')' if config else ''}"
                f": {exc}. This source will be skipped."
            )
            continue

    # ------------------------------------------------------------------
    # Write the master CSV
    # ------------------------------------------------------------------
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["No", "Instruction", "Expected Output", "Source"])
        writer.writeheader()
        for i, row in enumerate(all_rows, start=1):
            writer.writerow(
                {
                    "No": i,
                    "Instruction": row["instruction"],
                    "Expected Output": row["expected_output"],
                    "Source": row["source"],
                }
            )

    print(f"✅  Successfully generated {os.path.abspath(csv_path)}")
    print(
        f"   Total rows written: {len(all_rows)} "
        f"(≈ {len(DATASETS)} datasets × {MAX_ROWS} rows each)"
    )


if __name__ == "__main__":
    main()