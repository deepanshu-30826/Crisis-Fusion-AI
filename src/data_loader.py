"""Data loading, schema inspection, and label filtering for crisis reports."""

import glob
import gzip
import json
import os
import pprint
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# Schema keys for labels.json (adjust these if real TREC-IS schema differs)
LABEL_ID_KEY: str = "tweet_id"
LABEL_CATEGORY_KEY: str = "categories"
LABEL_PRIORITY_KEY: str = "priority"


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load project configuration from YAML file."""
    p = Path(config_path)
    if not p.exists():
        # Fallback to search in parent directories if run from subdirectories
        for parent in p.resolve().parents:
            candidate = parent / "config.yaml"
            if candidate.exists():
                p = candidate
                break
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def inspect_label_schema(path: str, n: int = 2) -> None:
    """Pretty-print the first n top-level entries of the labels file.

    Helps confirm the schema keys before performing joins.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"[inspect_label_schema] File does not exist: {path}")
        return

    print(f"=== Inspecting schema for: {path} (first {n} entries) ===")
    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            sample = data[:n]
            pprint.pprint(sample, indent=2)
        elif isinstance(data, dict):
            sample_keys = list(data.keys())[:n]
            sample = {k: data[k] for k in sample_keys}
            pprint.pprint(sample, indent=2)
        else:
            print(f"Unexpected JSON root type: {type(data)}")
    except Exception as e:
        print(f"Error inspecting schema: {e}")


def load_tweets(glob_pattern: str) -> pd.DataFrame:
    """Load tweets from one or more gzipped JSON files.

    Each file contains one JSON tweet per line with fields:
    - 'id_str' or 'id'
    - 'full_text' or 'text'

    Returns:
        pd.DataFrame with columns: ['tweet_id', 'text', 'source_file']
    """
    records: List[Dict[str, Any]] = []
    matched_files = glob.glob(glob_pattern)

    if not matched_files:
        print(f"[load_tweets] Warning: No files matched pattern '{glob_pattern}'")
        return pd.DataFrame(columns=["tweet_id", "text", "source_file"])

    for file_path in matched_files:
        filename = Path(file_path).name
        opener = gzip.open if file_path.endswith(".gz") else open
        try:
            with opener(file_path, "rt", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        tweet_json = json.loads(line)
                        tweet_id = tweet_json.get("id_str") or tweet_json.get("id")
                        text = tweet_json.get("full_text") or tweet_json.get("text")

                        if tweet_id is not None and text is not None:
                            records.append({
                                "tweet_id": str(tweet_id).strip(),
                                "text": str(text),
                                "source_file": filename
                            })
                    except json.JSONDecodeError:
                        continue
        except Exception as err:
            print(f"[load_tweets] Error reading {file_path}: {err}")

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["tweet_id", "text", "source_file"])
    # Deduplicate by tweet_id if present in multiple files
    df = df.drop_duplicates(subset=["tweet_id"]).reset_index(drop=True)
    return df


def load_labels(path: str) -> pd.DataFrame:
    """Load TREC Incident Streams labels file.

    Supports:
    - List of dicts: [{'tweet_id': ..., 'categories': ..., 'priority': ...}]
    - Dict of dicts: {'tweet_id_1': {'categories': ..., 'priority': ...}}

    Returns:
        pd.DataFrame with columns: ['tweet_id', 'categories', 'priority']
    """
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"[load_labels] Warning: Labels file not found at '{path}'")
        return pd.DataFrame(columns=["tweet_id", "categories", "priority"])

    with open(path_obj, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                tweet_id = item.get(LABEL_ID_KEY) or item.get("id") or item.get("id_str")
                cats = item.get(LABEL_CATEGORY_KEY)
                prio = item.get(LABEL_PRIORITY_KEY)
                if tweet_id is not None:
                    rec = {
                        "tweet_id": str(tweet_id).strip(),
                        "categories": cats,
                        "priority": prio
                    }
                    if "cluster_id" in item:
                        rec["cluster_id"] = item["cluster_id"]
                    elif "event_id" in item:
                        rec["cluster_id"] = item["event_id"]
                    records.append(rec)
    elif isinstance(data, dict):
        # Case where root is a dict keyed by tweet_id
        for k, v in data.items():
            if isinstance(v, dict):
                tweet_id = v.get(LABEL_ID_KEY, k)
                cats = v.get(LABEL_CATEGORY_KEY)
                prio = v.get(LABEL_PRIORITY_KEY)
                rec = {
                    "tweet_id": str(tweet_id).strip(),
                    "categories": cats,
                    "priority": prio
                }
                if "cluster_id" in v:
                    rec["cluster_id"] = v["cluster_id"]
                elif "event_id" in v:
                    rec["cluster_id"] = v["event_id"]
                records.append(rec)
            else:
                records.append({
                    "tweet_id": str(k).strip(),
                    "categories": v,
                    "priority": None
                })

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["tweet_id", "categories", "priority"])
    return df.drop_duplicates(subset=["tweet_id"]).reset_index(drop=True)


def build_filtered_dataset(
    tweets_glob: Optional[str] = None,
    labels_path: Optional[str] = None,
    output_path: Optional[str] = None,
    config_path: str = "config.yaml"
) -> pd.DataFrame:
    """Load tweets and labels, perform inner join, filter nulls, and save to CSV.

    Per the challenge requirements:
    - Inner-join on tweet_id
    - Keep only rows where BOTH categories and priority are non-null
    - Print row counts at each stage (raw tweets, merged, filtered)
    - Save result to data/processed/filtered_reports.csv

    Returns:
        Filtered pd.DataFrame
    """
    config = load_config(config_path)
    paths_cfg = config.get("paths", {})

    tweets_glob = tweets_glob or paths_cfg.get("raw_data_glob", "data/raw/*.json.gz")
    labels_path = labels_path or paths_cfg.get("raw_labels_path", "data/raw/labels.json")
    output_path = output_path or paths_cfg.get("processed_reports_path", "data/processed/filtered_reports.csv")

    print(f"[build_filtered_dataset] Loading tweets from '{tweets_glob}'...")
    df_tweets = load_tweets(tweets_glob)
    print(f"  Stage 1: Raw tweets loaded: {len(df_tweets):,} rows")

    print(f"[build_filtered_dataset] Loading labels from '{labels_path}'...")
    df_labels = load_labels(labels_path)
    print(f"  Stage 1: Raw labels loaded: {len(df_labels):,} rows")

    if df_tweets.empty or df_labels.empty:
        print("  Warning: Empty tweets or labels. Result will be empty.")
        merged_df = pd.DataFrame(columns=["tweet_id", "text", "source_file", "categories", "priority"])
        filtered_df = merged_df
    else:
        merged_df = pd.merge(df_tweets, df_labels, on="tweet_id", how="inner")
        print(f"  Stage 2: Merged dataset: {len(merged_df):,} rows")

        # Required-label filter: BOTH categories and priority must be non-null and non-empty
        def is_valid_label(val: Any) -> bool:
            if val is None or pd.isna(val):
                return False
            if isinstance(val, (list, tuple, dict, str)) and len(val) == 0:
                return False
            return True

        valid_mask = merged_df["categories"].apply(is_valid_label) & merged_df["priority"].apply(is_valid_label)
        filtered_df = merged_df[valid_mask].reset_index(drop=True)
        print(f"  Stage 3: Filtered dataset (non-null categories & priority): {len(filtered_df):,} rows")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(out_file, index=False)
    print(f"[build_filtered_dataset] Saved {len(filtered_df):,} rows to '{output_path}'")

    return filtered_df


if __name__ == "__main__":
    import sys
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(cfg_file)
    inspect_label_schema(cfg.get("paths", {}).get("raw_labels_path", "data/raw/labels.json"))
    build_filtered_dataset(config_path=cfg_file)
