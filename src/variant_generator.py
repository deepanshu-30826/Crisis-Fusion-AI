"""Deterministic variant generator for local robustness testing of crisis reports."""

import hashlib
import math
import string
from typing import List, Dict, Any

import pandas as pd

DATE_SEED: str = "20260911"


def compute_hash_int(message_id: str, date_seed: str = DATE_SEED) -> int:
    """Compute deterministic integer hash from date_seed and message_id."""
    h_str = f"{date_seed}:{message_id}"
    h_hex = hashlib.sha256(h_str.encode("utf-8")).hexdigest()
    return int(h_hex, 16)


def should_generate_variant(h_int: int) -> bool:
    """Return True if message qualifies for variant generation (last digit in {0,1,2,3,4,5})."""
    return (h_int % 10) in {0, 1, 2, 3, 4, 5}


def select_transform(h_int: int) -> int:
    """Select transform index 0, 1, 2, or 3 based on hash."""
    return h_int % 4


def transform_0_lowercase_strip_punct(text: str) -> str:
    """Transform 0: Lowercase the text and strip ASCII punctuation."""
    lowered = text.lower()
    return lowered.translate(str.maketrans("", "", string.punctuation))


def transform_1_add_affixes(text: str) -> str:
    """Transform 1: Prepend 'Update: ' and append ' Please verify.'."""
    return f"Update: {text} Please verify."


def transform_2_truncate_tail(text: str) -> str:
    """Transform 2: Remove final ceil(0.15 * n_tokens) tokens (whitespace-split), keeping >= 1 token."""
    tokens = text.split()
    n = len(tokens)
    if n <= 1:
        return text
    k = math.ceil(0.15 * n)
    keep = max(1, n - k)
    return " ".join(tokens[:keep])


def transform_3_char_swap(text: str) -> str:
    """Transform 3: At positions 40, 80, 120... swap char with next char if both are alphanumeric."""
    chars = list(text)
    length = len(chars)
    # Positions 40, 80, 120, ... (step 40)
    for pos in range(40, length - 1, 40):
        if chars[pos].isalnum() and chars[pos + 1].isalnum():
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return "".join(chars)


def apply_transform(text: str, transform_type: int) -> str:
    """Apply the specified transform index to the text."""
    if transform_type == 0:
        return transform_0_lowercase_strip_punct(text)
    elif transform_type == 1:
        return transform_1_add_affixes(text)
    elif transform_type == 2:
        return transform_2_truncate_tail(text)
    elif transform_type == 3:
        return transform_3_char_swap(text)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


def generate_variants(
    df: pd.DataFrame,
    id_col: str = "tweet_id",
    text_col: str = "clean_text",
    date_seed: str = DATE_SEED
) -> pd.DataFrame:
    """Generate variants for qualifying rows in the dataframe.

    Returns:
        DataFrame with columns: ['source_id', 'variant_text', 'transform_type']
    """
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        msg_id = str(row[id_col])
        text = str(row[text_col])
        h_int = compute_hash_int(msg_id, date_seed=date_seed)

        if should_generate_variant(h_int):
            t_type = select_transform(h_int)
            variant_text = apply_transform(text, t_type)
            records.append({
                "source_id": msg_id,
                "variant_text": variant_text,
                "transform_type": t_type
            })

    return pd.DataFrame(records, columns=["source_id", "variant_text", "transform_type"])


if __name__ == "__main__":
    print("=== Testing Transform Families ===")
    sample_text = "Urgent: Flash flood on Highway 101 near bridge. Rescue teams requested immediately!"
    long_sample = "Urgent medical supplies needed at Central Hospital: blood units, oxygen tanks, and clean water urgently!"

    print(f"Original: {sample_text}\n")
    print(f"Transform 0 (lowercase & strip punct):\n  -> {transform_0_lowercase_strip_punct(sample_text)}\n")
    print(f"Transform 1 (add affixes):\n  -> {transform_1_add_affixes(sample_text)}\n")
    print(f"Transform 2 (truncate tail):\n  -> {transform_2_truncate_tail(sample_text)}\n")
    print(f"Transform 3 (char swap on long text):\n  -> {transform_3_char_swap(long_sample)}\n")

    # Test generation with sample DataFrame
    test_df = pd.DataFrame({
        "tweet_id": [f"tweet_{i}" for i in range(10)],
        "clean_text": [sample_text] * 10
    })
    variants = generate_variants(test_df)
    print(f"Generated {len(variants)} variants out of {len(test_df)} messages.")
    print(variants.head())
