"""Light-touch, meaning-preserving text preprocessing for crisis reports."""

import re
import pandas as pd


def clean_text(text: str) -> str:
    """Perform conservative, meaning-preserving normalization on crisis text.

    Key principles:
    - Lowercase text
    - Strip URLs, @mentions, and RT/retweet boilerplate
    - Keep punctuation carrying urgent sentiment (e.g., '!', '?')
    - Strip emojis and decorative Unicode glyph clutter
    - Preserve negations (not, no, n't, never), numbers, and critical tokens
    - Collapse repeated whitespace
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    cleaned = text.lower()

    # 2. Strip URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # 3. Strip @mentions
    cleaned = re.sub(r"@\w+", " ", cleaned)

    # 4. Strip RT/retweet boilerplate
    cleaned = re.sub(r"^(rt\s*[:\s]+)+", " ", cleaned)
    cleaned = re.sub(r"\b(retweet|rt)\b", " ", cleaned)

    # 5. Remove decorative symbols / emojis while keeping letters, digits, whitespace,
    # and essential punctuation (!, ?, ., ,, -, ', ", /)
    # We strip decorative emojis/symbols (e.g., emojis, stars, arrows)
    cleaned = re.sub(r"[^\w\s!?.,'\":;/\-]", " ", cleaned)

    # 6. Collapse repeated whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Apply clean_text to text_col and add clean_text column.

    Returns:
        DataFrame with an added 'clean_text' column.
    """
    df_copy = df.copy()
    if text_col in df_copy.columns:
        df_copy["clean_text"] = df_copy[text_col].apply(clean_text)
    else:
        df_copy["clean_text"] = ""
    return df_copy


if __name__ == "__main__":
    # Test suite to verify meaning preservation
    s1 = "The building is NOT damaged!"
    s2 = "The building is damaged!"

    c1 = clean_text(s1)
    c2 = clean_text(s2)

    print(f"Original 1: '{s1}' -> Cleaned 1: '{c1}'")
    print(f"Original 2: '{s2}' -> Cleaned 2: '{c2}'")

    assert c1 != c2, f"Assertion Failed: '{c1}' equals '{c2}', negation was destroyed!"
    assert "not" in c1, f"Assertion Failed: 'not' was stripped from '{c1}'"
    assert "!" in c1, f"Assertion Failed: '!' was stripped from '{c1}'"

    # Test numbers and RT stripping
    s3 = "RT @emergency_ops: 45 injured in flood zone! https://t.co/abc1234"
    c3 = clean_text(s3)
    print(f"Original 3: '{s3}' -> Cleaned 3: '{c3}'")
    assert "45" in c3, "Numbers should be preserved"
    assert "@" not in c3, "Mentions should be stripped"
    assert "http" not in c3, "URLs should be stripped"

    print("All preprocessing assertions passed successfully!")
