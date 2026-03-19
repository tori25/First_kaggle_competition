"""
News Headline Embedding Pipeline
=================================
Builds low-dimensional semantic embeddings for news headlines using
TF-IDF weighting and Latent Semantic Analysis (LSA) via Truncated SVD.

Usage
-----
    python solution.py --input headlines.csv --output submission.csv

The input CSV must contain a column with the headline text.  By default the
column is named "headline"; use --text-col to change it.

The output CSV will have one row per headline.  The first column is the
original headline; the remaining columns are the embedding dimensions
(emb_0, emb_1, …, emb_{n_components-1}).

Design choices
--------------
* TF-IDF with sublinear term-frequency scaling reduces the dominance of
  very common words beyond what IDF alone provides.
* stop-word removal (English) and character n-grams (1-2) help the model
  generalise over morphological variants without a full stemmer.
* TruncatedSVD (i.e. LSA) projects the high-dimensional TF-IDF space to a
  dense, low-dimensional latent semantic space that clusters well with
  K-means.
* L2-normalisation of the final embeddings makes cosine distance equivalent
  to Euclidean distance, which benefits K-means.
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


# ---------------------------------------------------------------------------
# TF-IDF configuration (shared between the pipeline builder and the vocab
# probe used in build_embeddings)
# ---------------------------------------------------------------------------

_TFIDF_PARAMS = dict(
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9'\-]*\b",
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
    stop_words="english",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_lsa_pipeline(n_components: int = 100, random_state: int = 42) -> Pipeline:
    """Return a sklearn Pipeline that maps raw text to LSA embeddings.

    Parameters
    ----------
    n_components:
        Number of latent dimensions to retain after SVD.  100 is a sensible
        default for news-headline datasets; increase for larger/richer corpora.
    random_state:
        Random seed for reproducibility.
    """
    tfidf = TfidfVectorizer(**_TFIDF_PARAMS)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    normalizer = Normalizer(norm="l2", copy=False)

    return Pipeline([("tfidf", tfidf), ("svd", svd), ("normalizer", normalizer)])


def load_headlines(path: str, text_col: str) -> pd.DataFrame:
    """Load the input CSV and return a DataFrame with at least *text_col*."""
    df = pd.read_csv(path)
    if text_col not in df.columns:
        available = ", ".join(df.columns.tolist())
        sys.exit(
            f"Column '{text_col}' not found in {path}.\n"
            f"Available columns: {available}\n"
            f"Use --text-col to specify the correct column name."
        )
    # Drop rows where the headline is missing
    n_before = len(df)
    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[info] Dropped {n_dropped} rows with missing headlines.")
    return df


def build_embeddings(headlines: pd.Series, n_components: int, random_state: int) -> np.ndarray:
    """Fit the LSA pipeline on *headlines* and return the embedding matrix.

    n_components is automatically capped at (vocabulary_size - 1) so the
    pipeline works correctly on small corpora (e.g. during testing).
    """
    # Determine actual vocabulary size first so we can cap n_components safely
    tfidf_probe = TfidfVectorizer(**_TFIDF_PARAMS)
    tfidf_probe.fit(headlines.astype(str))
    vocab_size = len(tfidf_probe.vocabulary_)
    effective_n = min(n_components, max(1, vocab_size - 1))
    if effective_n < n_components:
        print(
            f"[info] Vocabulary size ({vocab_size}) is smaller than requested "
            f"n_components ({n_components}); using {effective_n} components instead."
        )

    pipeline = build_lsa_pipeline(n_components=effective_n, random_state=random_state)
    embeddings = pipeline.fit_transform(headlines.astype(str))
    return embeddings


def save_submission(df: pd.DataFrame, text_col: str, embeddings: np.ndarray, output_path: str) -> None:
    """Write the submission CSV to *output_path*."""
    n_dims = embeddings.shape[1]
    col_names = [f"emb_{i}" for i in range(n_dims)]
    emb_df = pd.DataFrame(embeddings, columns=col_names)
    result = pd.concat([df[[text_col]].reset_index(drop=True), emb_df], axis=1)
    result.to_csv(output_path, index=False)
    print(f"[info] Saved {len(result)} embeddings ({n_dims} dims) → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate LSA embeddings for news headlines."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input CSV file containing headlines.",
    )
    parser.add_argument(
        "--output", "-o",
        default="submission.csv",
        help="Path to write the output submission CSV (default: submission.csv).",
    )
    parser.add_argument(
        "--text-col",
        default="headline",
        help="Name of the CSV column containing headline text (default: headline).",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=100,
        help="Number of LSA dimensions (default: 100).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"[info] Loading headlines from '{args.input}' …")
    df = load_headlines(args.input, args.text_col)
    print(f"[info] {len(df)} headlines loaded.")

    print(f"[info] Building TF-IDF + SVD (LSA) pipeline with {args.n_components} components …")
    embeddings = build_embeddings(df[args.text_col], args.n_components, args.random_state)
    print(f"[info] Embedding matrix shape: {embeddings.shape}")

    save_submission(df, args.text_col, embeddings, args.output)


if __name__ == "__main__":
    main()
