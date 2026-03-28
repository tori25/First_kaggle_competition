"""
Tests for the news headline embedding pipeline (solution.py).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

from solution import build_embeddings, build_lsa_pipeline, load_headlines, save_submission


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_HEADLINES = [
    "Tech giant unveils new AI assistant",
    "Startup raises millions for quantum computing",
    "Government announces sweeping tax reform",
    "Senate votes to confirm new cabinet secretary",
    "Clinical trial shows promising cancer therapy results",
    "Health officials urge vaccination ahead of flu season",
    "Astronomers detect water vapour on distant exoplanet",
    "Biologists discover new deep-sea species",
    "Local football team wins championship",
    "Olympic committee announces new sports",
]

# Larger corpus used for pipeline unit tests that need a non-trivial vocabulary
LARGE_HEADLINES = [
    "New smartphone model breaks all sales records",
    "Tech giant unveils revolutionary AI assistant for consumers",
    "Startup raises millions for quantum computing breakthrough",
    "Social media platform announces major privacy overhaul",
    "Self-driving car completes cross-country journey safely",
    "Open-source software community celebrates major milestone",
    "Cloud provider suffers massive outage affecting millions worldwide",
    "Silicon Valley sees surge in robotics investment this year",
    "Cybersecurity firm warns of new ransomware strain targeting hospitals",
    "Next-generation chip doubles computing performance benchmarks",
    "Local team wins championship after decade-long drought in finals",
    "Star athlete signs record-breaking contract extension deal",
    "Olympic committee announces new sports for upcoming summer games",
    "Marathon runner sets new world record in extreme desert heat",
    "Football league expands with two new franchise teams joining",
    "Tennis player beats top seed in stunning upset at tournament",
    "Basketball superstar announces retirement after twenty seasons",
    "Soccer World Cup host city finalises new stadium construction plans",
    "Swimmer breaks own world record at national championships event",
    "Rugby union team clinches Grand Slam title in final match",
    "Government announces sweeping tax reform package for businesses",
    "Senate votes to confirm new cabinet secretary after hearing",
    "International summit focuses on climate change policy commitments",
    "Prime minister calls early election amid growing economic uncertainty",
    "Opposition party unveils ambitious national healthcare reform plan",
    "Diplomat visits troubled region amid rising border tensions",
    "New trade agreement expected to boost exports significantly next year",
    "City mayor proposes major infrastructure spending bill for transit",
    "Foreign minister meets counterpart to discuss ongoing peace talks",
    "Parliament debates controversial new immigration legislation reform",
    "Clinical trial shows promising results for new cancer immunotherapy",
    "Health officials urge mass vaccination ahead of winter flu season",
    "Researchers discover new link between daily diet and human longevity",
    "Hospital system adopts artificial intelligence for early disease detection",
    "Mental health awareness campaign reaches record national participation levels",
    "Pharmaceutical company recalls large batch of blood pressure medication",
    "Study links urban air pollution to increased dementia risk in elderly",
    "Surge in telemedicine reshapes how patients access primary care services",
    "Scientists develop smart bandage that accelerates chronic wound healing",
    "New clinical guidelines recommend earlier colorectal cancer screening",
    "Astronomers detect water vapour on distant rocky exoplanet atmosphere",
    "Marine biologists discover new species during deep-sea expedition voyage",
    "Physicists confirm long-sought elementary particle at large collider",
    "Mars rover uncovers geological evidence of ancient river delta system",
    "Gene-editing technique shows promise for treating hereditary blood diseases",
    "Paleontologists unearth largest sauropod dinosaur skeleton ever found",
    "Climate researchers warn of imminent tipping point for Arctic sea ice",
    "Chemists synthesise novel porous material with remarkable gas absorption",
    "Neuroscientists map previously unknown long-range brain connectivity network",
    "Volcanic eruption creates new island landmass in Pacific Ocean region",
]


@pytest.fixture()
def sample_csv(tmp_path):
    """Write a small CSV and return its path."""
    path = tmp_path / "headlines.csv"
    pd.DataFrame({"headline": SAMPLE_HEADLINES}).to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def sample_df():
    return pd.DataFrame({"headline": SAMPLE_HEADLINES})


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestBuildLsaPipeline:
    def test_pipeline_has_three_steps(self):
        pipeline = build_lsa_pipeline()
        assert list(pipeline.named_steps.keys()) == ["tfidf", "svd", "normalizer"]

    def test_pipeline_produces_correct_shape(self):
        pipeline = build_lsa_pipeline(n_components=5)
        result = pipeline.fit_transform(LARGE_HEADLINES)
        assert result.shape == (len(LARGE_HEADLINES), 5)

    def test_embeddings_are_unit_normalised(self):
        pipeline = build_lsa_pipeline(n_components=5)
        result = pipeline.fit_transform(LARGE_HEADLINES)
        norms = np.linalg.norm(result, axis=1)
        # Rows with all-zero TF-IDF (all terms are stop-words or below min_df)
        # produce zero-norm vectors; skip those in the unit-norm check
        non_zero = norms > 0
        np.testing.assert_allclose(norms[non_zero], np.ones(non_zero.sum()), atol=1e-6)

    def test_pipeline_is_reproducible(self):
        p1 = build_lsa_pipeline(n_components=5, random_state=0)
        p2 = build_lsa_pipeline(n_components=5, random_state=0)
        e1 = p1.fit_transform(LARGE_HEADLINES)
        e2 = p2.fit_transform(LARGE_HEADLINES)
        np.testing.assert_array_equal(e1, e2)

    def test_different_random_states_may_differ(self):
        p1 = build_lsa_pipeline(n_components=5, random_state=0)
        p2 = build_lsa_pipeline(n_components=5, random_state=99)
        e1 = p1.fit_transform(LARGE_HEADLINES)
        e2 = p2.fit_transform(LARGE_HEADLINES)
        # Both should be valid arrays; non-zero rows should have unit norm
        for e in (e1, e2):
            norms = np.linalg.norm(e, axis=1)
            non_zero = norms > 0
            np.testing.assert_allclose(norms[non_zero], np.ones(non_zero.sum()), atol=1e-6)


class TestBuildEmbeddings:
    def test_output_shape(self, sample_df):
        n = 5
        emb = build_embeddings(sample_df["headline"], n_components=n, random_state=42)
        # n_components may be capped if vocabulary is smaller than requested
        assert emb.shape[0] == len(SAMPLE_HEADLINES)
        assert emb.shape[1] <= n

    def test_output_dtype_is_float(self, sample_df):
        emb = build_embeddings(sample_df["headline"], n_components=5, random_state=42)
        assert np.issubdtype(emb.dtype, np.floating)

    def test_no_nan_in_output(self, sample_df):
        emb = build_embeddings(sample_df["headline"], n_components=5, random_state=42)
        assert not np.isnan(emb).any()


class TestLoadHeadlines:
    def test_loads_correct_number_of_rows(self, sample_csv):
        df = load_headlines(sample_csv, "headline")
        assert len(df) == len(SAMPLE_HEADLINES)

    def test_loads_correct_column(self, sample_csv):
        df = load_headlines(sample_csv, "headline")
        assert "headline" in df.columns

    def test_exits_on_missing_column(self, sample_csv):
        with pytest.raises(SystemExit):
            load_headlines(sample_csv, "nonexistent_col")

    def test_drops_rows_with_missing_headlines(self, tmp_path):
        path = tmp_path / "with_nulls.csv"
        headlines = SAMPLE_HEADLINES + [None, ""]
        pd.DataFrame({"headline": headlines}).to_csv(path, index=False)
        df = load_headlines(str(path), "headline")
        # Both None and empty string are treated as NaN when read back from CSV
        # so only the original SAMPLE_HEADLINES rows survive
        assert len(df) == len(SAMPLE_HEADLINES)


class TestSaveSubmission:
    def test_creates_output_file(self, tmp_path, sample_df):
        emb = build_embeddings(sample_df["headline"], n_components=5, random_state=42)
        out = str(tmp_path / "submission.csv")
        save_submission(sample_df, "headline", emb, out)
        assert os.path.exists(out)

    def test_output_has_correct_columns(self, tmp_path, sample_df):
        n = 5
        emb = build_embeddings(sample_df["headline"], n_components=n, random_state=42)
        actual_n = emb.shape[1]  # may be capped below n
        out = str(tmp_path / "submission.csv")
        save_submission(sample_df, "headline", emb, out)
        result = pd.read_csv(out)
        assert "headline" in result.columns
        for i in range(actual_n):
            assert f"emb_{i}" in result.columns

    def test_output_row_count_matches_input(self, tmp_path, sample_df):
        emb = build_embeddings(sample_df["headline"], n_components=5, random_state=42)
        out = str(tmp_path / "submission.csv")
        save_submission(sample_df, "headline", emb, out)
        result = pd.read_csv(out)
        assert len(result) == len(sample_df)


# ---------------------------------------------------------------------------
# End-to-end / CLI integration test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_cli_produces_submission(self, sample_csv, tmp_path):
        """Run the CLI entry-point and verify a valid submission is created."""
        from solution import main

        out = str(tmp_path / "submission.csv")
        main(["--input", sample_csv, "--output", out, "--n-components", "5"])
        assert os.path.exists(out)
        result = pd.read_csv(out)
        assert len(result) == len(SAMPLE_HEADLINES)
        # Should have headline column + at least 1 embedding column (≤5 due to capping)
        assert result.shape[1] >= 2
        assert "headline" in result.columns

    def test_cli_default_output_name(self, sample_csv, tmp_path, monkeypatch):
        """Verify default output filename is submission.csv."""
        from solution import main

        monkeypatch.chdir(tmp_path)
        main(["--input", sample_csv, "--n-components", "5"])
        assert os.path.exists(tmp_path / "submission.csv")

    def test_cli_custom_text_col(self, tmp_path):
        """Support non-default column name via --text-col."""
        from solution import main

        path = tmp_path / "data.csv"
        pd.DataFrame({"title": SAMPLE_HEADLINES}).to_csv(path, index=False)
        out = str(tmp_path / "out.csv")
        main(["--input", str(path), "--output", out, "--text-col", "title", "--n-components", "5"])
        result = pd.read_csv(out)
        assert "title" in result.columns
        assert len(result) == len(SAMPLE_HEADLINES)
