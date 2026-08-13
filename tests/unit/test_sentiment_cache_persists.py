"""Every rebuild rescored the same fifteen thousand articles.

`_CACHE` was a module-level dict, so it died with the process. FinBERT on
CPU spent over half an hour on the 2026-08-13 corpus, and the next rebuild
spent it again on the same texts — the corpus grows by a few hundred
articles a day and the rest is identical.

Measured across a module reload, which is as close to a fresh process as a
test gets: 24 texts took 170s cold and 3.9s from disk, with identical
labels.

The cache key includes the model name. A different model is a different
answer, and a cache that forgot which one produced a row would hand
FinBERT's verdicts to whatever replaced it.
"""
import pandas as pd
import pytest

import src.sentiment.sentiment_models as models


@pytest.fixture
def cache_file(tmp_path, monkeypatch):
    path = tmp_path / "sentiment" / "scores.parquet"
    monkeypatch.setattr(models, "_CACHE_PATH", path)
    monkeypatch.setattr(models, "_CACHE", {})
    monkeypatch.setattr(models, "_CACHE_LOADED", False)
    monkeypatch.setattr(models, "_CACHE_DIRTY", False)
    return path


def test_nothing_is_written_when_nothing_was_scored(cache_file):
    models._CACHE["abc"] = {"text": "x", "label": "positive", "score": 0.9}
    # Not dirty: this row came from disk, rewriting it would be churn.
    assert models.save_sentiment_cache() == 0
    assert not cache_file.exists()


def test_scores_survive_a_restart(cache_file, monkeypatch):
    models._CACHE["abc"] = {"text": "good news", "label": "positive", "score": 0.91}
    monkeypatch.setattr(models, "_CACHE_DIRTY", True)

    assert models.save_sentiment_cache() == 1
    assert cache_file.exists()

    # A fresh process: empty memory, cache never loaded.
    monkeypatch.setattr(models, "_CACHE", {})
    monkeypatch.setattr(models, "_CACHE_LOADED", False)

    assert models.load_sentiment_cache() == 1
    assert models._CACHE["abc"]["label"] == "positive"
    assert models._CACHE["abc"]["score"] == pytest.approx(0.91)


def test_the_key_changes_with_the_model(monkeypatch):
    """FinBERT's answer must not be served for a different model."""
    text = "Company reports record earnings"
    with_finbert = models._stable_hash(text)

    monkeypatch.setattr(models, "_CACHE_MODEL", "some/other-model")
    with_other = models._stable_hash(text)

    assert with_finbert != with_other


def test_a_corrupt_cache_is_survived_not_fatal(cache_file, monkeypatch):
    """A bad cache means slow, never wrong."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(b"this is not parquet")

    assert models.load_sentiment_cache() == 0
    assert models._CACHE == {}


def test_a_partial_write_cannot_replace_a_good_cache(cache_file, monkeypatch):
    """The write goes to a temp file and is renamed into place."""
    models._CACHE["a"] = {"text": "t", "label": "neutral", "score": 0.0}
    monkeypatch.setattr(models, "_CACHE_DIRTY", True)
    models.save_sentiment_cache()

    assert cache_file.exists()
    assert not cache_file.with_suffix(".parquet.tmp").exists(), (
        "the temporary file must be renamed, not left beside the cache"
    )
    stored = pd.read_parquet(cache_file)
    assert set(stored.columns) == {"hash", "text", "label", "score"}


def test_the_cache_is_written_before_the_run_ends(cache_file, monkeypatch):
    """A run stopped part-way must not lose everything it scored.

    Three runs were interrupted in one day -- a Windows update and two
    deliberate restarts -- and each lost over two hours of FinBERT because
    the cache was written once, at the end.
    """
    saves = []
    real_save = models.save_sentiment_cache

    def counting_save():
        saves.append(len(models._CACHE))
        return real_save()

    monkeypatch.setattr(models, "save_sentiment_cache", counting_save)
    monkeypatch.setattr(models, "_SAVE_EVERY", 2)
    monkeypatch.setattr(models, "get_finbert_pipeline", lambda device=None: "pipe")

    def fake_batch(pipe, uncached, indices, batch, label_map, i, **kwargs):
        rows = []
        for idx in indices:
            row = {"text": batch[idx], "label": "neutral", "score": 0.0}
            models._CACHE[models._stable_hash(batch[idx])] = row
            models._CACHE_DIRTY = True
            rows.append(row)
        return rows

    monkeypatch.setattr(models, "_process_batch", fake_batch)

    models.analyze_sentiment([f"text {i}" for i in range(8)], batch_size=2)

    assert len(saves) > 1, (
        "the cache was only written once, at the end — an interrupted run "
        "loses everything it computed"
    )
