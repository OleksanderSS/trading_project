# src/feature_engineering/nlp/gdelt_utils.py

import pandas as pd

from .keyword_features import KeywordExtractor


def _extract_date_from_gdelt(df: pd.DataFrame) -> pd.Series:
    """Extract date from GDELT DATE or SQLDATE columns."""
    if "DATE" in df.columns:
        return pd.to_datetime(df["DATE"], errors="coerce").dt.normalize()
    elif "SQLDATE" in df.columns:
        return pd.to_datetime(df["SQLDATE"], errors="coerce").dt.normalize()
    else:
        return pd.Series([pd.NaT] * len(df))


def _assign_tickers(df: pd.DataFrame, tickers: dict) -> pd.Series:
    """Assign tickers based on content in GDELT columns."""
    ticker_series = pd.Series(["GENERAL"] * len(df), index=df.index)

    if tickers:
        for t in tickers.keys():
            mask = False
            for col in ["Themes", "V2Themes", "Actor1Name", "Actor2Name", "DocumentIdentifier", "SourceCommonName"]:
                if col in df.columns:
                    mask |= df[col].astype(str).str.contains(t, case=False, na=False)
            ticker_series.loc[mask] = t

    return ticker_series


def _extract_keywords(df: pd.DataFrame, keyword_dict: dict) -> pd.Series:
    """Extract keywords from GDELT columns using keyword dictionary."""
    if not keyword_dict:
        return pd.Series([[]] * len(df), index=df.index)

    extractor = KeywordExtractor(keyword_dict)
    return df.apply(
        lambda row: extractor.extract_keywords(
            " ".join([str(row.get(c, "")) for c in ["Themes", "V2Themes", "Actor1Name", "Actor2Name", "DocumentIdentifier", "SourceCommonName"]])
        ),
        axis=1
    )


def _calculate_mention_score(df: pd.DataFrame) -> pd.Series:
    """Calculate mention score from NumMentions or NumSources columns."""
    if "NumMentions" in df.columns:
        return pd.to_numeric(df["NumMentions"], errors="coerce").fillna(1)
    elif "NumSources" in df.columns:
        return pd.to_numeric(df["NumSources"], errors="coerce").fillna(1)
    else:
        return pd.Series([1] * len(df), index=df.index)


def normalize_gdelt_signals(df: pd.DataFrame, tickers: dict = None, keyword_dict: dict = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "ticker", "keywords", "mention_score", "signal_strength"])

    df = df.copy()

    df["date"] = _extract_date_from_gdelt(df)
    df["ticker"] = _assign_tickers(df, tickers)

    # Filter by keyword dictionary
    if keyword_dict:
        df["keywords"] = _extract_keywords(df, keyword_dict)
        df["match_count"] = df["keywords"].apply(len)
        df = df[df["match_count"] > 0].reset_index(drop=True)

    df["mention_score"] = _calculate_mention_score(df)
    df["signal_strength"] = df["mention_score"]

    return df[["date", "ticker", "keywords", "mention_score", "signal_strength"]]
