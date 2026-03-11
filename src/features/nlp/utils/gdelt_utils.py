# src/feature_engineering/nlp/gdelt_utils.py

import pandas as pd
from .keyword_features import KeywordExtractor

def normalize_gdelt_signals(df: pd.DataFrame, tickers: dict = None, keyword_dict: dict = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "ticker", "keywords", "mention_score", "signal_strength"])

    df = df.copy()

    # Date
    if "DATE" in df.columns:
        df["date"] = pd.to_datetime(df["DATE"], errors="coerce").dt.normalize()
    elif "SQLDATE" in df.columns:
        df["date"] = pd.to_datetime(df["SQLDATE"], errors="coerce").dt.normalize()
    else:
        df["date"] = pd.NaT

    # Tickers
    df["ticker"] = "GENERAL"
    if tickers:
        for t in tickers.keys():
            mask = False
            for col in ["Themes", "V2Themes", "Actor1Name", "Actor2Name", "DocumentIdentifier", "SourceCommonName"]:
                if col in df.columns:
                    mask |= df[col].astype(str).str.contains(t, case=False, na=False)
            df.loc[mask, "ticker"] = t

    # [SEARCH] Filter by keyword dictionary
    if keyword_dict:
        extractor = KeywordExtractor(keyword_dict)
        df["keywords"] = df.apply(
            lambda row: extractor.extract_keywords(
                " ".join([str(row.get(c, "")) for c in ["Themes", "V2Themes", "Actor1Name", "Actor2Name", "DocumentIdentifier", "SourceCommonName"]])
            ),
            axis=1
        )
        df["match_count"] = df["keywords"].apply(len)
        df = df[df["match_count"] > 0].reset_index(drop=True)

    # Mentions
    if "NumMentions" in df.columns:
        df["mention_score"] = pd.to_numeric(df["NumMentions"], errors="coerce").fillna(1)
    elif "NumSources" in df.columns:
        df["mention_score"] = pd.to_numeric(df["NumSources"], errors="coerce").fillna(1)
    else:
        df["mention_score"] = 1

    # Signal strength = number of mentions (for now without classification)
    df["signal_strength"] = df["mention_score"]

    return df[["date", "ticker", "keywords", "mention_score", "signal_strength"]]
