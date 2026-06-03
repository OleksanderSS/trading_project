import pandas as pd
from sklearn.preprocessing import LabelEncoder


def handle_categorical_features_split(
    x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame, exclude_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Кодує категоріальні колонки, використовуючи лише тренувальну вибірку."""

    # Визначаємо категоріальні колонки
    all_cols = x_train.columns
    cat_cols = [
        c
        for c in x_train.select_dtypes(include=["object", "category"]).columns
        if c not in exclude_cols and "ticker" not in c.lower() and "timeframe" not in c.lower()
    ]

    train_out = x_train.copy()
    val_out = x_val.copy()
    test_out = x_test.copy()

    info = {}
    for col in cat_cols:
        # Check nunique in train only
        nunique = train_out[col].nunique()
        if nunique < 2:
            train_out.drop(columns=[col], inplace=True)
            val_out.drop(columns=[col], inplace=True)
            test_out.drop(columns=[col], inplace=True)
            continue

        if nunique <= 5:
            # One-Hot encoding: Fit on train, apply to all
            # Handle categories not seen in train by creating 0 for all dummies if needed
            dummies_train = pd.get_dummies(train_out[col], prefix=col, drop_first=True)
            cols_dummies = dummies_train.columns

            train_out = pd.concat([train_out, dummies_train], axis=1).drop(columns=[col])

            dummies_val = pd.get_dummies(val_out[col], prefix=col, drop_first=True).reindex(
                columns=cols_dummies, fill_value=0
            )
            val_out = pd.concat([val_out, dummies_val], axis=1).drop(columns=[col])

            dummies_test = pd.get_dummies(test_out[col], prefix=col, drop_first=True).reindex(
                columns=cols_dummies, fill_value=0
            )
            test_out = pd.concat([test_out, dummies_test], axis=1).drop(columns=[col])

            info[col] = "one-hot"
        else:
            # Label encoding: Fit on train, apply to all
            # Note: Unseen categories in val/test will cause errors with basic LabelEncoder
            le = LabelEncoder()
            train_out[col] = le.fit_transform(train_out[col].astype(str))

            # For val/test, we need a way to handle unseen labels.
            # A common approach is mapping unseen to -1 or a special value.
            # Using dict mapping is safer.
            mapping = {label: i for i, label in enumerate(le.classes_)}
            val_out[col] = val_out[col].astype(str).map(mapping).fillna(-1)
            test_out[col] = test_out[col].astype(str).map(mapping).fillna(-1)
            info[col] = "label"

    return train_out, val_out, test_out, info
