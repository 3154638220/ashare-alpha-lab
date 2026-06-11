import numpy as np
import pandas as pd


def winsorize_series(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    low = s.quantile(lower)
    high = s.quantile(upper)
    return s.clip(low, high)


def zscore_series(s: pd.Series) -> pd.Series:
    std = s.std()
    if std == 0 or np.isnan(std):
        return s * 0
    return (s - s.mean()) / std


def industry_neutral_zscore(
    df: pd.DataFrame,
    factor_col: str,
    industry_col: str = "industry_code",
) -> pd.Series:
    return (
        df.groupby(industry_col)[factor_col]
        .transform(lambda x: zscore_series(winsorize_series(x)))
    )


def cross_sectional_industry_zscore(
    df: pd.DataFrame,
    factor_col: str,
    date_col: str = "trade_date",
    industry_col: str = "industry_code",
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    groups = df.groupby([date_col, industry_col], dropna=False)[factor_col]

    low = groups.transform("quantile", lower)
    high = groups.transform("quantile", upper)
    clipped = df[factor_col].clip(lower=low, upper=high)

    grouped_clipped = clipped.groupby([df[date_col], df[industry_col]], dropna=False)
    mean = grouped_clipped.transform("mean")
    std = grouped_clipped.transform("std")

    z = (clipped - mean) / std
    return z.mask((std == 0) | std.isna(), 0.0)
