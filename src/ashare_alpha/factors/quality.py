import pandas as pd

from .transform import cross_sectional_industry_zscore


def calc_quality_factor(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        ["ts_code", "trade_date", "industry_code", "roe_dt", "roa", "ocf_to_or"]
    ].copy()

    for col in ["roe_dt", "roa", "ocf_to_or"]:
        out[f"z_{col}"] = cross_sectional_industry_zscore(out, col)

    out["quality"] = (
        0.4 * out["z_roe_dt"] + 0.3 * out["z_roa"] + 0.3 * out["z_ocf_to_or"]
    )

    return out[["ts_code", "trade_date", "quality"]]
