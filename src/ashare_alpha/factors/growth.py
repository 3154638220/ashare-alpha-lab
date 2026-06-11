import pandas as pd

from .transform import cross_sectional_industry_zscore


def calc_growth_factor(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        ["ts_code", "trade_date", "industry_code", "or_yoy", "netprofit_yoy"]
    ].copy()

    for col in ["or_yoy", "netprofit_yoy"]:
        out[f"z_{col}"] = cross_sectional_industry_zscore(out, col)

    out["growth"] = 0.5 * out["z_or_yoy"] + 0.5 * out["z_netprofit_yoy"]

    return out[["ts_code", "trade_date", "growth"]]
