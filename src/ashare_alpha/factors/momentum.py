import pandas as pd

from .transform import cross_sectional_industry_zscore


def calc_momentum_reversal_factor(price: pd.DataFrame) -> pd.DataFrame:
    out = price[["ts_code", "trade_date", "industry_code", "adj_close"]].copy()

    g = out.groupby("ts_code")["adj_close"]

    out["mom_120_20"] = g.shift(20) / g.shift(120) - 1
    out["ret_20"] = out["adj_close"] / g.shift(20) - 1

    out["raw_momentum"] = out["mom_120_20"]
    out["raw_reversal"] = -out["ret_20"]

    out["momentum"] = cross_sectional_industry_zscore(out, "raw_momentum")
    out["reversal"] = cross_sectional_industry_zscore(out, "raw_reversal")

    return out[["ts_code", "trade_date", "momentum", "reversal"]]
