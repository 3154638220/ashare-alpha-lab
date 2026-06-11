import numpy as np
import pandas as pd

from .transform import cross_sectional_industry_zscore


def calc_value_factor(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["ts_code", "trade_date", "industry_code", "pb", "pe_ttm"]].copy()

    out = out[(out["pb"] > 0) & (out["pe_ttm"] > 0)]

    out["raw_value_pb"] = -np.log(out["pb"])
    out["raw_value_pe"] = -np.log(out["pe_ttm"])

    out["value_pb"] = cross_sectional_industry_zscore(out, "raw_value_pb")
    out["value_pe"] = cross_sectional_industry_zscore(out, "raw_value_pe")

    out["value"] = 0.5 * out["value_pb"] + 0.5 * out["value_pe"]

    return out[["ts_code", "trade_date", "value"]]
