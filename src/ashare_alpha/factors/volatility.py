import pandas as pd

from .transform import cross_sectional_industry_zscore


def calc_lowvol_factor(
    price: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    out = price[["ts_code", "trade_date", "industry_code", "ret"]].copy()

    out["vol60"] = (
        out.groupby("ts_code")["ret"]
        .rolling(window)
        .std()
        .reset_index(level=0, drop=True)
    )

    out["raw_lowvol"] = -out["vol60"]

    out["lowvol"] = cross_sectional_industry_zscore(out, "raw_lowvol")

    return out[["ts_code", "trade_date", "lowvol"]]
