import pandas as pd


def calc_industry_exposure(
    positions: pd.DataFrame,
    industry: pd.DataFrame,
) -> pd.DataFrame:
    positions = positions.copy()
    industry = industry.copy()
    positions["trade_date"] = positions["trade_date"].astype(str)
    industry["trade_date"] = industry["trade_date"].astype(str)

    df = positions.merge(
        industry[["ts_code", "trade_date", "industry_name"]],
        on=["ts_code", "trade_date"],
        how="left",
    )

    total_mv = df.groupby("trade_date")["market_value"].transform("sum")
    df["weight"] = df["market_value"] / total_mv

    exposure = (
        df.groupby(["trade_date", "industry_name"])["weight"]
        .sum()
        .reset_index()
    )

    return exposure


def calc_industry_exposure_summary(
    exposure: pd.DataFrame,
) -> pd.DataFrame:
    avg_exposure = (
        exposure.groupby("industry_name")["weight"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    avg_exposure.columns = ["industry_name", "avg_weight"]
    return avg_exposure
