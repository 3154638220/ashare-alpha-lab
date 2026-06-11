import pandas as pd

from ashare_alpha.logger import logger


def filter_listed_days(
    df: pd.DataFrame,
    stock_basic: pd.DataFrame,
    trade_date: str,
    min_list_days: int,
) -> pd.DataFrame:
    stock_basic = stock_basic.copy()
    stock_basic["list_date"] = stock_basic["list_date"].astype(str)

    listed = stock_basic[stock_basic["list_date"] <= trade_date].copy()

    listed["listed_days"] = (
        pd.to_datetime(trade_date) - pd.to_datetime(listed["list_date"])
    ).dt.days

    valid_codes = listed.loc[listed["listed_days"] >= min_list_days, "ts_code"]

    return df[df["ts_code"].isin(valid_codes)]


def filter_st(df: pd.DataFrame, stock_basic: pd.DataFrame) -> pd.DataFrame:
    names = stock_basic[["ts_code", "name"]].copy()
    df = df.merge(names, on="ts_code", how="left")

    mask = ~df["name"].str.contains("ST|退", na=False)
    return df[mask].drop(columns=["name"])


def filter_liquidity(
    df: pd.DataFrame,
    price_panel: pd.DataFrame,
    trade_date: str,
    min_avg_amount: float,
    window: int = 20,
) -> pd.DataFrame:
    hist = price_panel[price_panel["trade_date"] <= trade_date].copy()
    hist = hist.sort_values(["ts_code", "trade_date"])

    hist["avg_amount_20"] = (
        hist.groupby("ts_code")["amount"]
        .rolling(window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    latest = hist.groupby("ts_code").tail(1)
    valid_codes = latest.loc[
        latest["avg_amount_20"] >= min_avg_amount, "ts_code"
    ]

    return df[df["ts_code"].isin(valid_codes)]


def filter_valuation_valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["pb"] > 0) & (df["pe_ttm"] > 0)]


def build_universe(
    trade_date: str,
    factor_input: pd.DataFrame,
    price_panel: pd.DataFrame,
    stock_basic: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    universe_cfg = config["strategy"]["universe"]

    df = factor_input[factor_input["trade_date"] == trade_date].copy()

    before = len(df)

    df = filter_listed_days(
        df, stock_basic, trade_date, universe_cfg["min_list_days"]
    )
    logger.info("After listed days filter: %d -> %d", before, len(df))

    if universe_cfg.get("exclude_st", True):
        before = len(df)
        df = filter_st(df, stock_basic)
        logger.info("After ST filter: %d -> %d", before, len(df))

    if universe_cfg.get("exclude_negative_pe", True):
        before = len(df)
        df = filter_valuation_valid(df)
        logger.info("After valuation filter: %d -> %d", before, len(df))

    if "min_avg_amount_20" in universe_cfg:
        before = len(df)
        df = filter_liquidity(
            df,
            price_panel,
            trade_date,
            universe_cfg["min_avg_amount_20"],
        )
        logger.info("After liquidity filter: %d -> %d", before, len(df))

    return df
