import pandas as pd

from ashare_alpha.logger import logger


def build_price_panel(
    daily: pd.DataFrame,
    adj: pd.DataFrame,
    daily_basic: pd.DataFrame,
    stk_limit: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Building price panel ...")

    df = daily.merge(adj, on=["ts_code", "trade_date"], how="left")

    df["adj_open"] = df["open"] * df["adj_factor"]
    df["adj_high"] = df["high"] * df["adj_factor"]
    df["adj_low"] = df["low"] * df["adj_factor"]
    df["adj_close"] = df["close"] * df["adj_factor"]
    df["ret"] = df.groupby("ts_code")["adj_close"].pct_change()

    df = df.drop(columns=["adj_factor"], errors="ignore")

    df = df.merge(
        daily_basic[["ts_code", "trade_date", "pe_ttm", "pb", "total_mv", "circ_mv", "turnover_rate"]],
        on=["ts_code", "trade_date"],
        how="left",
    )

    df = df.merge(
        stk_limit[["ts_code", "trade_date", "up_limit", "down_limit"]],
        on=["ts_code", "trade_date"],
        how="left",
    )

    logger.info("Price panel built: %d rows", len(df))
    return df
