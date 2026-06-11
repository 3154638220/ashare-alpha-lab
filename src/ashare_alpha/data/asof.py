import pandas as pd

from ashare_alpha.logger import logger


def build_fundamental_asof(
    trade_dates: list[str],
    fina: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Building fundamental as-of with %d trade dates ...", len(trade_dates))

    fina = fina.copy()
    fina = fina.dropna(subset=["ann_date"])
    fina = fina.sort_values(["ts_code", "ann_date", "end_date"])

    result = []

    for trade_date in trade_dates:
        available = fina[fina["ann_date"] <= trade_date]
        latest = (
            available.sort_values(["ts_code", "ann_date", "end_date"])
            .groupby("ts_code")
            .tail(1)
        )
        latest = latest.copy()
        latest["trade_date"] = trade_date
        result.append(latest)

    df = pd.concat(result, ignore_index=True)
    logger.info("Fundamental as-of built: %d rows", len(df))
    return df


def build_industry_asof(
    trade_dates: list[str],
    industry_member: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Building industry as-of with %d trade dates ...", len(trade_dates))

    df = industry_member.copy()
    df = df.rename(columns={
        "index_code": "industry_code",
        "index_name": "industry_name",
        "con_code": "ts_code",
    })

    result = []

    for trade_date in trade_dates:
        avail = df[df["in_date"] <= trade_date].copy()
        avail = avail[avail["ts_code"].notna()]
        latest = avail.groupby("ts_code").tail(1)
        latest = latest.copy()
        latest["trade_date"] = trade_date
        result.append(latest)

    out = pd.concat(result, ignore_index=True)
    logger.info("Industry as-of built: %d rows", len(out))
    return out
