import numpy as np
import pandas as pd


def calc_var(nav: pd.DataFrame, confidence: float = 0.95) -> float:
    daily_ret = nav["daily_ret"].dropna()
    if daily_ret.empty:
        return float("nan")
    return np.percentile(daily_ret, 100 * (1 - confidence))


def calc_cvar(nav: pd.DataFrame, confidence: float = 0.95) -> float:
    daily_ret = nav["daily_ret"].dropna()
    if daily_ret.empty:
        return float("nan")
    var = calc_var(nav, confidence)
    return daily_ret[daily_ret <= var].mean()


def calc_concentration(positions: pd.DataFrame) -> float:
    if positions.empty:
        return float("nan")

    latest_date = positions["trade_date"].max()
    latest = positions[positions["trade_date"] == latest_date]

    if latest.empty:
        return float("nan")

    total_mv = latest["market_value"].sum()
    if total_mv == 0:
        return float("nan")

    weights = latest["market_value"] / total_mv
    return (weights ** 2).sum()
