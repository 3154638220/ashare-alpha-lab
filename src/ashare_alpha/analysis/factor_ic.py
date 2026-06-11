import pandas as pd

from ashare_alpha.logger import logger


def calc_forward_return(
    price: pd.DataFrame,
    horizon: int = 20,
) -> pd.DataFrame:
    df = price[["ts_code", "trade_date", "adj_close"]].copy()
    df = df.sort_values(["ts_code", "trade_date"])

    df["future_close"] = df.groupby("ts_code")["adj_close"].shift(-horizon)

    df["future_return"] = df["future_close"] / df["adj_close"] - 1

    return df[["ts_code", "trade_date", "future_return"]]


def calc_rank_ic(
    factor: pd.DataFrame,
    price: pd.DataFrame,
    factor_col: str = "score",
    horizon: int = 20,
) -> pd.DataFrame:
    future_ret = calc_forward_return(price, horizon)

    df = factor.merge(future_ret, on=["ts_code", "trade_date"], how="inner")

    records = []

    for trade_date, g in df.groupby("trade_date"):
        if len(g) < 20:
            continue

        ic = g[factor_col].corr(g["future_return"], method="spearman")

        records.append({"trade_date": trade_date, "rank_ic": ic})

    return pd.DataFrame(records)


def calc_icir(ic_df: pd.DataFrame) -> float:
    mean_ic = ic_df["rank_ic"].mean()
    std_ic = ic_df["rank_ic"].std()

    if std_ic == 0:
        return 0.0

    return mean_ic / std_ic


def calc_factor_ic_summary(
    factor_scores: pd.DataFrame,
    price: pd.DataFrame,
    factor_names: list[str] | None = None,
    horizon: int = 20,
) -> dict:
    if factor_names is None:
        factor_names = [c for c in factor_scores.columns if c not in ("ts_code", "trade_date")]

    summary = {}

    for name in factor_names:
        if name not in factor_scores.columns:
            continue

        ic_df = calc_rank_ic(factor_scores, price, factor_col=name, horizon=horizon)
        icir = calc_icir(ic_df)

        summary[name] = {
            "mean_ic": float(ic_df["rank_ic"].mean()) if not ic_df.empty else float("nan"),
            "std_ic": float(ic_df["rank_ic"].std()) if not ic_df.empty else float("nan"),
            "icir": float(icir),
            "pos_ic_ratio": float((ic_df["rank_ic"] > 0).mean()) if not ic_df.empty else float("nan"),
        }

        logger.info("Factor %s: mean_IC=%.4f, ICIR=%.4f", name, summary[name]["mean_ic"], icir)

    return summary
