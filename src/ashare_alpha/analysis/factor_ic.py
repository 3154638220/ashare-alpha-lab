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
    ic_matrix = calc_rank_ic_matrix(
        factor,
        price,
        factor_names=[factor_col],
        horizon=horizon,
    )
    if ic_matrix.empty:
        return pd.DataFrame(columns=["trade_date", "rank_ic"])

    return ic_matrix[["trade_date", "rank_ic"]].copy()


def calc_rank_ic_matrix(
    factor: pd.DataFrame,
    price: pd.DataFrame,
    factor_names: list[str],
    horizon: int = 20,
    min_obs: int = 20,
) -> pd.DataFrame:
    available_factors = [name for name in factor_names if name in factor.columns]
    if not available_factors:
        return pd.DataFrame(columns=["trade_date", "factor", "rank_ic"])

    future_ret = calc_forward_return(price, horizon)
    df = factor[["ts_code", "trade_date"] + available_factors].merge(
        future_ret,
        on=["ts_code", "trade_date"],
        how="inner",
    )

    records = []

    for trade_date, g in df.groupby("trade_date", sort=False):
        if len(g) < min_obs:
            continue

        ret = g["future_return"]
        for factor_name in available_factors:
            valid = g[factor_name].notna() & ret.notna()
            if valid.sum() < min_obs:
                continue

            factor_rank = g.loc[valid, factor_name].rank()
            ret_rank = ret.loc[valid].rank()
            ic = factor_rank.corr(ret_rank)

            records.append({
                "trade_date": trade_date,
                "factor": factor_name,
                "rank_ic": ic,
            })

    return pd.DataFrame(records)


def calc_icir(ic_df: pd.DataFrame) -> float:
    if ic_df.empty:
        return float("nan")

    mean_ic = ic_df["rank_ic"].mean()
    std_ic = ic_df["rank_ic"].std()

    if pd.isna(std_ic):
        return float("nan")

    if std_ic == 0:
        return 0.0

    return mean_ic / std_ic


def summarize_rank_ic(ic_df: pd.DataFrame) -> dict:
    if ic_df.empty:
        return {
            "observations": 0,
            "mean_ic": float("nan"),
            "std_ic": float("nan"),
            "icir": float("nan"),
            "pos_ic_ratio": float("nan"),
        }

    rank_ic = ic_df["rank_ic"].dropna()
    if rank_ic.empty:
        return {
            "observations": 0,
            "mean_ic": float("nan"),
            "std_ic": float("nan"),
            "icir": float("nan"),
            "pos_ic_ratio": float("nan"),
        }

    return {
        "observations": int(rank_ic.count()),
        "mean_ic": float(rank_ic.mean()),
        "std_ic": float(rank_ic.std()),
        "icir": float(calc_icir(pd.DataFrame({"rank_ic": rank_ic}))),
        "pos_ic_ratio": float((rank_ic > 0).mean()),
    }


def calc_ic_by_year(
    factor_scores: pd.DataFrame,
    price: pd.DataFrame,
    factor_names: list[str],
    horizon: int = 20,
) -> pd.DataFrame:
    records = []
    ic_all = calc_rank_ic_matrix(
        factor_scores,
        price,
        factor_names=factor_names,
        horizon=horizon,
    )

    for name in factor_names:
        ic_df = ic_all[ic_all["factor"] == name][["trade_date", "rank_ic"]].copy()
        if ic_df.empty:
            continue

        ic_df = ic_df.copy()
        ic_df["year"] = pd.to_datetime(
            ic_df["trade_date"].astype(str),
            format="%Y%m%d",
            errors="coerce",
        ).dt.year

        for year, group in ic_df.dropna(subset=["year"]).groupby("year"):
            summary = summarize_rank_ic(group)
            records.append({
                "factor": name,
                "year": int(year),
                **summary,
            })

    return pd.DataFrame(records)


def calc_factor_ic_summary(
    factor_scores: pd.DataFrame,
    price: pd.DataFrame,
    factor_names: list[str] | None = None,
    horizon: int = 20,
) -> dict:
    if factor_names is None:
        factor_names = [c for c in factor_scores.columns if c not in ("ts_code", "trade_date")]

    summary = {}
    ic_all = calc_rank_ic_matrix(
        factor_scores,
        price,
        factor_names=factor_names,
        horizon=horizon,
    )

    for name in factor_names:
        ic_df = ic_all[ic_all["factor"] == name][["trade_date", "rank_ic"]].copy()
        ic_summary = summarize_rank_ic(ic_df)

        summary[name] = {
            "mean_ic": ic_summary["mean_ic"],
            "std_ic": ic_summary["std_ic"],
            "icir": ic_summary["icir"],
            "pos_ic_ratio": ic_summary["pos_ic_ratio"],
        }

        logger.info(
            "Factor %s: mean_IC=%.4f, ICIR=%.4f",
            name,
            summary[name]["mean_ic"],
            summary[name]["icir"],
        )

    return summary
