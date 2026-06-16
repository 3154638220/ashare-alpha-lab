import numpy as np
import pandas as pd

from .factor_ic import calc_forward_return


def _assign_quantile_groups(values: pd.Series, n_groups: int) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=n_groups, labels=False) + 1


def calc_factor_group_returns(
    factor_scores: pd.DataFrame,
    price: pd.DataFrame,
    factor_names: list[str],
    horizon: int = 20,
    n_groups: int = 5,
    min_obs: int = 20,
) -> pd.DataFrame:
    future_ret = calc_forward_return(price, horizon=horizon)
    records = []

    for factor_name in factor_names:
        if factor_name not in factor_scores.columns:
            continue

        cols = ["ts_code", "trade_date", factor_name]
        df = factor_scores[cols].merge(
            future_ret,
            on=["ts_code", "trade_date"],
            how="inner",
        )
        df = df.dropna(subset=[factor_name, "future_return"])

        for trade_date, group in df.groupby("trade_date"):
            if len(group) < max(min_obs, n_groups) or group[factor_name].nunique() < n_groups:
                continue

            grouped = group.copy()
            try:
                grouped["group"] = _assign_quantile_groups(grouped[factor_name], n_groups)
            except ValueError:
                continue

            day_returns = (
                grouped.groupby("group")
                .agg(
                    mean_forward_return=("future_return", "mean"),
                    count=("future_return", "count"),
                )
                .reset_index()
            )

            for row in day_returns.itertuples(index=False):
                records.append({
                    "trade_date": trade_date,
                    "factor": factor_name,
                    "group": int(row.group),
                    "mean_forward_return": float(row.mean_forward_return),
                    "count": int(row.count),
                })

    return pd.DataFrame(records)


def summarize_factor_group_returns(
    group_returns: pd.DataFrame,
    horizon: int = 20,
) -> pd.DataFrame:
    if group_returns.empty:
        return pd.DataFrame(columns=[
            "factor",
            "group",
            "mean_forward_return",
            "annualized_forward_return",
            "std_forward_return",
            "observation_dates",
            "avg_count",
            "monotonicity",
        ])

    summary = (
        group_returns.groupby(["factor", "group"])
        .agg(
            mean_forward_return=("mean_forward_return", "mean"),
            std_forward_return=("mean_forward_return", "std"),
            observation_dates=("trade_date", "nunique"),
            avg_count=("count", "mean"),
        )
        .reset_index()
    )
    summary["annualized_forward_return"] = (1 + summary["mean_forward_return"]) ** (252 / horizon) - 1
    summary["monotonicity"] = np.nan

    long_short_records = []
    for factor_name, factor_summary in summary.groupby("factor"):
        by_group = factor_summary.sort_values("group")
        if len(by_group) >= 2 and by_group["mean_forward_return"].nunique(dropna=True) > 1:
            monotonicity = by_group["group"].corr(
                by_group["mean_forward_return"],
                method="spearman",
            )
            summary.loc[summary["factor"] == factor_name, "monotonicity"] = monotonicity

        daily = group_returns[group_returns["factor"] == factor_name]
        low = daily["group"].min()
        high = daily["group"].max()
        wide = daily.pivot_table(
            index="trade_date",
            columns="group",
            values="mean_forward_return",
            aggfunc="mean",
        )
        if low in wide.columns and high in wide.columns:
            long_short = (wide[high] - wide[low]).dropna()
            if not long_short.empty:
                mean_ls = long_short.mean()
                long_short_records.append({
                    "factor": factor_name,
                    "group": "long_short",
                    "mean_forward_return": float(mean_ls),
                    "annualized_forward_return": float((1 + mean_ls) ** (252 / horizon) - 1),
                    "std_forward_return": float(long_short.std()),
                    "observation_dates": int(long_short.count()),
                    "avg_count": np.nan,
                    "monotonicity": float(summary.loc[
                        summary["factor"] == factor_name,
                        "monotonicity",
                    ].iloc[0]),
                })

    if long_short_records:
        summary = pd.concat([summary, pd.DataFrame(long_short_records)], ignore_index=True)

    return summary
