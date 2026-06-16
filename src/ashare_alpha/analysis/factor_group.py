import numpy as np
import pandas as pd

from .factor_ic import calc_forward_return, calc_forward_return_from_prices


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
            monotonicity = _safe_corr(
                by_group["group"].rank(),
                by_group["mean_forward_return"].rank(),
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


def calc_factor_payoff_by_date(
    factor_scores: pd.DataFrame,
    price: pd.DataFrame,
    factor_names: list[str],
    horizon: int = 20,
    n_groups: int = 5,
    min_obs: int = 20,
    forward_returns: pd.DataFrame | None = None,
    return_label: str = "close_to_close",
) -> pd.DataFrame:
    if forward_returns is None:
        forward_returns = calc_forward_return(price, horizon=horizon)

    records = []

    for factor_name in factor_names:
        if factor_name not in factor_scores.columns:
            continue

        cols = ["ts_code", "trade_date", factor_name]
        df = factor_scores[cols].merge(
            forward_returns,
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

            group_returns = grouped.groupby("group")["future_return"].mean()
            low_group = int(group_returns.index.min())
            high_group = int(group_returns.index.max())
            low_return = float(group_returns.loc[low_group])
            high_return = float(group_returns.loc[high_group])
            long_short_return = high_return - low_return

            factor_values = grouped[factor_name].astype(float)
            returns = grouped["future_return"].astype(float)
            rank_ic = _safe_corr(factor_values.rank(), returns.rank())
            pearson_ic = _safe_corr(factor_values, returns)

            factor_std = factor_values.std()
            if pd.isna(factor_std) or factor_std == 0:
                standardized_payoff = np.nan
            else:
                standardized_factor = (factor_values - factor_values.mean()) / factor_std
                standardized_payoff = float(
                    (standardized_factor * (returns - returns.mean())).mean()
                )

            rank_sign = np.sign(rank_ic) if pd.notna(rank_ic) else 0
            long_short_sign = np.sign(long_short_return)
            sign_conflict = (
                bool(rank_sign != 0 and long_short_sign != 0 and rank_sign != long_short_sign)
            )

            records.append({
                "trade_date": trade_date,
                "factor": factor_name,
                "return_label": return_label,
                "observations": int(len(grouped)),
                "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
                "pearson_ic": float(pearson_ic) if pd.notna(pearson_ic) else np.nan,
                "standardized_payoff": standardized_payoff,
                "market_forward_return": float(returns.mean()),
                "return_dispersion": float(returns.std()),
                "low_group": low_group,
                "high_group": high_group,
                "low_group_return": low_return,
                "high_group_return": high_return,
                "long_short_return": float(long_short_return),
                "rank_ic_long_short_sign_conflict": sign_conflict,
            })

    return pd.DataFrame(records)


def calc_execution_payoff_by_date(
    factor_scores: pd.DataFrame,
    price: pd.DataFrame,
    factor_names: list[str],
    horizon: int = 20,
    signal_lag: int = 1,
    n_groups: int = 5,
    min_obs: int = 20,
) -> pd.DataFrame:
    execution_returns = calc_forward_return_from_prices(
        price,
        horizon=horizon,
        start_price_col="adj_open",
        end_price_col="adj_close",
        start_lag=signal_lag,
        end_lag=signal_lag + horizon,
    )
    return calc_factor_payoff_by_date(
        factor_scores=factor_scores,
        price=price,
        factor_names=factor_names,
        horizon=horizon,
        n_groups=n_groups,
        min_obs=min_obs,
        forward_returns=execution_returns,
        return_label=f"open_lag_{signal_lag}_to_close_{signal_lag + horizon}",
    )


def _annualize_return(mean_return: float, horizon: int) -> float:
    if pd.isna(mean_return) or mean_return <= -1:
        return float("nan")
    return float((1 + mean_return) ** (252 / horizon) - 1)


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    valid = pd.concat([left, right], axis=1).dropna()
    if len(valid) < 2:
        return float("nan")
    if valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))


def summarize_factor_payoff_by_date(
    payoff_by_date: pd.DataFrame,
    horizon: int = 20,
) -> pd.DataFrame:
    if payoff_by_date.empty:
        return pd.DataFrame(columns=[
            "factor",
            "return_label",
            "observations",
            "mean_rank_ic",
            "mean_pearson_ic",
            "mean_standardized_payoff",
            "mean_long_short_return",
            "annualized_long_short_return",
            "long_short_std",
            "rank_ic_pos_ratio",
            "long_short_pos_ratio",
            "rank_ic_long_short_sign_conflict_ratio",
            "rank_ic_long_short_corr",
            "return_weighted_rank_ic",
            "mean_long_short_when_rank_ic_positive",
            "mean_long_short_when_rank_ic_negative",
            "worst_long_short_return",
            "best_long_short_return",
        ])

    records = []
    for (factor_name, return_label), group in payoff_by_date.groupby(["factor", "return_label"]):
        rank_ic = group["rank_ic"].dropna()
        long_short = group["long_short_return"].dropna()
        abs_ls = long_short.abs()
        if not rank_ic.empty and abs_ls.reindex(rank_ic.index).sum() > 0:
            weights = abs_ls.reindex(rank_ic.index).fillna(0.0)
            return_weighted_rank_ic = float((rank_ic * weights).sum() / weights.sum())
        else:
            return_weighted_rank_ic = np.nan

        ic_positive = group[group["rank_ic"] > 0]
        ic_negative = group[group["rank_ic"] < 0]
        mean_ls = float(long_short.mean()) if not long_short.empty else np.nan

        records.append({
            "factor": factor_name,
            "return_label": return_label,
            "observations": int(group["trade_date"].nunique()),
            "mean_rank_ic": float(group["rank_ic"].mean()),
            "mean_pearson_ic": float(group["pearson_ic"].mean()),
            "mean_standardized_payoff": float(group["standardized_payoff"].mean()),
            "mean_long_short_return": mean_ls,
            "annualized_long_short_return": _annualize_return(mean_ls, horizon),
            "long_short_std": float(group["long_short_return"].std()),
            "rank_ic_pos_ratio": float((group["rank_ic"] > 0).mean()),
            "long_short_pos_ratio": float((group["long_short_return"] > 0).mean()),
            "rank_ic_long_short_sign_conflict_ratio": float(
                group["rank_ic_long_short_sign_conflict"].mean()
            ),
            "rank_ic_long_short_corr": _safe_corr(group["rank_ic"], group["long_short_return"]),
            "return_weighted_rank_ic": return_weighted_rank_ic,
            "mean_long_short_when_rank_ic_positive": float(
                ic_positive["long_short_return"].mean()
            ) if not ic_positive.empty else np.nan,
            "mean_long_short_when_rank_ic_negative": float(
                ic_negative["long_short_return"].mean()
            ) if not ic_negative.empty else np.nan,
            "worst_long_short_return": float(group["long_short_return"].min()),
            "best_long_short_return": float(group["long_short_return"].max()),
        })

    return pd.DataFrame(records)
