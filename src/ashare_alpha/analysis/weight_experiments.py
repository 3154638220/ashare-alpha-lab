import numpy as np
import pandas as pd


FACTOR_NAMES = ["value", "quality", "growth", "lowvol", "momentum", "reversal"]


def derive_ic_weighted_static_weights(
    factor_summary: pd.DataFrame,
    min_mean_ic: float = 0.01,
) -> dict[str, float]:
    if factor_summary.empty:
        return {}

    required = {"factor", "mean_ic"}
    if not required.issubset(factor_summary.columns):
        missing = ", ".join(sorted(required - set(factor_summary.columns)))
        raise ValueError(f"factor_summary is missing required columns: {missing}")

    summary = factor_summary.copy()
    summary["mean_ic"] = pd.to_numeric(summary["mean_ic"], errors="coerce")
    summary = summary[summary["mean_ic"] >= min_mean_ic]
    summary = summary[summary["factor"].isin(FACTOR_NAMES)]

    total = summary["mean_ic"].sum()
    if not np.isfinite(total) or total <= 0:
        return {}

    return {
        str(row.factor): float(row.mean_ic / total)
        for row in summary.itertuples(index=False)
    }


def flatten_performance_metrics(metrics: dict) -> dict:
    keys = [
        "final_nav",
        "total_return",
        "annual_return",
        "annual_vol",
        "sharpe",
        "max_drawdown",
        "calmar",
        "turnover",
    ]
    flat = {key: metrics.get(key, np.nan) for key in keys}

    for benchmark_name, benchmark_metrics in metrics.get("benchmarks", {}).items():
        for metric_name in [
            "benchmark_total_return",
            "active_total_return",
            "excess_return",
            "annual_excess_return",
            "tracking_error",
            "information_ratio",
            "max_excess_drawdown",
        ]:
            flat[f"{benchmark_name}_{metric_name}"] = benchmark_metrics.get(metric_name, np.nan)

    return flat


def calc_portfolio_factor_exposure(
    target_weights: pd.DataFrame,
    factor_names: list[str] | None = None,
) -> pd.DataFrame:
    if factor_names is None:
        factor_names = FACTOR_NAMES

    if target_weights.empty:
        return pd.DataFrame()

    date_col = "rebalance_date" if "rebalance_date" in target_weights.columns else "trade_date"
    records = []

    for rebalance_date, group in target_weights.groupby(date_col, sort=True):
        weight_sum = group["target_weight"].sum()
        record = {
            "rebalance_date": rebalance_date,
            "holding_count": int(group["ts_code"].nunique()),
            "total_weight": float(weight_sum),
            "max_stock_weight": float(group["target_weight"].max()),
        }

        if "industry_code" in group.columns and not group.empty:
            industry_weight = group.groupby("industry_code", dropna=False)["target_weight"].sum()
            record["max_industry_weight"] = float(industry_weight.max())

        for factor_name in factor_names:
            exposure_col = f"exposure_{factor_name}"
            if factor_name not in group.columns or weight_sum <= 0:
                record[exposure_col] = np.nan
                continue
            values = pd.to_numeric(group[factor_name], errors="coerce")
            weights = group["target_weight"].where(values.notna(), 0.0)
            denom = weights.sum()
            record[exposure_col] = (
                float((values.fillna(0.0) * group["target_weight"]).sum() / denom)
                if denom > 0
                else np.nan
            )

        records.append(record)

    return pd.DataFrame(records)


def summarize_portfolio_factor_exposure(exposure: pd.DataFrame) -> dict:
    if exposure.empty:
        return {}

    summary = {
        "avg_holding_count": float(exposure["holding_count"].mean()),
        "avg_total_weight": float(exposure["total_weight"].mean()),
        "avg_max_stock_weight": float(exposure["max_stock_weight"].mean()),
    }

    if "max_industry_weight" in exposure.columns:
        summary["avg_max_industry_weight"] = float(exposure["max_industry_weight"].mean())
        summary["max_observed_industry_weight"] = float(exposure["max_industry_weight"].max())

    for col in [c for c in exposure.columns if c.startswith("exposure_")]:
        summary[f"avg_{col}"] = float(exposure[col].mean())

    return summary
