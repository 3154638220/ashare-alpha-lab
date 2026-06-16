from copy import deepcopy

import numpy as np
import pandas as pd


FACTOR_NAMES = ["value", "quality", "growth", "lowvol", "momentum", "reversal"]
COST_RATE_KEYS = ("commission_rate", "stamp_tax_rate", "exchange_fee_rate", "slippage_rate")


def make_equal_factor_weights(factor_names: list[str] | tuple[str, ...]) -> dict[str, float]:
    names = [str(name) for name in factor_names]
    if not names:
        raise ValueError("factor_names must contain at least one factor")
    if len(set(names)) != len(names):
        raise ValueError("factor_names must not contain duplicate factors")

    weight = 1.0 / len(names)
    return {name: weight for name in names}


def build_reversal_ablation_experiments() -> list[dict]:
    return [
        {
            "experiment_id": "baseline_v1_reversal_only",
            "weights": make_equal_factor_weights(["reversal"]),
            "is_reference": False,
            "notes": "Single-factor reversal baseline for checking whether paired-factor returns are mostly reversal-driven.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth",
            "weights": make_equal_factor_weights(["reversal", "growth"]),
            "is_reference": False,
            "notes": "Equal-weight reversal and growth to test whether growth adds payoff beyond reversal.",
        },
        {
            "experiment_id": "baseline_v1_reversal_lowvol",
            "weights": make_equal_factor_weights(["reversal", "lowvol"]),
            "is_reference": False,
            "notes": "Equal-weight reversal and lowvol to test lowvol's contribution despite payoff conflict.",
        },
        {
            "experiment_id": "baseline_v1_reversal_value",
            "weights": make_equal_factor_weights(["reversal", "value"]),
            "is_reference": False,
            "notes": "Equal-weight reversal and value to test value's contribution despite payoff conflict.",
        },
    ]


def build_reversal_growth_validation_experiments() -> list[dict]:
    reversal_growth_weights = make_equal_factor_weights(["reversal", "growth"])
    return [
        {
            "experiment_id": "baseline_v1_growth_only",
            "weights": make_equal_factor_weights(["growth"]),
            "is_reference": False,
            "notes": "Single-factor growth control to check whether reversal_growth is truly complementary.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_cost_2x",
            "weights": reversal_growth_weights,
            "cost_multiplier": 2.0,
            "is_reference": False,
            "notes": "Equal-weight reversal and growth under 2x trading cost stress.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_cost_3x",
            "weights": reversal_growth_weights,
            "cost_multiplier": 3.0,
            "is_reference": False,
            "notes": "Equal-weight reversal and growth under 3x trading cost stress.",
        },
    ]


def build_reversal_growth_turnover_control_experiments() -> list[dict]:
    reversal_growth_weights = make_equal_factor_weights(["reversal", "growth"])
    return [
        {
            "experiment_id": "baseline_v1_reversal_growth_buffer_70",
            "weights": reversal_growth_weights,
            "selection": {
                "method": "rank_buffer",
                "top_n": 50,
                "entry_rank": 50,
                "exit_rank": 70,
            },
            "is_reference": False,
            "notes": "Equal-weight reversal and growth with Top50 entry and narrower Top70 holding buffer.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_buffer_80",
            "weights": reversal_growth_weights,
            "selection": {
                "method": "rank_buffer",
                "top_n": 50,
                "entry_rank": 50,
                "exit_rank": 80,
            },
            "is_reference": False,
            "notes": "Equal-weight reversal and growth with Top50 entry and Top80 holding buffer.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_buffer_100",
            "weights": reversal_growth_weights,
            "selection": {
                "method": "rank_buffer",
                "top_n": 50,
                "entry_rank": 50,
                "exit_rank": 100,
            },
            "is_reference": False,
            "notes": "Equal-weight reversal and growth with Top50 entry and wider Top100 holding buffer.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_top60_buffer_90",
            "weights": reversal_growth_weights,
            "selection": {
                "method": "rank_buffer",
                "top_n": 60,
                "entry_rank": 60,
                "exit_rank": 90,
            },
            "is_reference": False,
            "notes": "Equal-weight reversal and growth with Top60 entry and Top90 holding buffer.",
        },
    ]


def build_reversal_growth_buffer_cost_experiments() -> list[dict]:
    reversal_growth_weights = make_equal_factor_weights(["reversal", "growth"])
    return [
        {
            "experiment_id": "baseline_v1_reversal_growth_buffer_80_cost_2x",
            "weights": reversal_growth_weights,
            "selection": {
                "method": "rank_buffer",
                "top_n": 50,
                "entry_rank": 50,
                "exit_rank": 80,
            },
            "cost_multiplier": 2.0,
            "is_reference": False,
            "notes": "Top50/Top80 buffered reversal_growth under 2x trading cost stress.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_buffer_80_cost_3x",
            "weights": reversal_growth_weights,
            "selection": {
                "method": "rank_buffer",
                "top_n": 50,
                "entry_rank": 50,
                "exit_rank": 80,
            },
            "cost_multiplier": 3.0,
            "is_reference": False,
            "notes": "Top50/Top80 buffered reversal_growth under 3x trading cost stress.",
        },
    ]


def scale_cost_config(cost_config: dict, multiplier: float) -> dict:
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")

    scaled = deepcopy(cost_config)
    for key in COST_RATE_KEYS:
        if key in scaled:
            scaled[key] = float(scaled[key]) * multiplier
    return scaled


def _merge_universe_industry(score: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    industry_cols = [c for c in ["industry_code", "industry_name"] if c in universe.columns]
    if not industry_cols:
        return score

    industry_meta = universe[["ts_code"] + industry_cols].drop_duplicates("ts_code")
    out = score.merge(industry_meta, on="ts_code", how="left", suffixes=("", "_universe"))
    for col in industry_cols:
        universe_col = f"{col}_universe"
        if universe_col in out.columns:
            out[col] = out[col].combine_first(out[universe_col])
            out = out.drop(columns=[universe_col])
    return out


def build_rank_buffered_signal(
    score: pd.DataFrame,
    universe: pd.DataFrame,
    previous_holdings: set[str] | list[str] | tuple[str, ...] | None,
    top_n: int = 50,
    entry_rank: int | None = None,
    exit_rank: int = 80,
) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    if entry_rank is None:
        entry_rank = top_n

    if entry_rank <= 0 or exit_rank <= 0:
        raise ValueError("entry_rank and exit_rank must be positive")
    if exit_rank < entry_rank:
        raise ValueError("exit_rank must be greater than or equal to entry_rank")

    valid_codes = universe["ts_code"].unique()
    candidates = score[score["ts_code"].isin(valid_codes)].copy()
    if candidates.empty:
        return candidates

    candidates = _merge_universe_industry(candidates, universe)
    candidates = candidates.sort_values("score", ascending=False).reset_index(drop=True)
    candidates["rank"] = range(1, len(candidates) + 1)

    previous = {str(code) for code in (previous_holdings or [])}
    if not previous:
        selected = candidates.head(top_n).copy()
        selected["is_buffer_retained"] = False
    else:
        code_str = candidates["ts_code"].astype(str)
        retained = candidates[(code_str.isin(previous)) & (candidates["rank"] <= exit_rank)].copy()
        retained = retained.sort_values("rank").head(top_n)
        retained["is_buffer_retained"] = True

        selected_codes = set(retained["ts_code"].astype(str))
        fill_pool = candidates[
            (~candidates["ts_code"].astype(str).isin(selected_codes))
            & (candidates["rank"] <= entry_rank)
        ].copy()
        fill_pool["is_buffer_retained"] = False

        need = top_n - len(retained)
        selected = pd.concat([retained, fill_pool.head(max(need, 0))], ignore_index=True)

        if len(selected) < top_n:
            selected_codes = set(selected["ts_code"].astype(str))
            fallback = candidates[~candidates["ts_code"].astype(str).isin(selected_codes)].copy()
            fallback["is_buffer_retained"] = False
            selected = pd.concat(
                [selected, fallback.head(top_n - len(selected))],
                ignore_index=True,
            )

    selected = selected.sort_values("rank").head(top_n).copy()
    selected["buffer_entry_rank"] = int(entry_rank)
    selected["buffer_exit_rank"] = int(exit_rank)
    return selected


def calc_rebalance_turnover(target_weights: pd.DataFrame) -> pd.DataFrame:
    if target_weights.empty:
        return pd.DataFrame()

    date_col = "rebalance_date" if "rebalance_date" in target_weights.columns else "trade_date"
    records = []
    previous_weights: dict[str, float] | None = None

    for rebalance_date, group in target_weights.groupby(date_col, sort=True):
        current_weights = {
            str(row.ts_code): float(row.target_weight)
            for row in group[["ts_code", "target_weight"]].itertuples(index=False)
        }
        current_codes = set(current_weights)

        record = {
            "rebalance_date": rebalance_date,
            "holding_count": int(len(current_codes)),
        }

        if previous_weights is None:
            record.update({
                "retained_count": pd.NA,
                "added_count": pd.NA,
                "removed_count": pd.NA,
                "name_retention": pd.NA,
                "target_weight_turnover": pd.NA,
            })
        else:
            previous_codes = set(previous_weights)
            all_codes = previous_codes | current_codes
            retained = previous_codes & current_codes
            abs_diff = sum(
                abs(current_weights.get(code, 0.0) - previous_weights.get(code, 0.0))
                for code in all_codes
            )
            record.update({
                "retained_count": int(len(retained)),
                "added_count": int(len(current_codes - previous_codes)),
                "removed_count": int(len(previous_codes - current_codes)),
                "name_retention": (
                    float(len(retained) / len(previous_codes))
                    if previous_codes
                    else pd.NA
                ),
                "target_weight_turnover": float(0.5 * abs_diff),
            })

        records.append(record)
        previous_weights = current_weights

    return pd.DataFrame(records)


def summarize_rebalance_turnover(turnover: pd.DataFrame) -> dict:
    if turnover.empty:
        return {}

    summary = {}
    metrics = [
        "target_weight_turnover",
        "name_retention",
        "added_count",
        "removed_count",
    ]
    for metric in metrics:
        if metric in turnover.columns:
            values = pd.to_numeric(turnover[metric], errors="coerce")
            summary[f"avg_{metric}"] = float(values.mean())
    return summary


def _parse_date_series(values: pd.Series) -> pd.Series:
    as_str = values.astype(str)
    parsed = pd.to_datetime(as_str, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(as_str, errors="coerce")
    return parsed.fillna(fallback)


def _prepare_nav_series(nav: pd.DataFrame) -> pd.DataFrame:
    if nav.empty:
        return pd.DataFrame()

    out = nav.copy()
    if "nav" not in out.columns:
        if "total_value" not in out.columns:
            raise ValueError("nav must include either nav or total_value")
        out["nav"] = out["total_value"] / out["total_value"].iloc[0]

    out["trade_date_parsed"] = _parse_date_series(out["trade_date"])
    out = out.dropna(subset=["trade_date_parsed"]).sort_values("trade_date_parsed")
    return out


def calc_annual_return_diagnostics(nav: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_nav_series(nav)
    if prepared.empty:
        return pd.DataFrame()

    prepared["year"] = prepared["trade_date_parsed"].dt.year
    records = []
    for year, group in prepared.groupby("year", sort=True):
        first = group.iloc[0]
        last = group.iloc[-1]
        if first["nav"] == 0:
            annual_return = np.nan
        else:
            annual_return = float(last["nav"] / first["nav"] - 1)
        records.append({
            "year": int(year),
            "start_date": first["trade_date"],
            "end_date": last["trade_date"],
            "trading_days": int(len(group)),
            "start_nav": float(first["nav"]),
            "end_nav": float(last["nav"]),
            "return": annual_return,
        })

    return pd.DataFrame(records)


def calc_monthly_drawdown_diagnostics(nav: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_nav_series(nav)
    if prepared.empty:
        return pd.DataFrame()

    prepared["drawdown"] = prepared["nav"] / prepared["nav"].cummax() - 1
    prepared["year_month"] = prepared["trade_date_parsed"].dt.to_period("M").astype(str)

    records = []
    for year_month, group in prepared.groupby("year_month", sort=True):
        first = group.iloc[0]
        last = group.iloc[-1]
        worst_idx = group["drawdown"].idxmin()
        worst = group.loc[worst_idx]
        if first["nav"] == 0:
            monthly_return = np.nan
        else:
            monthly_return = float(last["nav"] / first["nav"] - 1)
        records.append({
            "year_month": str(year_month),
            "start_date": first["trade_date"],
            "end_date": last["trade_date"],
            "trading_days": int(len(group)),
            "monthly_return": monthly_return,
            "month_end_drawdown": float(last["drawdown"]),
            "min_drawdown": float(group["drawdown"].min()),
            "worst_drawdown_date": worst["trade_date"],
        })

    return pd.DataFrame(records)


def summarize_annual_return_diagnostics(annual_returns: pd.DataFrame) -> dict:
    if annual_returns.empty:
        return {}

    returns = pd.to_numeric(annual_returns["return"], errors="coerce")
    return {
        "min_annual_return": float(returns.min()),
        "max_annual_return": float(returns.max()),
        "negative_year_count": int((returns < 0).sum()),
    }


def summarize_monthly_drawdown_diagnostics(monthly_drawdown: pd.DataFrame) -> dict:
    if monthly_drawdown.empty:
        return {}

    monthly = monthly_drawdown.copy()
    monthly["monthly_return"] = pd.to_numeric(monthly["monthly_return"], errors="coerce")
    monthly["min_drawdown"] = pd.to_numeric(monthly["min_drawdown"], errors="coerce")

    worst_return = monthly.loc[monthly["monthly_return"].idxmin()]
    worst_drawdown = monthly.loc[monthly["min_drawdown"].idxmin()]
    return {
        "worst_month": str(worst_return["year_month"]),
        "worst_month_return": float(worst_return["monthly_return"]),
        "worst_drawdown_month": str(worst_drawdown["year_month"]),
        "worst_month_min_drawdown": float(worst_drawdown["min_drawdown"]),
    }


def summarize_rebalance_turnover_by_year(turnover: pd.DataFrame) -> pd.DataFrame:
    if turnover.empty:
        return pd.DataFrame()

    out = turnover.copy()
    out["rebalance_date_parsed"] = _parse_date_series(out["rebalance_date"])
    out = out.dropna(subset=["rebalance_date_parsed"])
    if out.empty:
        return pd.DataFrame()

    out["year"] = out["rebalance_date_parsed"].dt.year
    numeric_cols = [
        "target_weight_turnover",
        "name_retention",
        "added_count",
        "removed_count",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    aggregations = {
        "rebalance_count": ("rebalance_date", "count"),
    }
    if "target_weight_turnover" in out.columns:
        aggregations["avg_target_weight_turnover"] = ("target_weight_turnover", "mean")
        aggregations["max_target_weight_turnover"] = ("target_weight_turnover", "max")
    if "name_retention" in out.columns:
        aggregations["avg_name_retention"] = ("name_retention", "mean")
    if "added_count" in out.columns:
        aggregations["avg_added_count"] = ("added_count", "mean")
    if "removed_count" in out.columns:
        aggregations["avg_removed_count"] = ("removed_count", "mean")

    return out.groupby("year", sort=True).agg(**aggregations).reset_index()


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
