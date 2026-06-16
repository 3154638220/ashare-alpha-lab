from pathlib import Path

import numpy as np
import pandas as pd

from .factor_decay import calc_factor_decay
from .factor_group import calc_factor_group_returns, summarize_factor_group_returns
from .factor_ic import calc_factor_ic_summary, calc_ic_by_year


def calc_factor_coverage(
    factor_scores: pd.DataFrame,
    factor_names: list[str],
) -> pd.DataFrame:
    records = []
    total_by_date = factor_scores.groupby("trade_date")["ts_code"].count()

    for factor_name in factor_names:
        if factor_name not in factor_scores.columns:
            continue

        valid_by_date = factor_scores.groupby("trade_date")[factor_name].apply(lambda s: s.notna().sum())
        for trade_date, total_count in total_by_date.items():
            valid_count = int(valid_by_date.get(trade_date, 0))
            coverage = valid_count / total_count if total_count else np.nan
            records.append({
                "trade_date": trade_date,
                "factor": factor_name,
                "total_count": int(total_count),
                "valid_count": valid_count,
                "coverage": float(coverage),
            })

    return pd.DataFrame(records)


def filter_factor_scores_by_universe(
    factor_scores: pd.DataFrame,
    eligible_universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if eligible_universe is None or eligible_universe.empty:
        return factor_scores.copy()

    required = {"ts_code", "trade_date"}
    if not required.issubset(eligible_universe.columns):
        missing = ", ".join(sorted(required - set(eligible_universe.columns)))
        raise ValueError(f"eligible_universe is missing required columns: {missing}")

    scores = factor_scores.copy()
    keys = eligible_universe[["ts_code", "trade_date"]].drop_duplicates().copy()
    scores["trade_date"] = scores["trade_date"].astype(str)
    keys["trade_date"] = keys["trade_date"].astype(str)

    return scores.merge(keys, on=["ts_code", "trade_date"], how="inner")


def _factor_recommendation(row: pd.Series) -> str:
    mean_ic = row.get("mean_ic", np.nan)
    pos_ic_ratio = row.get("pos_ic_ratio", np.nan)
    long_short = row.get("long_short_mean_return", np.nan)

    if pd.notna(mean_ic) and mean_ic < 0:
        return "negative_ic_review_reverse_or_remove"
    if pd.notna(mean_ic) and abs(mean_ic) < 0.01 and pd.notna(pos_ic_ratio) and pos_ic_ratio < 0.5:
        return "weak_observe_or_downweight"
    if pd.notna(mean_ic) and mean_ic > 0.02 and pd.notna(long_short) and long_short > 0:
        return "positive_keep"
    if pd.notna(mean_ic) and mean_ic > 0:
        return "weak_positive_observe"
    return "insufficient_evidence"


def build_factor_summary(
    factor_ic: dict,
    coverage: pd.DataFrame,
    group_summary: pd.DataFrame,
    decay: pd.DataFrame,
    factor_names: list[str],
    base_horizon: int = 20,
) -> pd.DataFrame:
    records = []
    coverage_summary = (
        coverage.groupby("factor")
        .agg(
            avg_coverage=("coverage", "mean"),
            min_coverage=("coverage", "min"),
            avg_valid_count=("valid_count", "mean"),
        )
        .reset_index()
        if not coverage.empty
        else pd.DataFrame()
    )

    long_short = (
        group_summary[group_summary["group"].astype(str) == "long_short"]
        .set_index("factor")
        if not group_summary.empty
        else pd.DataFrame()
    )
    decay_base = (
        decay[decay["horizon"] == base_horizon].set_index("factor")
        if not decay.empty
        else pd.DataFrame()
    )

    for factor_name in factor_names:
        ic = factor_ic.get(factor_name, {})
        row = {
            "factor": factor_name,
            "mean_ic": ic.get("mean_ic", np.nan),
            "std_ic": ic.get("std_ic", np.nan),
            "icir": ic.get("icir", np.nan),
            "pos_ic_ratio": ic.get("pos_ic_ratio", np.nan),
        }

        if not coverage_summary.empty:
            cov = coverage_summary[coverage_summary["factor"] == factor_name]
            if not cov.empty:
                row.update({
                    "avg_coverage": float(cov.iloc[0]["avg_coverage"]),
                    "min_coverage": float(cov.iloc[0]["min_coverage"]),
                    "avg_valid_count": float(cov.iloc[0]["avg_valid_count"]),
                })

        if not long_short.empty and factor_name in long_short.index:
            ls = long_short.loc[factor_name]
            row.update({
                "long_short_mean_return": float(ls["mean_forward_return"]),
                "long_short_annualized_return": float(ls["annualized_forward_return"]),
                "group_monotonicity": float(ls["monotonicity"]),
            })

        if not decay_base.empty and factor_name in decay_base.index:
            dec = decay_base.loc[factor_name]
            row.update({
                "decay_horizon": int(base_horizon),
                "decay_mean_ic": float(dec["mean_ic"]),
                "decay_icir": float(dec["icir"]),
            })

        row["recommendation"] = _factor_recommendation(pd.Series(row))
        records.append(row)

    return pd.DataFrame(records)


def _format_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No data._"

    view = df[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(_format_value(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


def _write_factor_markdown(
    output_dir: Path,
    factor_name: str,
    summary: pd.DataFrame,
    ic_by_year: pd.DataFrame,
    group_summary: pd.DataFrame,
    decay: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    summary_row = summary[summary["factor"] == factor_name]
    year_view = ic_by_year[ic_by_year["factor"] == factor_name].sort_values("year")
    group_view = group_summary[group_summary["factor"] == factor_name].sort_values("group", key=lambda s: s.astype(str))
    decay_view = decay[decay["factor"] == factor_name].sort_values("horizon")
    coverage_view = coverage[coverage["factor"] == factor_name]

    lines = [
        f"# Factor report: {factor_name}",
        "",
        "## Summary",
        _markdown_table(summary_row, list(summary_row.columns)) if not summary_row.empty else "_No data._",
        "",
        "## IC by year",
        _markdown_table(year_view, ["year", "observations", "mean_ic", "icir", "pos_ic_ratio"]),
        "",
        "## Group returns",
        _markdown_table(
            group_view,
            [
                "group",
                "mean_forward_return",
                "annualized_forward_return",
                "observation_dates",
                "monotonicity",
            ],
        ),
        "",
        "## Decay",
        _markdown_table(decay_view, ["horizon", "observations", "mean_ic", "icir", "pos_ic_ratio"]),
        "",
        "## Coverage",
    ]

    if coverage_view.empty:
        lines.append("_No data._")
    else:
        cov_summary = pd.DataFrame([{
            "avg_coverage": coverage_view["coverage"].mean(),
            "min_coverage": coverage_view["coverage"].min(),
            "avg_valid_count": coverage_view["valid_count"].mean(),
            "min_valid_count": coverage_view["valid_count"].min(),
        }])
        lines.append(_markdown_table(cov_summary, list(cov_summary.columns)))

    (output_dir / f"{factor_name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_factor_diagnostics_report(
    factor_scores: pd.DataFrame,
    price: pd.DataFrame,
    output_dir: str | Path,
    factor_names: list[str] | None = None,
    base_horizon: int = 20,
    decay_horizons: tuple[int, ...] = (5, 10, 20, 40, 60),
    n_groups: int = 5,
    eligible_universe: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    if factor_names is None:
        ignored = {"ts_code", "trade_date", "industry_code", "industry_name", "score", "factor_count"}
        factor_names = [c for c in factor_scores.columns if c not in ignored and not c.startswith("weight_")]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_scores = filter_factor_scores_by_universe(factor_scores, eligible_universe)

    factor_ic = calc_factor_ic_summary(
        factor_scores,
        price,
        factor_names=factor_names,
        horizon=base_horizon,
    )
    ic_by_year = calc_ic_by_year(
        factor_scores,
        price,
        factor_names=factor_names,
        horizon=base_horizon,
    )
    coverage = calc_factor_coverage(factor_scores, factor_names)
    group_returns = calc_factor_group_returns(
        factor_scores,
        price,
        factor_names=factor_names,
        horizon=base_horizon,
        n_groups=n_groups,
    )
    group_summary = summarize_factor_group_returns(
        group_returns,
        horizon=base_horizon,
    )
    decay = calc_factor_decay(
        factor_scores,
        price,
        factor_names=factor_names,
        horizons=decay_horizons,
    )
    summary = build_factor_summary(
        factor_ic,
        coverage,
        group_summary,
        decay,
        factor_names=factor_names,
        base_horizon=base_horizon,
    )

    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    ic_by_year.to_csv(output_dir / "ic_by_year.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(output_dir / "group_returns.csv", index=False, encoding="utf-8-sig")
    decay.to_csv(output_dir / "decay.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output_dir / "coverage.csv", index=False, encoding="utf-8-sig")

    for factor_name in factor_names:
        _write_factor_markdown(
            output_dir,
            factor_name,
            summary,
            ic_by_year,
            group_summary,
            decay,
            coverage,
        )

    return {
        "summary": summary,
        "ic_by_year": ic_by_year,
        "group_returns": group_summary,
        "decay": decay,
        "coverage": coverage,
    }
