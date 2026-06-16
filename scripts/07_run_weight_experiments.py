import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import pandas as pd
from tqdm import tqdm

from ashare_alpha.analysis.metrics import calc_performance
from ashare_alpha.analysis.weight_experiments import (
    calc_portfolio_factor_exposure,
    derive_ic_weighted_static_weights,
    flatten_performance_metrics,
    summarize_portfolio_factor_exposure,
)
from ashare_alpha.backtest.engine import BacktestEngine
from ashare_alpha.backtest.recorder import Recorder
from ashare_alpha.data.calendar import get_first_trading_day_of_month, get_next_trade_date, get_trade_dates
from ashare_alpha.data.loader import load_factor_panel, load_panel, load_raw
from ashare_alpha.factors.composite import calc_composite_score
from ashare_alpha.logger import init_logger, logger
from ashare_alpha.settings import load_config
from ashare_alpha.strategy.constraints import apply_position_constraints
from ashare_alpha.strategy.portfolio import generate_target_weights
from ashare_alpha.strategy.signal import generate_signal
from ashare_alpha.strategy.universe import build_universe


FACTOR_NAMES = ["value", "quality", "growth", "lowvol", "momentum", "reversal"]


def _benchmark_config(data_config: dict) -> dict[str, str]:
    benchmarks = data_config.get("benchmarks")
    if benchmarks:
        return {str(name): str(code) for name, code in benchmarks.items()}

    benchmark_code = data_config.get("benchmark_code")
    if benchmark_code:
        return {"benchmark": str(benchmark_code)}

    return {}


def _load_benchmarks(config: dict) -> dict[str, pd.DataFrame]:
    raw_dir = config["data"]["raw_dir"]
    benchmark_codes = _benchmark_config(config["data"])
    if not benchmark_codes:
        return {}

    try:
        index_daily = load_raw("index_daily", raw_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load benchmark index_daily: %s", exc)
        return {}

    benchmarks = {}
    for name, code in benchmark_codes.items():
        code_mask = index_daily["ts_code"].astype(str) == code
        frame = index_daily.loc[code_mask].copy()

        if frame.empty and "benchmark_name" in index_daily.columns:
            name_mask = index_daily["benchmark_name"].astype(str) == name
            frame = index_daily.loc[name_mask].copy()

        if frame.empty:
            logger.warning("No benchmark data found for %s (%s)", name, code)
            continue

        benchmarks[name] = frame.sort_values("trade_date")

    return benchmarks


def _load_optional_stock_status(processed_dir: str, raw_dir: str) -> pd.DataFrame | None:
    for data_dir, name, loader in [
        (processed_dir, "stock_status_panel", load_panel),
        (raw_dir, "stock_status", load_raw),
    ]:
        path = Path(data_dir) / f"{name}.parquet"
        if path.exists():
            logger.info("Loading optional stock status from %s", path)
            return loader(name, data_dir)
    logger.info("No historical stock status panel found; using current-name ST fallback")
    return None


def _load_factor_summary(result_dir: str) -> tuple[pd.DataFrame, str]:
    candidates = [
        Path(result_dir) / "factor_report_rebalance_universe" / "summary.csv",
        Path(result_dir) / "factor_report" / "summary.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path), str(path)

    raise FileNotFoundError(
        "No factor diagnostic summary found. Run scripts/06_make_report.py first."
    )


def _experiment_definitions(config: dict, factor_summary: pd.DataFrame) -> list[dict]:
    current_weights = deepcopy(config["strategy"]["factors"]["weights"])
    ic_weights = derive_ic_weighted_static_weights(factor_summary, min_mean_ic=0.01)
    if not ic_weights:
        raise ValueError("Could not derive IC-weighted static weights from factor summary")

    return [
        {
            "experiment_id": "baseline_v0_current_weights",
            "weights": current_weights,
            "is_reference": True,
            "notes": "Current strategy.yaml weights; missing leverage is renormalized by existing scorer.",
        },
        {
            "experiment_id": "baseline_v1_no_momentum",
            "weights": {
                "value": 0.25,
                "quality": 0.20,
                "growth": 0.15,
                "lowvol": 0.15,
                "reversal": 0.10,
            },
            "is_reference": False,
            "notes": "Remove positive momentum while keeping the remaining current raw weights.",
        },
        {
            "experiment_id": "baseline_v1_reverse_momentum",
            "weights": {
                "value": 0.25,
                "quality": 0.20,
                "growth": 0.15,
                "lowvol": 0.15,
                "momentum": -0.10,
                "reversal": 0.10,
            },
            "is_reference": False,
            "notes": "Use momentum as a contrarian signal with a negative score coefficient.",
        },
        {
            "experiment_id": "baseline_v1_value_lowvol_reversal",
            "weights": {
                "value": 1 / 3,
                "lowvol": 1 / 3,
                "reversal": 1 / 3,
            },
            "is_reference": False,
            "notes": "Equal-weight the three currently favored core factors.",
        },
        {
            "experiment_id": "baseline_v1_ic_weighted_static",
            "weights": ic_weights,
            "is_reference": False,
            "notes": "Static weights proportional to positive rebalance-universe mean IC, min_mean_ic=0.01.",
        },
    ]


def _effective_weights(score: pd.DataFrame) -> dict[str, float]:
    weights = {}
    for col in score.columns:
        if not col.startswith("weight_"):
            continue
        values = score[col].dropna()
        if not values.empty:
            weights[col.replace("weight_", "")] = float(values.iloc[0])
    return weights


def _build_universe_cache(
    config: dict,
    price_panel: pd.DataFrame,
    stock_basic: pd.DataFrame,
    stock_status: pd.DataFrame | None,
    trade_dates: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rebalance_dates = get_first_trading_day_of_month(trade_dates)
    universe_by_date = {}
    filter_stats = []

    for trade_date in tqdm(rebalance_dates, desc="Building rebalance universes"):
        date_data = price_panel[price_panel["trade_date"] == trade_date]
        if date_data.empty:
            continue

        universe, stats = build_universe(
            trade_date,
            price_panel,
            price_panel,
            stock_basic,
            config,
            stock_status=stock_status,
            return_filter_stats=True,
        )
        filter_stats.append(stats)
        if not universe.empty:
            universe_by_date[trade_date] = universe

    stats = pd.concat(filter_stats, ignore_index=True) if filter_stats else pd.DataFrame()
    return universe_by_date, stats


def _generate_experiment_weights(
    score: pd.DataFrame,
    universe_by_date: dict[str, pd.DataFrame],
    config: dict,
    trade_dates: list[str],
) -> pd.DataFrame:
    all_weights = []

    for trade_date in tqdm(sorted(universe_by_date), desc="Generating experiment signals"):
        universe = universe_by_date[trade_date]
        date_scores = score[score["trade_date"] == trade_date]
        if date_scores.empty:
            continue

        signal = generate_signal(date_scores, universe, config)
        if signal.empty:
            continue

        weights = generate_target_weights(signal, config)
        weights = apply_position_constraints(weights, config)
        if weights.empty:
            continue

        weights["rebalance_date"] = trade_date
        weights["execution_date"] = get_next_trade_date(trade_date, trade_dates) or trade_date
        all_weights.append(weights)

    if not all_weights:
        return pd.DataFrame()

    return pd.concat(all_weights, ignore_index=True)


def _run_backtest(
    config: dict,
    price_panel: pd.DataFrame,
    target_weights: pd.DataFrame,
    trade_dates: list[str],
) -> dict:
    bt_cfg = config["backtest"]
    start_date = bt_cfg["start_date"]
    end_date = bt_cfg["end_date"]
    backtest_dates = [d for d in trade_dates if start_date <= d <= end_date]
    rebalance_dates = sorted(
        d
        for d in target_weights["execution_date"].dropna().astype(str).unique()
        if start_date <= d <= end_date
    )

    engine = BacktestEngine(
        price_panel=price_panel,
        target_weights=target_weights,
        trade_dates=backtest_dates,
        rebalance_dates=rebalance_dates,
        init_cash=bt_cfg.get("init_cash", 10000000),
        cost_config=config["cost"],
    )
    return engine.run()


def _save_filter_stats(filter_stats: pd.DataFrame, output_dir: Path) -> None:
    if filter_stats.empty:
        return

    filter_stats.to_csv(output_dir / "universe_filter_stats.csv", index=False, encoding="utf-8-sig")
    summary = (
        filter_stats.groupby("filter", sort=False)
        .agg(
            rebalance_dates=("trade_date", "nunique"),
            avg_before=("before", "mean"),
            avg_after=("after", "mean"),
            avg_removed=("removed", "mean"),
            max_removed=("removed", "max"),
            total_removed=("removed", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "universe_filter_stats_summary.csv", index=False, encoding="utf-8-sig")


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str, ensure_ascii=False)


def _save_experiment_outputs(
    experiment: dict,
    output_dir: Path,
    config: dict,
    score: pd.DataFrame,
    target_weights: pd.DataFrame,
    results: dict,
    metrics: dict,
    exposure: pd.DataFrame,
    exposure_summary: dict,
    filter_stats: pd.DataFrame,
    factor_summary_source: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    Recorder(result_dir=str(output_dir)).save_all(results)
    target_weights.to_parquet(output_dir / "target_weights.parquet", index=False)
    exposure.to_csv(output_dir / "portfolio_factor_exposure.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([exposure_summary]).to_csv(
        output_dir / "portfolio_factor_exposure_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _save_filter_stats(filter_stats, output_dir)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(
        output_dir / "experiment_manifest.json",
        {
            "experiment_id": experiment["experiment_id"],
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "is_reference": experiment["is_reference"],
            "raw_weights": experiment["weights"],
            "effective_weights": _effective_weights(score),
            "notes": experiment["notes"],
            "data_start": config["data"]["start_date"],
            "data_end": config["data"]["end_date"],
            "backtest_start": config["backtest"]["start_date"],
            "backtest_end": config["backtest"]["end_date"],
            "factor_summary_source": factor_summary_source,
            "universe": "current pipeline universe filters",
        },
    )


def _format_pct(value) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:.2f}%"


def _format_float(value) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.4f}"


def _write_comparison_markdown(summary: pd.DataFrame, output_dir: Path) -> None:
    cols = [
        "experiment_id",
        "final_nav",
        "annual_return",
        "max_drawdown",
        "turnover",
        "csi500_annual_excess_return",
        "csi500_information_ratio",
        "avg_exposure_value",
        "avg_exposure_lowvol",
        "avg_exposure_reversal",
        "avg_exposure_momentum",
    ]
    available = [c for c in cols if c in summary.columns]
    lines = [
        "# Weight experiment comparison",
        "",
        "| " + " | ".join(available) + " |",
        "| " + " | ".join(["---"] * len(available)) + " |",
    ]
    pct_cols = {
        "annual_return",
        "max_drawdown",
        "turnover",
        "csi500_annual_excess_return",
    }
    for row in summary[available].itertuples(index=False, name=None):
        cells = []
        for col, value in zip(available, row):
            if col == "experiment_id":
                cells.append(str(value))
            elif col in pct_cols:
                cells.append(_format_pct(value))
            else:
                cells.append(_format_float(value))
        lines.append("| " + " | ".join(cells) + " |")

    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    config = load_config()
    init_logger(log_file="logs/weight_experiments.log")

    processed_dir = config["data"]["processed_dir"]
    factor_dir = config["data"]["factor_dir"]
    raw_dir = config["data"]["raw_dir"]
    result_dir = config["backtest"]["output"]["result_dir"]
    experiment_root = Path(result_dir) / "experiments" / "weight_baselines"
    experiment_root.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data ...")
    price_panel = load_panel("price_panel", processed_dir)
    factor_panel = load_factor_panel(factor_dir)
    stock_basic = load_raw("stock_basic", raw_dir)
    stock_status = _load_optional_stock_status(processed_dir, raw_dir)
    trade_dates = get_trade_dates(load_raw("trade_calendar", raw_dir))
    benchmarks = _load_benchmarks(config)
    factor_summary, factor_summary_source = _load_factor_summary(result_dir)
    experiments = _experiment_definitions(config, factor_summary)

    universe_by_date, filter_stats = _build_universe_cache(
        config,
        price_panel,
        stock_basic,
        stock_status,
        trade_dates,
    )
    logger.info("Cached %d non-empty rebalance universes", len(universe_by_date))

    summary_records = []
    for experiment in experiments:
        experiment_id = experiment["experiment_id"]
        logger.info("Running experiment: %s", experiment_id)
        output_dir = experiment_root / experiment_id

        score = calc_composite_score(factor_panel, experiment["weights"])
        target_weights = _generate_experiment_weights(score, universe_by_date, config, trade_dates)
        if target_weights.empty:
            logger.warning("No target weights generated for %s", experiment_id)
            continue

        results = _run_backtest(config, price_panel, target_weights, trade_dates)
        metrics = calc_performance(results["nav"], results["trades"], benchmarks=benchmarks)
        exposure = calc_portfolio_factor_exposure(target_weights, FACTOR_NAMES)
        exposure_summary = summarize_portfolio_factor_exposure(exposure)

        _save_experiment_outputs(
            experiment,
            output_dir,
            config,
            score,
            target_weights,
            results,
            metrics,
            exposure,
            exposure_summary,
            filter_stats,
            factor_summary_source,
        )

        summary_records.append({
            "experiment_id": experiment_id,
            "is_reference": experiment["is_reference"],
            **flatten_performance_metrics(metrics),
            **exposure_summary,
        })

    summary = pd.DataFrame(summary_records)
    summary.to_csv(experiment_root / "summary.csv", index=False, encoding="utf-8-sig")
    _write_comparison_markdown(summary, experiment_root)
    logger.info("Saved weight experiment comparison to %s", experiment_root)


if __name__ == "__main__":
    main()
