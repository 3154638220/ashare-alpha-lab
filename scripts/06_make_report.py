import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import pandas as pd

from ashare_alpha.settings import load_config
from ashare_alpha.logger import init_logger, logger
from ashare_alpha.analysis.metrics import calc_performance, save_metrics, align_nav_with_benchmark
from ashare_alpha.analysis.factor_ic import calc_factor_ic_summary
from ashare_alpha.analysis.factor_report import generate_factor_diagnostics_report
from ashare_alpha.analysis.attribution import calc_industry_exposure
from ashare_alpha.visualize.report import generate_report
from ashare_alpha.data.loader import load_panel, load_factor_score, load_raw


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


def _save_excess_nav(nav: pd.DataFrame, benchmarks: dict[str, pd.DataFrame], result_dir: str) -> None:
    records = []
    for name, benchmark in benchmarks.items():
        aligned = align_nav_with_benchmark(nav, benchmark)
        if aligned.empty:
            logger.warning("No overlapping dates for benchmark %s", name)
            continue
        aligned.insert(1, "benchmark", name)
        records.append(aligned)

    if not records:
        return

    excess_nav = pd.concat(records, ignore_index=True)
    path = Path(result_dir) / "excess_nav.csv"
    excess_nav.to_csv(path, index=False)
    logger.info("Saved excess NAV to %s", path)


def _save_universe_filter_summary(signal_dir: str, result_dir: str) -> None:
    stats_path = Path(signal_dir) / "universe_filter_stats.csv"
    if not stats_path.exists():
        logger.info("No universe filter stats found at %s", stats_path)
        return

    stats = pd.read_csv(stats_path)
    if stats.empty:
        logger.info("Universe filter stats are empty")
        return

    summary = (
        stats.groupby("filter", sort=False)
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

    path = Path(result_dir) / "universe_filter_stats_summary.csv"
    summary.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Saved universe filter summary to %s", path)


def main():
    config = load_config()
    init_logger(log_file="logs/report.log")

    result_dir = config["backtest"]["output"]["result_dir"]
    processed_dir = config["data"]["processed_dir"]
    factor_dir = config["data"]["factor_dir"]
    signal_dir = config["data"]["signal_dir"]

    logger.info("Loading results ...")
    nav = pd.read_csv(Path(result_dir) / "nav.csv")
    trades = pd.read_csv(Path(result_dir) / "trades.csv")
    positions = pd.read_csv(Path(result_dir) / "positions.csv")
    benchmarks = _load_benchmarks(config)

    logger.info("Calculating performance metrics ...")
    metrics = calc_performance(nav, trades, benchmarks=benchmarks)

    logger.info("Performance metrics:")
    logger.info(json.dumps(metrics, indent=2, default=str))

    save_metrics(metrics, str(Path(result_dir) / "metrics.json"))
    logger.info("Saved metrics to %s", Path(result_dir) / "metrics.json")
    _save_excess_nav(nav, benchmarks, result_dir)
    _save_universe_filter_summary(signal_dir, result_dir)

    try:
        price_panel = load_panel("price_panel", processed_dir)
        factor_scores = load_factor_score(factor_dir)

        factor_names = ["value", "quality", "growth", "lowvol", "momentum", "reversal"]
        ic_summary = calc_factor_ic_summary(factor_scores, price_panel, factor_names)

        ic_path = Path(result_dir) / "factor_ic.json"
        with open(ic_path, "w", encoding="utf-8") as f:
            json.dump(ic_summary, f, indent=2, default=str)
        logger.info("Saved factor IC to %s", ic_path)

        factor_report_dir = Path(result_dir) / "factor_report"
        generate_factor_diagnostics_report(
            factor_scores,
            price_panel,
            output_dir=factor_report_dir,
            factor_names=factor_names,
        )
        logger.info("Saved factor diagnostics report to %s", factor_report_dir)

        rebalance_universe_path = Path(signal_dir) / "rebalance_universe.parquet"
        if rebalance_universe_path.exists():
            rebalance_universe = pd.read_parquet(rebalance_universe_path)
            rebalance_factor_report_dir = Path(result_dir) / "factor_report_rebalance_universe"
            generate_factor_diagnostics_report(
                factor_scores,
                price_panel,
                output_dir=rebalance_factor_report_dir,
                factor_names=factor_names,
                eligible_universe=rebalance_universe,
            )
            logger.info(
                "Saved rebalance-universe factor diagnostics report to %s",
                rebalance_factor_report_dir,
            )
        else:
            logger.info("No rebalance universe found at %s", rebalance_universe_path)

    except Exception as e:
        logger.warning("Could not calculate factor IC: %s", e)
        ic_summary = None

    try:
        industry = load_panel("industry_asof", processed_dir)
        exposure = calc_industry_exposure(positions, industry)
    except Exception as e:
        logger.warning("Could not calculate industry exposure: %s", e)
        exposure = None

    logger.info("Generating charts ...")
    generate_report(
        nav=nav,
        trades=trades,
        positions=positions,
        benchmarks=benchmarks,
        exposure=exposure,
        output_dir=result_dir,
        figures_dir=str(Path(result_dir) / "figures"),
    )

    logger.info("Report generation complete.")


if __name__ == "__main__":
    main()
