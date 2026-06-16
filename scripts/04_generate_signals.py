import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from tqdm import tqdm

from ashare_alpha.settings import load_config
from ashare_alpha.logger import init_logger, logger
from ashare_alpha.data.loader import load_panel, load_raw, load_factor_score
from ashare_alpha.data.calendar import get_trade_dates, get_first_trading_day_of_month, get_next_trade_date
from ashare_alpha.strategy.universe import build_universe
from ashare_alpha.strategy.signal import generate_signal
from ashare_alpha.strategy.portfolio import generate_target_weights
from ashare_alpha.strategy.constraints import apply_position_constraints


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


def main():
    config = load_config()
    init_logger(log_file="logs/signals.log")

    processed_dir = config["data"]["processed_dir"]
    factor_dir = config["data"]["factor_dir"]
    signal_dir = config["data"]["signal_dir"]
    raw_dir = config["data"]["raw_dir"]
    Path(signal_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading data ...")
    price_panel = load_panel("price_panel", processed_dir)
    factor_scores = load_factor_score(factor_dir)
    stock_basic = load_raw("stock_basic", raw_dir)
    stock_status = _load_optional_stock_status(processed_dir, raw_dir)

    cal = load_raw("trade_calendar", raw_dir)
    trade_dates = get_trade_dates(cal)

    rebalance_dates = get_first_trading_day_of_month(trade_dates)
    logger.info("Rebalance dates: %d", len(rebalance_dates))

    all_weights = []
    all_filter_stats = []
    all_universes = []

    for trade_date in tqdm(rebalance_dates, desc="Generating signals"):
        try:
            date_data = price_panel[price_panel["trade_date"] == trade_date]
            if date_data.empty:
                continue

            date_scores = factor_scores[factor_scores["trade_date"] == trade_date]
            if date_scores.empty:
                continue

            universe, filter_stats = build_universe(
                trade_date,
                price_panel,
                price_panel,
                stock_basic,
                config,
                stock_status=stock_status,
                return_filter_stats=True,
            )
            all_filter_stats.append(filter_stats)
            if not universe.empty:
                universe_cols = [
                    c
                    for c in ["ts_code", "trade_date", "industry_code", "industry_name"]
                    if c in universe.columns
                ]
                universe_snapshot = universe[universe_cols].copy()
                universe_snapshot["rebalance_date"] = trade_date
                all_universes.append(universe_snapshot)

            if universe.empty:
                logger.warning("Empty universe for %s", trade_date)
                continue

            signal = generate_signal(date_scores, universe, config)

            if signal.empty:
                continue

            weights = generate_target_weights(signal, config)
            weights = apply_position_constraints(weights, config)

            weights["rebalance_date"] = trade_date

            exec_date = get_next_trade_date(trade_date, trade_dates)
            weights["execution_date"] = exec_date or trade_date

            all_weights.append(weights)

        except Exception as e:
            logger.error("Error on %s: %s", trade_date, e)
            continue

    if all_weights:
        result = pd.concat(all_weights, ignore_index=True)

        weight_path = Path(signal_dir) / "target_weights.parquet"
        result.to_parquet(weight_path, index=False)
        logger.info("Saved target_weights to %s (%d rows)", weight_path, len(result))
    else:
        logger.warning("No target weights generated.")

    if all_filter_stats:
        filter_stats = pd.concat(all_filter_stats, ignore_index=True)
        stats_path = Path(signal_dir) / "universe_filter_stats.csv"
        filter_stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
        logger.info("Saved universe filter stats to %s (%d rows)", stats_path, len(filter_stats))

    if all_universes:
        rebalance_universe = pd.concat(all_universes, ignore_index=True).drop_duplicates(
            ["ts_code", "trade_date", "rebalance_date"]
        )
        universe_path = Path(signal_dir) / "rebalance_universe.parquet"
        rebalance_universe.to_parquet(universe_path, index=False)
        logger.info(
            "Saved rebalance universe to %s (%d rows)",
            universe_path,
            len(rebalance_universe),
        )

    logger.info("Signal generation complete.")


if __name__ == "__main__":
    main()
