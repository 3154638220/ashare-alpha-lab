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

    cal = load_raw("trade_calendar", raw_dir)
    trade_dates = get_trade_dates(cal)

    rebalance_dates = get_first_trading_day_of_month(trade_dates)
    logger.info("Rebalance dates: %d", len(rebalance_dates))

    all_weights = []

    for trade_date in tqdm(rebalance_dates, desc="Generating signals"):
        try:
            date_data = price_panel[price_panel["trade_date"] == trade_date]
            if date_data.empty:
                continue

            date_scores = factor_scores[factor_scores["trade_date"] == trade_date]
            if date_scores.empty:
                continue

            universe = build_universe(
                trade_date, price_panel, price_panel, stock_basic, config
            )

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

    logger.info("Signal generation complete.")


if __name__ == "__main__":
    main()
