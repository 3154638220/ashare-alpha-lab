import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from ashare_alpha.settings import load_config
from ashare_alpha.logger import init_logger, logger
from ashare_alpha.data.loader import load_panel, load_raw, load_target_weights
from ashare_alpha.data.calendar import get_trade_dates, get_first_trading_day_of_month, get_next_trade_date
from ashare_alpha.backtest.engine import BacktestEngine
from ashare_alpha.backtest.recorder import Recorder


def main():
    config = load_config()
    init_logger(log_file="logs/backtest.log")

    processed_dir = config["data"]["processed_dir"]
    signal_dir = config["data"]["signal_dir"]
    raw_dir = config["data"]["raw_dir"]

    logger.info("Loading data ...")
    price_panel = load_panel("price_panel", processed_dir)
    target_weights = load_target_weights(signal_dir)

    cal = load_raw("trade_calendar", raw_dir)
    trade_dates = get_trade_dates(cal)

    bt_cfg = config["backtest"]
    start_date = bt_cfg["start_date"]
    end_date = bt_cfg["end_date"]

    trade_dates = [d for d in trade_dates if start_date <= d <= end_date]

    rebalance_dates = get_first_trading_day_of_month(trade_dates)

    target_weights = target_weights.copy()

    if "execution_date" in target_weights.columns:
        exec_dates = sorted(target_weights["execution_date"].unique())
        rebalance_dates = [d for d in exec_dates if start_date <= d <= end_date]
    else:
        rebalance_dates = get_first_trading_day_of_month(trade_dates)

    init_cash = bt_cfg.get("init_cash", 10000000)
    cost_config = config["cost"]

    logger.info(
        "Starting backtest: %s ~ %s, init_cash=%.0f, %d rebalance dates",
        start_date, end_date, init_cash, len(rebalance_dates),
    )

    engine = BacktestEngine(
        price_panel=price_panel,
        target_weights=target_weights,
        trade_dates=trade_dates,
        rebalance_dates=rebalance_dates,
        init_cash=init_cash,
        cost_config=cost_config,
    )

    results = engine.run()

    recorder = Recorder(result_dir=bt_cfg["output"]["result_dir"])
    recorder.save_all(results)

    nav = results["nav"]
    logger.info(
        "Backtest complete. Final NAV: %.4f, Final Value: %.2f",
        nav["total_value"].iloc[-1] / init_cash,
        nav["total_value"].iloc[-1],
    )

    logger.info("Results saved to %s", bt_cfg["output"]["result_dir"])


if __name__ == "__main__":
    main()
