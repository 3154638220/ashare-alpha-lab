import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from ashare_alpha.settings import load_config
from ashare_alpha.logger import init_logger, logger
from ashare_alpha.data.loader import load_raw
from ashare_alpha.data.preprocess import build_price_panel
from ashare_alpha.data.calendar import get_trade_dates


def main():
    config = load_config()
    init_logger(log_file="logs/build_panel.log")

    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw data ...")
    daily = load_raw("daily", raw_dir)
    adj = load_raw("adj_factor", raw_dir)
    daily_basic = load_raw("daily_basic", raw_dir)
    stk_limit = load_raw("stk_limit", raw_dir)

    logger.info("Building price panel ...")
    price_panel = build_price_panel(daily, adj, daily_basic, stk_limit)

    price_path = Path(processed_dir) / "price_panel.parquet"
    price_panel.to_parquet(price_path, index=False)
    logger.info("Saved price_panel to %s (%d rows)", price_path, len(price_panel))

    cal = load_raw("trade_calendar", raw_dir)
    trade_dates = get_trade_dates(cal)

    logger.info("Loading fundamental and industry data ...")
    fina = load_raw("fina_indicator", raw_dir)

    from ashare_alpha.data.asof import build_fundamental_asof, build_industry_asof

    fundamental = build_fundamental_asof(trade_dates, fina)
    fund_path = Path(processed_dir) / "fundamental_asof.parquet"
    fundamental.to_parquet(fund_path, index=False)
    logger.info("Saved fundamental_asof to %s (%d rows)", fund_path, len(fundamental))

    try:
        industry = load_raw("industry_member", raw_dir)
        industry_asof = build_industry_asof(trade_dates, industry)
        ind_path = Path(processed_dir) / "industry_asof.parquet"
        industry_asof.to_parquet(ind_path, index=False)
        logger.info("Saved industry_asof to %s (%d rows)", ind_path, len(industry_asof))
    except Exception as e:
        logger.warning("Could not build industry_asof: %s", e)

    logger.info("Build panel complete.")


if __name__ == "__main__":
    main()
