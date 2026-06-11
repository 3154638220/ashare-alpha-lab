import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from ashare_alpha.settings import load_config
from ashare_alpha.logger import init_logger, logger
from ashare_alpha.data.loader import load_panel, load_raw
from ashare_alpha.data.calendar import get_trade_dates
from ashare_alpha.factors.value import calc_value_factor
from ashare_alpha.factors.quality import calc_quality_factor
from ashare_alpha.factors.growth import calc_growth_factor
from ashare_alpha.factors.volatility import calc_lowvol_factor
from ashare_alpha.factors.momentum import calc_momentum_reversal_factor
from ashare_alpha.factors.composite import calc_composite_score


def main():
    config = load_config()
    init_logger(log_file="logs/factors.log")

    processed_dir = config["data"]["processed_dir"]
    factor_dir = config["data"]["factor_dir"]
    raw_dir = config["data"]["raw_dir"]
    Path(factor_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading panels ...")
    price_panel = load_panel("price_panel", processed_dir)

    try:
        fundamental = load_panel("fundamental_asof", processed_dir)
    except Exception:
        logger.warning("No fundamental_asof found, using price_panel only")
        fundamental = None

    try:
        industry = load_panel("industry_asof", processed_dir)
    except Exception:
        industry = None

    cal = load_raw("trade_calendar", raw_dir)
    trade_dates = get_trade_dates(cal)

    if industry is not None:
        industry_cols = [c for c in ["industry_code", "industry_name"] if c in industry.columns]
        price_panel = price_panel.merge(
            industry[["ts_code", "trade_date"] + industry_cols],
            on=["ts_code", "trade_date"],
            how="left",
        )
        if "industry_code" not in price_panel.columns:
            price_panel["industry_code"] = "unknown"
        if "industry_name" not in price_panel.columns:
            price_panel["industry_name"] = "unknown"
    else:
        price_panel["industry_code"] = "unknown"
        price_panel["industry_name"] = "unknown"

    if fundamental is not None:
        fina_cols = [c for c in ["roe_dt", "roa", "ocf_to_or", "or_yoy", "netprofit_yoy", "debt_to_assets", "roe"]
                     if c in fundamental.columns]
        if fina_cols:
            price_panel = price_panel.merge(
                fundamental[["ts_code", "trade_date"] + fina_cols],
                on=["ts_code", "trade_date"],
                how="left",
            )

    for col in ["roe_dt", "roa", "ocf_to_or", "or_yoy", "netprofit_yoy", "debt_to_assets"]:
        if col not in price_panel.columns:
            price_panel[col] = 0.0

    logger.info("Calculating value factor ...")
    value_df = calc_value_factor(price_panel)

    logger.info("Calculating quality factor ...")
    quality_df = calc_quality_factor(price_panel)

    logger.info("Calculating growth factor ...")
    growth_df = calc_growth_factor(price_panel)

    logger.info("Calculating low volatility factor ...")
    lowvol_df = calc_lowvol_factor(price_panel)

    logger.info("Calculating momentum/reversal factors ...")
    mom_df = calc_momentum_reversal_factor(price_panel)

    logger.info("Merging factor panels ...")
    factors = value_df.merge(quality_df, on=["ts_code", "trade_date"], how="outer")
    factors = factors.merge(growth_df, on=["ts_code", "trade_date"], how="outer")
    factors = factors.merge(lowvol_df, on=["ts_code", "trade_date"], how="outer")
    factors = factors.merge(mom_df, on=["ts_code", "trade_date"], how="outer")

    factor_panel_path = Path(factor_dir) / "factor_panel.parquet"
    factors.to_parquet(factor_panel_path, index=False)
    logger.info("Saved factor_panel to %s (%d rows)", factor_panel_path, len(factors))

    logger.info("Calculating composite score ...")
    weights = config["strategy"]["factors"]["weights"]
    scores = calc_composite_score(factors, weights)

    score_path = Path(factor_dir) / "factor_score.parquet"
    scores.to_parquet(score_path, index=False)
    logger.info("Saved factor_score to %s (%d rows)", score_path, len(scores))

    logger.info("Factor calculation complete.")


if __name__ == "__main__":
    main()
