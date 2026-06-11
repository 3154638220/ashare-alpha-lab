import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from tqdm import tqdm

from ashare_alpha.settings import load_config
from ashare_alpha.logger import init_logger, logger
from ashare_alpha.data.akshare_downloader import AkshareDownloader
from ashare_alpha.data.tushare_client import get_tushare_client
from ashare_alpha.data.downloader import TushareDownloader


def main():
    config = load_config()
    init_logger(log_file="logs/download.log")

    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    source = config["data"].get("source", "akshare")

    if source == "akshare":
        download_cfg = config["data"].get("download", {})
        downloader = AkshareDownloader(
            raw_dir=config["data"]["raw_dir"],
            start_date=start_date,
            end_date=end_date,
            benchmark_code=config["data"].get("benchmark_code", "000905.SH"),
            benchmark_codes=config["data"].get("benchmarks"),
            max_workers=download_cfg.get("max_workers", 4),
            retries=download_cfg.get("retries", 3),
            retry_sleep=download_cfg.get("retry_sleep", 1.0),
        )
        downloader.download_all()
        logger.info("AkShare download complete.")
        return

    token = config["data"]["tushare_token"]
    pro = get_tushare_client(token)
    downloader = TushareDownloader(pro, config["data"]["raw_dir"])

    downloader.download_stock_basic()
    cal = downloader.download_trade_calendar(start_date, end_date)
    downloader.download_index_daily(config["data"].get("benchmarks"), start_date, end_date)

    trade_dates = cal.loc[cal["is_open"] == 1, "cal_date"].sort_values().tolist()

    logger.info("Downloading daily data for %d trade dates ...", len(trade_dates))

    daily_list = []
    adj_list = []
    basic_list = []
    limit_list = []

    for trade_date in tqdm(trade_dates, desc="Downloading daily"):
        daily_list.append(downloader.download_daily_by_date(trade_date))
        adj_list.append(downloader.download_adj_factor_by_date(trade_date))
        basic_list.append(downloader.download_daily_basic_by_date(trade_date))
        limit_list.append(downloader.download_stk_limit_by_date(trade_date))

    downloader.save(pd.concat(daily_list), "daily")
    downloader.save(pd.concat(adj_list), "adj_factor")
    downloader.save(pd.concat(basic_list), "daily_basic")
    downloader.save(pd.concat(limit_list), "stk_limit")

    downloader.download_fina_indicator(start_date, end_date)

    logger.info("Download complete.")


if __name__ == "__main__":
    main()
