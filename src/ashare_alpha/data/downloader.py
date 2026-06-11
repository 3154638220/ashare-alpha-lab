from pathlib import Path
import pandas as pd

from ashare_alpha.logger import logger


def _normalise_benchmark_codes(
    benchmark_codes: dict[str, str] | list[str] | tuple[str, ...] | None,
    default_code: str = "000905.SH",
) -> dict[str, str]:
    if not benchmark_codes:
        return {"benchmark": default_code}

    if isinstance(benchmark_codes, dict):
        return {str(name): str(code) for name, code in benchmark_codes.items()}

    return {str(code).split(".")[0].lower(): str(code) for code in benchmark_codes}


class TushareDownloader:
    def __init__(self, pro, raw_dir: str):
        self.pro = pro
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def save(self, df: pd.DataFrame, name: str) -> None:
        path = self.raw_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved %s to %s (%d rows)", name, path, len(df))

    def download_stock_basic(self) -> pd.DataFrame:
        logger.info("Downloading stock_basic ...")
        df = self.pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,exchange,list_date,delist_date,is_hs",
        )
        self.save(df, "stock_basic")
        return df

    def download_trade_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        logger.info("Downloading trade_calendar %s ~ %s ...", start_date, end_date)
        df = self.pro.trade_cal(
            exchange="SSE",
            start_date=start_date,
            end_date=end_date,
        )
        self.save(df, "trade_calendar")
        return df

    def download_industry_member(self) -> pd.DataFrame:
        logger.info("Downloading industry_member ...")
        df = self.pro.index_member_all(
            index_code="SW.L1",
        )
        self.save(df, "industry_member")
        return df

    def download_daily_by_date(self, trade_date: str) -> pd.DataFrame:
        return self.pro.daily(trade_date=trade_date)

    def download_adj_factor_by_date(self, trade_date: str) -> pd.DataFrame:
        return self.pro.adj_factor(trade_date=trade_date)

    def download_daily_basic_by_date(self, trade_date: str) -> pd.DataFrame:
        return self.pro.daily_basic(trade_date=trade_date)

    def download_stk_limit_by_date(self, trade_date: str) -> pd.DataFrame:
        return self.pro.stk_limit(trade_date=trade_date)

    def download_index_daily(
        self,
        benchmark_codes: dict[str, str] | list[str] | tuple[str, ...] | None,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        logger.info("Downloading benchmark index daily %s ~ %s ...", start_date, end_date)
        records = []

        for name, ts_code in _normalise_benchmark_codes(benchmark_codes).items():
            df = self.pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df.empty:
                logger.warning("Empty index daily data for %s %s", name, ts_code)
                continue
            df = df.copy()
            df["benchmark_name"] = name
            records.append(df)

        if not records:
            raise RuntimeError("No benchmark index daily data downloaded")

        result = pd.concat(records, ignore_index=True)
        result = result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        self.save(result, "index_daily")
        return result

    def download_fina_indicator(self, start_date: str, end_date: str) -> pd.DataFrame:
        logger.info("Downloading fina_indicator %s ~ %s ...", start_date, end_date)
        df = self.pro.fina_indicator(
            start_date=start_date,
            end_date=end_date,
            fields=(
                "ts_code,ann_date,end_date,roe,roe_dt,roa,"
                "grossprofit_margin,netprofit_margin,debt_to_assets,"
                "ocf_to_or,or_yoy,netprofit_yoy"
            ),
        )
        self.save(df, "fina_indicator")
        return df
