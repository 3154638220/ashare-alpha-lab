from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Callable

import akshare as ak
import numpy as np
import pandas as pd
from tqdm import tqdm

from ashare_alpha.logger import logger


SH_SUFFIX = ".SH"
SZ_SUFFIX = ".SZ"


def _compact_date(value: str | datetime | pd.Timestamp) -> str:
    return pd.to_datetime(value).strftime("%Y%m%d")


def _hyphen_date(value: str | datetime | pd.Timestamp) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _ts_code_from_code(code: str) -> str:
    code = str(code).zfill(6)
    suffix = SH_SUFFIX if code.startswith(("5", "6", "9")) else SZ_SUFFIX
    return f"{code}{suffix}"


def _sina_symbol_from_ts_code(ts_code: str) -> str:
    code, exchange = ts_code.split(".")
    prefix = "sh" if exchange == "SH" else "sz"
    return f"{prefix}{code}"


def _normalise_benchmark_codes(
    benchmark_codes: dict[str, str] | list[str] | tuple[str, ...] | None,
    default_code: str,
) -> dict[str, str]:
    if not benchmark_codes:
        return {"benchmark": default_code}

    if isinstance(benchmark_codes, dict):
        return {str(name): str(code) for name, code in benchmark_codes.items()}

    return {str(code).split(".")[0].lower(): str(code) for code in benchmark_codes}


def _retry(call: Callable[[], pd.DataFrame], label: str, retries: int, retry_sleep: float) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                sleep_for = retry_sleep * attempt
                logger.warning("%s failed on attempt %d/%d: %s; retrying in %.1fs", label, attempt, retries, exc, sleep_for)
                time.sleep(sleep_for)
    raise RuntimeError(f"{label} failed after {retries} attempts") from last_error


class AkshareDownloader:
    """Download real A-share data without a Tushare token."""

    def __init__(
        self,
        raw_dir: str,
        start_date: str,
        end_date: str,
        benchmark_code: str = "000905.SH",
        benchmark_codes: dict[str, str] | list[str] | tuple[str, ...] | None = None,
        max_workers: int = 4,
        retries: int = 3,
        retry_sleep: float = 1.0,
    ):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark_code = benchmark_code
        self.benchmark_codes = _normalise_benchmark_codes(benchmark_codes, benchmark_code)
        self.index_code = benchmark_code.split(".")[0]
        self.max_workers = max(1, int(max_workers))
        self.retries = max(1, int(retries))
        self.retry_sleep = float(retry_sleep)
        self._stock_basic: pd.DataFrame | None = None

    def save(self, df: pd.DataFrame, name: str) -> None:
        path = self.raw_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved %s to %s (%d rows)", name, path, len(df))

    def download_all(self) -> None:
        stock_basic = self.download_stock_basic()
        self.download_trade_calendar()
        self.download_index_daily()
        self.download_industry_member(stock_basic)
        self.download_market_history(stock_basic)
        self.download_fina_indicator(stock_basic)

    def download_index_constituents(self) -> pd.DataFrame:
        logger.info("Downloading index constituents for %s ...", self.index_code)

        def fetch_sina() -> pd.DataFrame:
            df = ak.index_stock_cons_sina(symbol=self.index_code)
            return pd.DataFrame({
                "code": df["code"].astype(str).str.zfill(6),
                "name": df["name"].astype(str),
            })

        def fetch_csindex() -> pd.DataFrame:
            df = ak.index_stock_cons_csindex(symbol=self.index_code)
            return pd.DataFrame({
                "code": df.iloc[:, 4].astype(str).str.zfill(6),
                "name": df.iloc[:, 5].astype(str),
            })

        for label, fetch in [("sina constituents", fetch_sina), ("csindex constituents", fetch_csindex)]:
            try:
                out = _retry(fetch, label, self.retries, self.retry_sleep)
                out = out.drop_duplicates("code").sort_values("code").reset_index(drop=True)
                logger.info("Downloaded %d constituents from %s", len(out), label)
                return out
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not download %s: %s", label, exc)

        raise RuntimeError(f"Could not download constituents for {self.index_code}")

    def download_stock_basic(self) -> pd.DataFrame:
        constituents = self.download_index_constituents()

        logger.info("Downloading exchange stock lists ...")
        sh = _retry(lambda: ak.stock_info_sh_name_code(symbol="主板A股"), "SSE stock list", self.retries, self.retry_sleep)
        try:
            kcb = _retry(lambda: ak.stock_info_sh_name_code(symbol="科创板"), "STAR stock list", self.retries, self.retry_sleep)
            sh = pd.concat([sh, kcb], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not download STAR stock list: %s", exc)

        sz = _retry(lambda: ak.stock_info_sz_name_code(symbol="A股列表"), "SZSE stock list", self.retries, self.retry_sleep)

        sh_basic = pd.DataFrame({
            "symbol": sh["证券代码"].astype(str).str.zfill(6),
            "name": sh["证券简称"].astype(str),
            "industry": np.nan,
            "market": "主板",
            "exchange": "SSE",
            "list_date": pd.to_datetime(sh["上市日期"], errors="coerce").dt.strftime("%Y%m%d"),
        })

        sz_basic = pd.DataFrame({
            "symbol": sz["A股代码"].astype(str).str.zfill(6),
            "name": sz["A股简称"].astype(str),
            "industry": sz.get("所属行业", pd.Series(np.nan, index=sz.index)).astype(str),
            "market": sz.get("板块", pd.Series("A股", index=sz.index)).astype(str),
            "exchange": "SZSE",
            "list_date": pd.to_datetime(sz["A股上市日期"], errors="coerce").dt.strftime("%Y%m%d"),
        })

        base = pd.concat([sh_basic, sz_basic], ignore_index=True).drop_duplicates("symbol")
        base = constituents.merge(base, left_on="code", right_on="symbol", how="left", suffixes=("_idx", ""))
        base["symbol"] = base["code"]
        base["name"] = base["name"].fillna(base["name_idx"])
        base["industry"] = base["industry"].replace({"nan": np.nan}).fillna("unknown")
        base["market"] = base["market"].fillna("A股")
        base["exchange"] = base["exchange"].fillna(base["symbol"].map(lambda x: "SSE" if str(x).startswith("6") else "SZSE"))
        base["list_date"] = base["list_date"].fillna("19000101")
        base["ts_code"] = base["symbol"].map(_ts_code_from_code)
        base["area"] = ""
        base["delist_date"] = None
        base["is_hs"] = "N"

        stock_basic = base[
            ["ts_code", "symbol", "name", "area", "industry", "market", "exchange", "list_date", "delist_date", "is_hs"]
        ].sort_values("ts_code").reset_index(drop=True)
        self._stock_basic = stock_basic
        self.save(stock_basic, "stock_basic")
        return stock_basic

    def download_trade_calendar(self) -> pd.DataFrame:
        logger.info("Downloading trade calendar ...")
        dates = _retry(ak.tool_trade_date_hist_sina, "trade calendar", self.retries, self.retry_sleep)
        dates = pd.to_datetime(dates["trade_date"], errors="coerce").dropna().sort_values()
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        open_dates = dates[(dates >= start) & (dates <= end)]

        cal = pd.DataFrame({"cal_date": pd.date_range(start, end, freq="D")})
        cal["cal_key"] = cal["cal_date"].dt.normalize()
        open_set = set(open_dates.dt.normalize())
        cal["exchange"] = "SSE"
        cal["is_open"] = cal["cal_key"].isin(open_set).astype(int)
        cal["cal_date"] = cal["cal_date"].dt.strftime("%Y%m%d")
        cal["pretrade_date"] = pd.NA

        last_open: str | None = None
        pretrade = []
        for _, row in cal.iterrows():
            pretrade.append(last_open if last_open is not None else row["cal_date"])
            if row["is_open"] == 1:
                last_open = row["cal_date"]
        cal["pretrade_date"] = pretrade
        cal = cal[["exchange", "cal_date", "is_open", "pretrade_date"]]
        self.save(cal, "trade_calendar")
        return cal

    def download_index_daily(self) -> pd.DataFrame:
        logger.info("Downloading benchmark index daily data ...")
        records = []

        for name, ts_code in self.benchmark_codes.items():
            code = ts_code.split(".")[0]
            label = f"{name} {ts_code} index daily"
            try:
                raw = _retry(
                    lambda code=code: ak.index_zh_a_hist(
                        symbol=code,
                        period="daily",
                        start_date=self.start_date,
                        end_date=self.end_date,
                    ),
                    label,
                    self.retries,
                    self.retry_sleep,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Eastmoney index download failed for %s: %s; trying Sina", ts_code, exc)
                raw = _retry(
                    lambda ts_code=ts_code: ak.stock_zh_index_daily(symbol=_sina_symbol_from_ts_code(ts_code)),
                    f"{name} {ts_code} Sina index daily",
                    self.retries,
                    self.retry_sleep,
                )

            out = self._normalise_index_daily(raw, ts_code=ts_code, benchmark_name=name)
            if out.empty:
                raise RuntimeError(f"Empty index daily data for {name} {ts_code}")
            records.append(out)

        result = pd.concat(records, ignore_index=True)
        result = result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        self.save(result, "index_daily")
        return result

    def _normalise_index_daily(self, df: pd.DataFrame, ts_code: str, benchmark_name: str) -> pd.DataFrame:
        rename_map = {
            "\u65e5\u671f": "date",
            "\u5f00\u76d8": "open",
            "\u6536\u76d8": "close",
            "\u6700\u9ad8": "high",
            "\u6700\u4f4e": "low",
            "\u6210\u4ea4\u91cf": "vol",
            "\u6210\u4ea4\u989d": "amount",
            "\u6da8\u8dcc\u5e45": "pct_chg",
            "\u6da8\u8dcc\u989d": "change",
            "volume": "vol",
        }

        out = df.rename(columns=rename_map).copy()
        if "date" not in out.columns:
            raise ValueError(f"Index daily data for {ts_code} has no date column")

        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date")

        for col in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]:
            if col not in out.columns:
                out[col] = np.nan
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out["trade_date"] = out["date"].dt.strftime("%Y%m%d")
        in_range = (out["trade_date"] >= self.start_date) & (out["trade_date"] <= self.end_date)
        out = out.loc[in_range].copy()

        if out["pre_close"].isna().all():
            out["pre_close"] = out["close"].shift(1)
        if out["change"].isna().all():
            out["change"] = out["close"] - out["pre_close"]
        if out["pct_chg"].isna().all():
            out["pct_chg"] = out["change"] / out["pre_close"] * 100

        out["ts_code"] = ts_code
        out["benchmark_name"] = benchmark_name
        return out[[
            "ts_code",
            "benchmark_name",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount",
        ]]

    def download_industry_member(self, stock_basic: pd.DataFrame | None = None) -> pd.DataFrame:
        logger.info("Building industry_member from exchange stock lists ...")
        if stock_basic is None:
            stock_basic = self._stock_basic
        if stock_basic is None:
            raise RuntimeError("stock_basic is required before industry_member")

        industry = stock_basic[["ts_code", "industry", "list_date"]].copy()
        industry["industry"] = industry["industry"].fillna("unknown").replace("", "unknown")
        industry["index_name"] = industry["industry"]
        industry["index_code"] = industry["industry"].astype("category").cat.codes.map(lambda x: f"AK.IND.{x:03d}")
        industry["con_code"] = industry["ts_code"]
        industry["in_date"] = industry["list_date"].fillna("19000101")
        industry["out_date"] = None
        out = industry[["index_code", "index_name", "con_code", "in_date", "out_date"]]
        self.save(out, "industry_member")
        return out

    def download_market_history(self, stock_basic: pd.DataFrame) -> None:
        logger.info("Downloading daily prices and valuation data for %d stocks ...", len(stock_basic))

        records = []
        failures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._download_one_stock_history, row.ts_code, row.name): row.ts_code
                for row in stock_basic.itertuples(index=False)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading history"):
                ts_code = futures[future]
                try:
                    records.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    failures.append({"ts_code": ts_code, "error": str(exc)})
                    logger.warning("History download failed for %s: %s", ts_code, exc)

        if not records:
            raise RuntimeError("No history data downloaded")

        daily = pd.concat([item["daily"] for item in records], ignore_index=True)
        adj = pd.concat([item["adj_factor"] for item in records], ignore_index=True)
        daily_basic = pd.concat([item["daily_basic"] for item in records], ignore_index=True)
        stk_limit = pd.concat([item["stk_limit"] for item in records], ignore_index=True)

        self.save(daily.sort_values(["trade_date", "ts_code"]), "daily")
        self.save(adj.sort_values(["trade_date", "ts_code"]), "adj_factor")
        self.save(daily_basic.sort_values(["trade_date", "ts_code"]), "daily_basic")
        self.save(stk_limit.sort_values(["trade_date", "ts_code"]), "stk_limit")

        if failures:
            failure_df = pd.DataFrame(failures)
            failure_df.to_csv(self.raw_dir / "download_failures.csv", index=False, encoding="utf-8-sig")
            logger.warning("History failed for %d stocks; details saved to download_failures.csv", len(failures))

    def _download_one_stock_history(self, ts_code: str, name: str) -> dict[str, pd.DataFrame]:
        symbol = _sina_symbol_from_ts_code(ts_code)
        buffered_start = (pd.to_datetime(self.start_date) - timedelta(days=40)).strftime("%Y%m%d")

        is_cdr = ts_code.split(".")[0].startswith("689")
        try:
            price = _retry(
                lambda: ak.stock_zh_a_daily(symbol=symbol, start_date=buffered_start, end_date=self.end_date, adjust=""),
                f"{ts_code} daily price",
                self.retries,
                self.retry_sleep,
            )
        except Exception:
            if not is_cdr:
                raise
            price = _retry(
                lambda: ak.stock_zh_a_cdr_daily(symbol=symbol, start_date=buffered_start, end_date=self.end_date),
                f"{ts_code} CDR daily price",
                self.retries,
                self.retry_sleep,
            )
        if price.empty:
            raise RuntimeError("empty daily price")

        if is_cdr:
            factor = pd.DataFrame()
        else:
            factor = _retry(
                lambda: ak.stock_zh_a_daily(symbol=symbol, adjust="hfq-factor"),
                f"{ts_code} hfq factor",
                self.retries,
                self.retry_sleep,
            )

        price = price.copy()
        price["date"] = pd.to_datetime(price["date"], errors="coerce")
        price = price.dropna(subset=["date"]).sort_values("date")
        for col in ["outstanding_share", "turnover"]:
            if col not in price.columns:
                price[col] = np.nan
        for col in ["open", "high", "low", "close", "volume", "amount", "outstanding_share", "turnover"]:
            price[col] = pd.to_numeric(price[col], errors="coerce")

        adj = self._build_adj_factor(ts_code, price[["date"]], factor)
        price = price.merge(adj[["date", "adj_factor"]], on="date", how="left")
        price["adj_factor"] = price["adj_factor"].fillna(1.0)

        price["trade_date"] = price["date"].dt.strftime("%Y%m%d")
        price["ts_code"] = ts_code
        price["pre_close"] = price["close"].shift(1)
        price["change"] = price["close"] - price["pre_close"]
        price["pct_chg"] = price["change"] / price["pre_close"] * 100

        in_range = (price["trade_date"] >= self.start_date) & (price["trade_date"] <= self.end_date)
        price = price.loc[in_range].copy()

        daily = price[[
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "volume",
            "amount",
        ]].rename(columns={"volume": "vol"})

        adj_factor = price[["ts_code", "trade_date", "adj_factor"]].copy()
        daily_basic = self._build_daily_basic(ts_code, price)
        stk_limit = self._build_stk_limit(ts_code, name, price)

        return {
            "daily": daily,
            "adj_factor": adj_factor,
            "daily_basic": daily_basic,
            "stk_limit": stk_limit,
        }

    def _build_adj_factor(self, ts_code: str, price_dates: pd.DataFrame, factor: pd.DataFrame) -> pd.DataFrame:
        if factor.empty or "hfq_factor" not in factor.columns:
            out = price_dates.copy()
            out["ts_code"] = ts_code
            out["adj_factor"] = 1.0
            return out

        factor = factor.copy()
        factor["date"] = pd.to_datetime(factor["date"], errors="coerce")
        factor["hfq_factor"] = pd.to_numeric(factor["hfq_factor"], errors="coerce")
        factor = factor.dropna(subset=["date", "hfq_factor"]).sort_values("date")

        out = pd.merge_asof(
            price_dates.sort_values("date"),
            factor[["date", "hfq_factor"]],
            on="date",
            direction="backward",
        )
        out["ts_code"] = ts_code
        out["adj_factor"] = out["hfq_factor"].fillna(1.0).astype(float)
        return out[["date", "ts_code", "adj_factor"]]

    def _build_daily_basic(self, ts_code: str, price: pd.DataFrame) -> pd.DataFrame:
        valuation = self._download_valuation(ts_code)
        base = price[["date", "ts_code", "trade_date", "close", "outstanding_share", "turnover"]].copy()
        if not valuation.empty:
            base = pd.merge_asof(
                base.sort_values("date"),
                valuation.sort_values("date"),
                on="date",
                direction="backward",
            )
        else:
            base["pe_ttm"] = np.nan
            base["pb"] = np.nan
            base["total_mv"] = np.nan

        base["turnover_rate"] = pd.to_numeric(base["turnover"], errors="coerce") * 100
        base["circ_mv"] = base["close"] * base["outstanding_share"] / 10000
        base["total_mv"] = pd.to_numeric(base.get("total_mv"), errors="coerce") * 10000
        base["pe_ttm"] = pd.to_numeric(base.get("pe_ttm"), errors="coerce")
        base["pb"] = pd.to_numeric(base.get("pb"), errors="coerce")
        base["pe"] = base["pe_ttm"]
        base["volume_ratio"] = np.nan
        base["ps"] = np.nan
        base["ps_ttm"] = np.nan
        base["dv_ratio"] = np.nan
        base["dv_ttm"] = np.nan
        base["total_share"] = np.nan
        base["float_share"] = base["outstanding_share"]
        base["free_share"] = np.nan

        return base[[
            "ts_code",
            "trade_date",
            "turnover_rate",
            "volume_ratio",
            "pe",
            "pe_ttm",
            "pb",
            "ps",
            "ps_ttm",
            "dv_ratio",
            "dv_ttm",
            "total_share",
            "float_share",
            "free_share",
            "total_mv",
            "circ_mv",
        ]]

    def _download_valuation(self, ts_code: str) -> pd.DataFrame:
        code = ts_code.split(".")[0]
        indicators = {
            "pe_ttm": "市盈率(TTM)",
            "pb": "市净率",
            "total_mv": "总市值",
        }
        frames = []
        for out_col, indicator in indicators.items():
            try:
                df = _retry(
                    lambda indicator=indicator: ak.stock_zh_valuation_baidu(
                        symbol=code,
                        indicator=indicator,
                        period="\u8fd1\u5341\u5e74",
                    ),
                    f"{ts_code} {indicator}",
                    self.retries,
                    self.retry_sleep,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Valuation download failed for %s %s: %s", ts_code, indicator, exc)
                continue
            if df.empty:
                continue
            df = df.rename(columns={"value": out_col})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df[out_col] = pd.to_numeric(df[out_col], errors="coerce")
            frames.append(df[["date", out_col]].dropna(subset=["date"]))

        if not frames:
            return pd.DataFrame()

        out = frames[0]
        for frame in frames[1:]:
            out = out.merge(frame, on="date", how="outer")
        return out.sort_values("date")

    def _build_stk_limit(self, ts_code: str, name: str, price: pd.DataFrame) -> pd.DataFrame:
        out = price[["ts_code", "trade_date", "pre_close"]].copy()
        limit_pct = self._limit_pct(ts_code, name)
        out["up_limit"] = (out["pre_close"] * (1 + limit_pct)).round(2)
        out["down_limit"] = (out["pre_close"] * (1 - limit_pct)).round(2)
        return out[["ts_code", "trade_date", "up_limit", "down_limit"]]

    @staticmethod
    def _limit_pct(ts_code: str, name: str) -> float:
        code = ts_code.split(".")[0]
        if "ST" in str(name).upper():
            return 0.05
        if code.startswith(("300", "301", "688", "689")):
            return 0.20
        return 0.10

    def download_fina_indicator(self, stock_basic: pd.DataFrame) -> pd.DataFrame:
        logger.info("Downloading financial indicators for %d stocks ...", len(stock_basic))
        records = []
        failures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._download_one_fina_indicator, row.ts_code): row.ts_code
                for row in stock_basic.itertuples(index=False)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading financials"):
                ts_code = futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        records.append(df)
                except Exception as exc:  # noqa: BLE001
                    failures.append({"ts_code": ts_code, "error": str(exc)})
                    logger.warning("Financial download failed for %s: %s", ts_code, exc)

        if records:
            out = pd.concat(records, ignore_index=True)
            out = out[(out["end_date"] >= self.start_date) & (out["end_date"] <= self.end_date)]
            out = out.sort_values(["ts_code", "end_date", "ann_date"]).reset_index(drop=True)
        else:
            out = pd.DataFrame(columns=[
                "ts_code",
                "ann_date",
                "end_date",
                "roe",
                "roe_dt",
                "roa",
                "grossprofit_margin",
                "netprofit_margin",
                "debt_to_assets",
                "ocf_to_or",
                "or_yoy",
                "netprofit_yoy",
            ])

        self.save(out, "fina_indicator")
        if failures:
            failure_df = pd.DataFrame(failures)
            failure_df.to_csv(self.raw_dir / "financial_failures.csv", index=False, encoding="utf-8-sig")
            logger.warning("Financials failed for %d stocks; details saved to financial_failures.csv", len(failures))
        return out

    def _download_one_fina_indicator(self, ts_code: str) -> pd.DataFrame:
        df = _retry(
            lambda: ak.stock_financial_analysis_indicator_em(symbol=ts_code, indicator="按报告期"),
            f"{ts_code} financial indicators",
            self.retries,
            self.retry_sleep,
        )
        if df.empty:
            return pd.DataFrame()

        out = pd.DataFrame({
            "ts_code": ts_code,
            "ann_date": pd.to_datetime(df.get("NOTICE_DATE"), errors="coerce").dt.strftime("%Y%m%d"),
            "end_date": pd.to_datetime(df.get("REPORT_DATE"), errors="coerce").dt.strftime("%Y%m%d"),
            "roe": pd.to_numeric(df.get("ROEJQ"), errors="coerce"),
            "roe_dt": pd.to_numeric(df.get("ROEJQ"), errors="coerce"),
            "roa": pd.to_numeric(df.get("ZZCJLL"), errors="coerce"),
            "grossprofit_margin": pd.to_numeric(df.get("XSMLL"), errors="coerce"),
            "netprofit_margin": pd.to_numeric(df.get("XSJLL"), errors="coerce"),
            "debt_to_assets": pd.to_numeric(df.get("ZCFZL"), errors="coerce"),
            "ocf_to_or": pd.to_numeric(df.get("JYXJLYYSR"), errors="coerce"),
            "or_yoy": pd.to_numeric(df.get("TOTALOPERATEREVETZ"), errors="coerce"),
            "netprofit_yoy": pd.to_numeric(df.get("PARENTNETPROFITTZ"), errors="coerce"),
        })
        out["ann_date"] = out["ann_date"].fillna(out["end_date"])
        out = out.dropna(subset=["end_date"])
        return out
