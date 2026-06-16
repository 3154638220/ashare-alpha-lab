import pandas as pd

from ashare_alpha.logger import logger


def _clean_date(value) -> pd.Series | str:
    if isinstance(value, pd.Series):
        text = value.astype("string").fillna("").str.strip()
        parsed = pd.to_datetime(text.mask(_is_missing_date(text)), errors="coerce")
        return parsed.dt.strftime("%Y%m%d").fillna("")
    if pd.isna(value):
        return ""
    parsed = pd.to_datetime(str(value).strip(), errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y%m%d")


def _is_missing_date(value: pd.Series) -> pd.Series:
    text = value.astype("string").fillna("").str.strip()
    return text.isin(["", "None", "NaT", "nan", "<NA>"])


def _suspension_mask(df: pd.DataFrame) -> pd.Series:
    return df["open"].isna() | df["close"].isna() | (df["vol"] == 0)


def _truthy_status(value: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(value, errors="coerce")
    text = value.astype("string").fillna("").str.strip().str.lower()
    return numeric.eq(1) | text.isin(["true", "t", "yes", "y", "st", "*st", "1", "是"])


def filter_listed_days(
    df: pd.DataFrame,
    stock_basic: pd.DataFrame,
    trade_date: str,
    min_list_days: int,
) -> pd.DataFrame:
    stock_basic = stock_basic.copy()
    stock_basic["list_date"] = stock_basic["list_date"].astype(str)

    listed = stock_basic[stock_basic["list_date"] <= trade_date].copy()

    listed["listed_days"] = (
        pd.to_datetime(trade_date) - pd.to_datetime(listed["list_date"])
    ).dt.days

    valid_codes = listed.loc[listed["listed_days"] >= min_list_days, "ts_code"]

    return df[df["ts_code"].isin(valid_codes)]


def filter_delisted(
    df: pd.DataFrame,
    stock_basic: pd.DataFrame,
    trade_date: str,
) -> pd.DataFrame:
    if "delist_date" not in stock_basic.columns:
        return df

    status = stock_basic[["ts_code", "delist_date"]].copy()
    status["delist_date"] = _clean_date(status["delist_date"])
    status = status.drop_duplicates("ts_code")
    merged = df.merge(status, on="ts_code", how="left")
    missing = _is_missing_date(merged["delist_date"])
    valid = missing | (merged["delist_date"] > trade_date)
    return merged.loc[valid].drop(columns=["delist_date"])


def filter_bj(df: pd.DataFrame, stock_basic: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [c for c in ["ts_code", "exchange", "market"] if c in stock_basic.columns]
    if len(meta_cols) == 1:
        return df[~df["ts_code"].astype(str).str.endswith(".BJ")]

    meta = stock_basic[meta_cols].drop_duplicates("ts_code").copy()
    meta = meta.rename(columns={c: f"{c}_basic" for c in ["exchange", "market"] if c in meta.columns})
    merged = df.merge(meta, on="ts_code", how="left")

    code = merged["ts_code"].astype(str)
    bj_mask = code.str.endswith(".BJ")

    if "exchange_basic" in merged.columns:
        exchange = merged["exchange_basic"].astype("string").fillna("").str.upper()
        bj_mask = bj_mask | exchange.isin(["BSE", "BJSE", "BEIJING"])

    if "market_basic" in merged.columns:
        market = merged["market_basic"].astype("string").fillna("")
        bj_mask = bj_mask | market.str.contains("北交所|北京证券交易所", na=False)

    return merged.loc[~bj_mask].drop(columns=[c for c in ["exchange_basic", "market_basic"] if c in merged.columns])


def filter_st(
    df: pd.DataFrame,
    stock_basic: pd.DataFrame,
    trade_date: str | None = None,
    stock_status: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if stock_status is not None and not stock_status.empty and {"ts_code", "trade_date", "is_st"}.issubset(stock_status.columns):
        status = stock_status.copy()
        status["trade_date"] = status["trade_date"].astype(str)
        if trade_date is not None:
            status = status[status["trade_date"] <= trade_date]
        status = status.sort_values(["ts_code", "trade_date"]).groupby("ts_code").tail(1)
        status["is_st"] = _truthy_status(status["is_st"])
        merged = df.merge(status[["ts_code", "is_st"]], on="ts_code", how="left")
        valid = ~merged["is_st"].eq(True)
        return merged.loc[valid].drop(columns=["is_st"])

    if stock_status is not None and not stock_status.empty:
        logger.warning("stock_status is missing required columns; falling back to current-name ST filter")

    names = stock_basic[["ts_code", "name"]].copy()
    df = df.merge(names, on="ts_code", how="left")

    mask = ~df["name"].str.contains("ST|退", na=False)
    return df[mask].drop(columns=["name"])


def filter_liquidity(
    df: pd.DataFrame,
    price_panel: pd.DataFrame,
    trade_date: str,
    min_avg_amount: float,
    window: int = 20,
) -> pd.DataFrame:
    hist = price_panel[price_panel["trade_date"] <= trade_date].copy()
    hist = hist.sort_values(["ts_code", "trade_date"])

    hist["avg_amount_20"] = (
        hist.groupby("ts_code")["amount"]
        .rolling(window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    latest = hist.groupby("ts_code").tail(1)
    valid_codes = latest.loc[
        latest["avg_amount_20"] >= min_avg_amount, "ts_code"
    ]

    return df[df["ts_code"].isin(valid_codes)]


def filter_suspend_days(
    df: pd.DataFrame,
    price_panel: pd.DataFrame,
    trade_date: str,
    max_suspend_days: int,
    window: int = 60,
) -> pd.DataFrame:
    hist = price_panel[price_panel["trade_date"] <= trade_date].copy()
    if hist.empty:
        return df.iloc[0:0].copy()

    hist = hist.sort_values(["ts_code", "trade_date"])
    hist = hist.groupby("ts_code").tail(window).copy()
    hist["is_suspended"] = _suspension_mask(hist)

    suspend_days = hist.groupby("ts_code")["is_suspended"].sum()
    valid_codes = suspend_days.loc[suspend_days <= max_suspend_days].index

    return df[df["ts_code"].isin(valid_codes)]


def filter_valuation_valid(
    df: pd.DataFrame,
    require_positive_pe: bool = True,
    require_positive_pb: bool = True,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    if require_positive_pb and "pb" in df.columns:
        mask &= df["pb"] > 0

    if require_positive_pe and "pe_ttm" in df.columns:
        mask &= df["pe_ttm"] > 0

    return df[mask]


def build_universe(
    trade_date: str,
    factor_input: pd.DataFrame,
    price_panel: pd.DataFrame,
    stock_basic: pd.DataFrame,
    config: dict,
    stock_status: pd.DataFrame | None = None,
    return_filter_stats: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    universe_cfg = config["strategy"]["universe"]

    df = factor_input[factor_input["trade_date"] == trade_date].copy()
    stats = []

    def apply_filter(name: str, frame: pd.DataFrame, func) -> pd.DataFrame:
        before = len(frame)
        out = func(frame)
        after = len(out)
        stats.append({
            "trade_date": trade_date,
            "filter": name,
            "before": before,
            "after": after,
            "removed": before - after,
        })
        logger.info("After %s filter: %d -> %d", name, before, after)
        return out

    stats.append({
        "trade_date": trade_date,
        "filter": "initial",
        "before": len(df),
        "after": len(df),
        "removed": 0,
    })

    if universe_cfg.get("exclude_new_stock", True):
        df = apply_filter(
            "listed_days",
            df,
            lambda frame: filter_listed_days(
                frame, stock_basic, trade_date, universe_cfg["min_list_days"]
            ),
        )

    df = apply_filter(
        "delisted",
        df,
        lambda frame: filter_delisted(frame, stock_basic, trade_date),
    )

    if universe_cfg.get("exclude_bj", False):
        df = apply_filter(
            "bj",
            df,
            lambda frame: filter_bj(frame, stock_basic),
        )

    if universe_cfg.get("exclude_st", True):
        df = apply_filter(
            "st",
            df,
            lambda frame: filter_st(frame, stock_basic, trade_date, stock_status),
        )

    if universe_cfg.get("exclude_negative_pe", True) or universe_cfg.get("exclude_negative_pb", True):
        df = apply_filter(
            "valuation",
            df,
            lambda frame: filter_valuation_valid(
                frame,
                require_positive_pe=universe_cfg.get("exclude_negative_pe", True),
                require_positive_pb=universe_cfg.get("exclude_negative_pb", True),
            ),
        )

    if "min_avg_amount_20" in universe_cfg:
        df = apply_filter(
            "liquidity",
            df,
            lambda frame: filter_liquidity(
                frame,
                price_panel,
                trade_date,
                universe_cfg["min_avg_amount_20"],
            ),
        )

    if universe_cfg.get("max_suspend_days_60") is not None:
        df = apply_filter(
            "suspend_days_60",
            df,
            lambda frame: filter_suspend_days(
                frame,
                price_panel,
                trade_date,
                universe_cfg["max_suspend_days_60"],
            ),
        )

    if return_filter_stats:
        return df, pd.DataFrame(stats)

    return df
