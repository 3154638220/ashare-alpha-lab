STOCK_BASIC_COLUMNS = [
    "ts_code",
    "symbol",
    "name",
    "area",
    "industry",
    "market",
    "exchange",
    "list_date",
    "delist_date",
    "is_hs",
]

TRADE_CALENDAR_COLUMNS = [
    "exchange",
    "cal_date",
    "is_open",
    "pretrade_date",
]

DAILY_COLUMNS = [
    "ts_code",
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
]

ADJ_FACTOR_COLUMNS = [
    "ts_code",
    "trade_date",
    "adj_factor",
]

DAILY_BASIC_COLUMNS = [
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
]

FINA_INDICATOR_COLUMNS = [
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
]

STK_LIMIT_COLUMNS = [
    "ts_code",
    "trade_date",
    "up_limit",
    "down_limit",
]

INDEX_DAILY_COLUMNS = [
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
]

PRICE_PANEL_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "close",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "ret",
    "amount",
    "vol",
    "pe_ttm",
    "pb",
    "total_mv",
    "circ_mv",
    "up_limit",
    "down_limit",
    "turnover_rate",
]

FACTOR_INPUT_COLUMNS = [
    "ts_code",
    "trade_date",
    "adj_open",
    "adj_close",
    "amount",
    "turnover_rate",
    "pe_ttm",
    "pb",
    "total_mv",
    "circ_mv",
    "roe_dt",
    "roa",
    "ocf_to_or",
    "or_yoy",
    "netprofit_yoy",
    "debt_to_assets",
    "industry_code",
    "industry_name",
]
