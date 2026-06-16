import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_filter_st():
    import pandas as pd
    from ashare_alpha.strategy.universe import filter_st

    df = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ", "600001.SH", "600002.SH"],
        "trade_date": ["20200101"] * 4,
    })

    stock_basic = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ", "600001.SH", "600002.SH"],
        "name": ["平安银行", "ST万科", "*ST华夏", "退市茅台"],
    })

    result = filter_st(df, stock_basic)

    assert len(result) == 1
    assert result.iloc[0]["ts_code"] == "000001.SZ"


def test_filter_st_uses_historical_status_when_available():
    import pandas as pd
    from ashare_alpha.strategy.universe import filter_st

    df = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
        "trade_date": ["20200110"] * 3,
    })

    stock_basic = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
        "name": ["ST当前名", "正常名", "正常名"],
    })

    stock_status = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ", "000002.SZ"],
        "trade_date": ["20200101", "20200101", "20200201"],
        "is_st": ["False", 1, 0],
    })

    result = filter_st(df, stock_basic, trade_date="20200110", stock_status=stock_status)

    assert result["ts_code"].tolist() == ["000001.SZ", "000003.SZ"]


def test_filter_delisted():
    import pandas as pd
    from ashare_alpha.strategy.universe import filter_delisted

    df = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
        "trade_date": ["20200110"] * 3,
    })

    stock_basic = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
        "delist_date": [None, "20200110", "20200111"],
    })

    result = filter_delisted(df, stock_basic, trade_date="20200110")

    assert result["ts_code"].tolist() == ["000001.SZ", "000003.SZ"]


def test_filter_bj():
    import pandas as pd
    from ashare_alpha.strategy.universe import filter_bj

    df = pd.DataFrame({
        "ts_code": ["000001.SZ", "430001.BJ", "600001.SH", "830001.SZ"],
        "trade_date": ["20200101"] * 4,
    })

    stock_basic = pd.DataFrame({
        "ts_code": ["000001.SZ", "430001.BJ", "600001.SH", "830001.SZ"],
        "exchange": ["SZSE", "BSE", "SSE", "SZSE"],
        "market": ["主板", "北交所", "主板", "北京证券交易所"],
    })

    result = filter_bj(df, stock_basic)

    assert result["ts_code"].tolist() == ["000001.SZ", "600001.SH"]


def test_filter_suspend_days():
    import pandas as pd
    from ashare_alpha.strategy.universe import filter_suspend_days

    trade_dates = [f"202001{i:02d}" for i in range(1, 13)]
    records = []
    for ts_code in ["000001.SZ", "000002.SZ"]:
        for idx, trade_date in enumerate(trade_dates):
            is_suspended = ts_code == "000002.SZ" and idx < 11
            records.append({
                "ts_code": ts_code,
                "trade_date": trade_date,
                "open": None if is_suspended else 10.0,
                "close": 10.0,
                "vol": 0 if is_suspended else 1000,
            })

    df = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ"],
        "trade_date": ["20200112", "20200112"],
    })
    price_panel = pd.DataFrame(records)

    result = filter_suspend_days(
        df,
        price_panel,
        trade_date="20200112",
        max_suspend_days=10,
    )

    assert result["ts_code"].tolist() == ["000001.SZ"]


def test_filter_valuation():
    import pandas as pd
    from ashare_alpha.strategy.universe import filter_valuation_valid

    df = pd.DataFrame({
        "ts_code": ["A", "B", "C"],
        "trade_date": ["20200101"] * 3,
        "pb": [1.0, -1.0, 0.0],
        "pe_ttm": [10, 20, -5],
    })

    result = filter_valuation_valid(df)

    assert len(result) == 1
    assert result.iloc[0]["ts_code"] == "A"


def test_build_universe_returns_filter_stats():
    import pandas as pd
    from ashare_alpha.strategy.universe import build_universe

    factor_input = pd.DataFrame({
        "ts_code": ["000001.SZ", "430001.BJ", "000002.SZ"],
        "trade_date": ["20200131"] * 3,
        "open": [10.0, 10.0, 10.0],
        "close": [10.0, 10.0, 10.0],
        "vol": [1000, 1000, 1000],
        "amount": [100.0] * 3,
        "pb": [1.0, 1.0, 1.0],
        "pe_ttm": [10.0, 10.0, 10.0],
    })
    stock_basic = pd.DataFrame({
        "ts_code": ["000001.SZ", "430001.BJ", "000002.SZ"],
        "name": ["平安银行", "北证测试", "退市测试"],
        "exchange": ["SZSE", "BSE", "SZSE"],
        "market": ["主板", "北交所", "主板"],
        "list_date": ["20180101", "20180101", "20180101"],
        "delist_date": [None, None, "20200101"],
    })
    config = {
        "strategy": {
            "universe": {
                "exclude_new_stock": True,
                "min_list_days": 1,
                "exclude_bj": True,
                "exclude_st": True,
                "exclude_negative_pe": True,
                "exclude_negative_pb": True,
                "max_suspend_days_60": 10,
            }
        }
    }

    universe, stats = build_universe(
        "20200131",
        factor_input,
        factor_input,
        stock_basic,
        config,
        return_filter_stats=True,
    )

    assert universe["ts_code"].tolist() == ["000001.SZ"]
    assert {"trade_date", "filter", "before", "after", "removed"}.issubset(stats.columns)
    assert "bj" in set(stats["filter"])
    assert "suspend_days_60" in set(stats["filter"])
