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
