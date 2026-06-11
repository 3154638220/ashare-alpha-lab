import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_winsorize():
    import pandas as pd
    from ashare_alpha.factors.transform import winsorize_series

    s = pd.Series([-10, 1, 2, 3, 4, 5, 100])
    result = winsorize_series(s, lower=0.1, upper=0.9)

    assert result.min() >= s.quantile(0.1)
    assert result.max() <= s.quantile(0.9)


def test_zscore():
    import pandas as pd
    from ashare_alpha.factors.transform import zscore_series

    s = pd.Series([1, 2, 3, 4, 5])
    result = zscore_series(s)

    assert abs(result.mean()) < 1e-10
    assert abs(result.std() - 1.0) < 1e-10


def test_zscore_constant():
    import pandas as pd
    from ashare_alpha.factors.transform import zscore_series

    s = pd.Series([5, 5, 5, 5, 5])
    result = zscore_series(s)

    assert (result == 0).all()


def test_value_factor():
    import pandas as pd
    from ashare_alpha.factors.value import calc_value_factor

    df = pd.DataFrame({
        "ts_code": ["A", "B", "A", "B"],
        "trade_date": ["20200101", "20200101", "20200102", "20200102"],
        "industry_code": ["I1", "I1", "I1", "I1"],
        "pb": [1.0, 2.0, 1.5, 2.5],
        "pe_ttm": [10, 20, 12, 25],
    })

    result = calc_value_factor(df)

    assert "value" in result.columns
    assert "ts_code" in result.columns
    assert "trade_date" in result.columns
    assert len(result) > 0


def test_composite_score():
    import pandas as pd
    from ashare_alpha.factors.composite import calc_composite_score

    df = pd.DataFrame({
        "ts_code": ["A", "B"],
        "trade_date": ["20200101", "20200101"],
        "value": [1.0, -1.0],
        "quality": [0.5, -0.5],
        "growth": [0.0, 0.0],
        "lowvol": [0.2, -0.2],
        "momentum": [0.1, -0.1],
        "reversal": [-0.1, 0.1],
    })

    weights = {"value": 0.25, "quality": 0.20, "growth": 0.15, "lowvol": 0.15, "momentum": 0.10, "reversal": 0.10}

    result = calc_composite_score(df, weights)

    assert "score" in result.columns
    assert len(result) == 2
