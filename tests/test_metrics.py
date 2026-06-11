import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_prepare_nav():
    import pandas as pd
    from ashare_alpha.analysis.metrics import prepare_nav

    nav = pd.DataFrame({
        "trade_date": ["20200102", "20200103", "20200104"],
        "total_value": [1000000, 1010000, 1020000],
        "cash": [500000, 490000, 480000],
    })

    result = prepare_nav(nav)

    assert "nav" in result.columns
    assert "daily_ret" in result.columns
    assert abs(result["nav"].iloc[0] - 1.0) < 1e-10


def test_calc_performance():
    import pandas as pd
    import numpy as np
    from ashare_alpha.analysis.metrics import calc_performance, prepare_nav

    n_days = 252
    returns = np.random.normal(0.001, 0.02, n_days)
    nav_values = 1000000 * np.cumprod(1 + returns)
    nav_values = np.insert(nav_values, 0, 1000000)

    dates = pd.date_range("2020-01-01", periods=n_days + 1, freq="B")
    dates = dates.strftime("%Y%m%d").tolist()

    nav = pd.DataFrame({
        "trade_date": dates,
        "total_value": nav_values,
        "cash": [500000] * (n_days + 1),
    })

    metrics = calc_performance(nav)

    assert "annual_return" in metrics
    assert "annual_vol" in metrics
    assert "sharpe" in metrics
    assert "max_drawdown" in metrics
    assert "calmar" in metrics


def test_benchmark_performance():
    import pandas as pd
    from ashare_alpha.analysis.metrics import calc_benchmark_performance, calc_performance

    nav = pd.DataFrame({
        "trade_date": ["20200102", "20200103", "20200106", "20200107"],
        "total_value": [100.0, 105.0, 103.0, 110.0],
        "cash": [0, 0, 0, 0],
    })
    benchmark = pd.DataFrame({
        "trade_date": ["20200102", "20200103", "20200106", "20200107"],
        "close": [100.0, 102.0, 104.0, 105.0],
    })

    result = calc_benchmark_performance(nav, benchmark)

    assert result["n_days"] == 4
    assert result["excess_return"] > 0
    assert result["max_excess_drawdown"] < 0
    assert "information_ratio" in result

    metrics = calc_performance(nav, benchmarks={"csi500": benchmark})
    assert "benchmarks" in metrics
    assert "csi500" in metrics["benchmarks"]


def test_max_drawdown():
    import pandas as pd
    from ashare_alpha.analysis.metrics import calc_max_drawdown, prepare_nav

    nav = pd.DataFrame({
        "trade_date": ["D1", "D2", "D3", "D4"],
        "total_value": [100, 80, 90, 60],
        "cash": [0, 0, 0, 0],
    })

    nav = prepare_nav(nav)
    mdd = calc_max_drawdown(nav)

    assert mdd < 0
    assert abs(mdd - (-0.4)) < 1e-10
