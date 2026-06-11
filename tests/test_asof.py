import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_no_financial_future_leakage():
    import pandas as pd

    path = Path("data/processed/fundamental_asof.parquet")
    if not path.exists():
        return

    df = pd.read_parquet(path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    df["ann_date"] = df["ann_date"].astype(str)
    df["trade_date"] = df["trade_date"].astype(str)

    violations = df[df["ann_date"] > df["trade_date"]]
    assert len(violations) == 0, f"Found {len(violations)} rows with ann_date > trade_date"


def test_asof_basic_functionality():
    import pandas as pd
    from ashare_alpha.data.asof import build_fundamental_asof

    fina = pd.DataFrame({
        "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ"],
        "ann_date": ["20200110", "20200420", "20200115"],
        "end_date": ["20191231", "20200331", "20191231"],
        "roe_dt": [10.0, 3.0, 8.0],
        "roa": [2.0, 0.5, 1.5],
        "ocf_to_or": [0.5, 0.2, 0.4],
        "or_yoy": [0.1, 0.0, 0.15],
        "netprofit_yoy": [0.2, 0.05, 0.1],
        "debt_to_assets": [0.4, 0.45, 0.5],
        "roe": [10.0, 3.0, 8.0],
        "grossprofit_margin": [0.3, 0.28, 0.35],
        "netprofit_margin": [0.1, 0.05, 0.12],
    })

    trade_dates = ["20200120", "20200501"]

    result = build_fundamental_asof(trade_dates, fina)

    assert len(result) > 0
    assert set(result["trade_date"].unique()) == set(trade_dates)

    for _, row in result.iterrows():
        assert row["ann_date"] <= row["trade_date"]

    rows_0101 = result[result["trade_date"] == "20200120"]
    assert len(rows_0101) == 2
