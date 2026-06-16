import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _make_factor_fixture():
    import pandas as pd

    dates = ["20200102", "20200103", "20210104", "20210105"]
    stocks = [f"S{i:03d}" for i in range(30)]

    factor_records = []
    price_records = []
    for date_idx, trade_date in enumerate(dates):
        for stock_idx, ts_code in enumerate(stocks):
            value = float(stock_idx)
            factor_records.append({
                "ts_code": ts_code,
                "trade_date": trade_date,
                "value": None if stock_idx == 0 and date_idx == 0 else value,
                "momentum": -value,
            })
            price_records.append({
                "ts_code": ts_code,
                "trade_date": trade_date,
                "adj_open": 100.0 * (1.0 + stock_idx / 1000.0) ** date_idx,
                "adj_close": 100.0 * (1.0 + stock_idx / 1000.0) ** date_idx,
            })

    return pd.DataFrame(factor_records), pd.DataFrame(price_records)


def test_factor_group_returns_and_decay():
    from ashare_alpha.analysis.factor_decay import calc_factor_decay
    from ashare_alpha.analysis.factor_group import (
        calc_factor_group_returns,
        summarize_factor_group_returns,
    )

    factors, price = _make_factor_fixture()

    group_returns = calc_factor_group_returns(
        factors,
        price,
        ["value"],
        horizon=1,
        n_groups=3,
        min_obs=3,
    )
    group_summary = summarize_factor_group_returns(group_returns, horizon=1)
    long_short = group_summary[group_summary["group"].astype(str) == "long_short"].iloc[0]

    assert long_short["mean_forward_return"] > 0
    assert long_short["monotonicity"] > 0

    decay = calc_factor_decay(factors, price, ["value", "momentum"], horizons=(1,))
    value_ic = decay[decay["factor"] == "value"].iloc[0]
    momentum_ic = decay[decay["factor"] == "momentum"].iloc[0]

    assert value_ic["mean_ic"] > 0.99
    assert momentum_ic["mean_ic"] < -0.99


def test_generate_factor_diagnostics_report(tmp_path):
    import pandas as pd
    from ashare_alpha.analysis.factor_report import generate_factor_diagnostics_report

    factors, price = _make_factor_fixture()

    outputs = generate_factor_diagnostics_report(
        factors,
        price,
        output_dir=tmp_path,
        factor_names=["value", "momentum"],
        base_horizon=1,
        decay_horizons=(1, 2),
        n_groups=3,
    )

    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "ic_by_year.csv").exists()
    assert (tmp_path / "group_returns.csv").exists()
    assert (tmp_path / "payoff_by_date.csv").exists()
    assert (tmp_path / "payoff_summary.csv").exists()
    assert (tmp_path / "execution_payoff_by_date.csv").exists()
    assert (tmp_path / "execution_payoff_summary.csv").exists()
    assert (tmp_path / "decay.csv").exists()
    assert (tmp_path / "coverage.csv").exists()
    assert (tmp_path / "value.md").exists()

    summary = pd.read_csv(tmp_path / "summary.csv")
    momentum = summary[summary["factor"] == "momentum"].iloc[0]
    value = summary[summary["factor"] == "value"].iloc[0]

    assert momentum["recommendation"] == "negative_ic_review_reverse_or_remove"
    assert value["long_short_mean_return"] > 0
    assert value["payoff_mean_pearson_ic"] > 0
    assert value["long_short_pos_ratio"] > 0
    assert outputs["coverage"]["coverage"].min() < 1.0
    assert not outputs["payoff_by_date"].empty
    assert not outputs["execution_payoff_summary"].empty


def test_factor_diagnostics_can_limit_to_eligible_universe(tmp_path):
    from ashare_alpha.analysis.factor_report import generate_factor_diagnostics_report

    factors, price = _make_factor_fixture()
    eligible = factors[
        factors["trade_date"].isin(["20200102", "20200103"])
        & factors["ts_code"].isin([f"S{i:03d}" for i in range(21)])
    ][["ts_code", "trade_date"]]

    outputs = generate_factor_diagnostics_report(
        factors,
        price,
        output_dir=tmp_path,
        factor_names=["value"],
        base_horizon=1,
        decay_horizons=(1,),
        n_groups=3,
        eligible_universe=eligible,
    )

    coverage = outputs["coverage"]
    assert set(coverage["trade_date"]) == {"20200102", "20200103"}
    assert coverage["total_count"].max() == 21
    assert outputs["group_returns"]["observation_dates"].max() == 2
