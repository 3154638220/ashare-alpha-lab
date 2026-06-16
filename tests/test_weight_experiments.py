import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_derive_ic_weighted_static_weights_excludes_weak_and_negative():
    import pandas as pd
    from ashare_alpha.analysis.weight_experiments import derive_ic_weighted_static_weights

    summary = pd.DataFrame({
        "factor": ["value", "quality", "growth", "momentum", "reversal"],
        "mean_ic": [0.03, 0.003, 0.01, -0.02, 0.05],
    })

    weights = derive_ic_weighted_static_weights(summary, min_mean_ic=0.01)

    assert set(weights) == {"value", "growth", "reversal"}
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert weights["reversal"] > weights["value"] > weights["growth"]


def test_reversal_ablation_experiments_are_equal_weighted_and_named():
    from ashare_alpha.analysis.weight_experiments import build_reversal_ablation_experiments

    experiments = build_reversal_ablation_experiments()
    by_id = {experiment["experiment_id"]: experiment for experiment in experiments}

    assert set(by_id) == {
        "baseline_v1_reversal_only",
        "baseline_v1_reversal_growth",
        "baseline_v1_reversal_lowvol",
        "baseline_v1_reversal_value",
    }
    assert by_id["baseline_v1_reversal_only"]["weights"] == {"reversal": 1.0}
    assert by_id["baseline_v1_reversal_growth"]["weights"] == {
        "reversal": 0.5,
        "growth": 0.5,
    }
    assert by_id["baseline_v1_reversal_lowvol"]["weights"] == {
        "reversal": 0.5,
        "lowvol": 0.5,
    }
    assert by_id["baseline_v1_reversal_value"]["weights"] == {
        "reversal": 0.5,
        "value": 0.5,
    }


def test_reversal_growth_validation_experiments_include_growth_control_and_cost_stress():
    from ashare_alpha.analysis.weight_experiments import build_reversal_growth_validation_experiments

    experiments = build_reversal_growth_validation_experiments()
    by_id = {experiment["experiment_id"]: experiment for experiment in experiments}

    assert set(by_id) == {
        "baseline_v1_growth_only",
        "baseline_v1_reversal_growth_cost_2x",
        "baseline_v1_reversal_growth_cost_3x",
    }
    assert by_id["baseline_v1_growth_only"]["weights"] == {"growth": 1.0}
    assert by_id["baseline_v1_reversal_growth_cost_2x"]["weights"] == {
        "reversal": 0.5,
        "growth": 0.5,
    }
    assert by_id["baseline_v1_reversal_growth_cost_2x"]["cost_multiplier"] == 2.0
    assert by_id["baseline_v1_reversal_growth_cost_3x"]["cost_multiplier"] == 3.0


def test_reversal_growth_turnover_control_experiments_define_rank_buffers():
    from ashare_alpha.analysis.weight_experiments import (
        build_reversal_growth_turnover_control_experiments,
    )

    experiments = build_reversal_growth_turnover_control_experiments()
    by_id = {experiment["experiment_id"]: experiment for experiment in experiments}

    assert set(by_id) == {
        "baseline_v1_reversal_growth_buffer_70",
        "baseline_v1_reversal_growth_buffer_80",
        "baseline_v1_reversal_growth_buffer_100",
        "baseline_v1_reversal_growth_top60_buffer_90",
    }
    assert by_id["baseline_v1_reversal_growth_buffer_70"]["selection"] == {
        "method": "rank_buffer",
        "top_n": 50,
        "entry_rank": 50,
        "exit_rank": 70,
    }
    assert by_id["baseline_v1_reversal_growth_buffer_80"]["weights"] == {
        "reversal": 0.5,
        "growth": 0.5,
    }
    assert by_id["baseline_v1_reversal_growth_buffer_80"]["selection"] == {
        "method": "rank_buffer",
        "top_n": 50,
        "entry_rank": 50,
        "exit_rank": 80,
    }
    assert by_id["baseline_v1_reversal_growth_buffer_100"]["selection"]["exit_rank"] == 100
    assert by_id["baseline_v1_reversal_growth_top60_buffer_90"]["selection"] == {
        "method": "rank_buffer",
        "top_n": 60,
        "entry_rank": 60,
        "exit_rank": 90,
    }


def test_reversal_growth_buffer_cost_experiments_stress_top80_buffer():
    from ashare_alpha.analysis.weight_experiments import (
        build_reversal_growth_buffer_cost_experiments,
    )

    experiments = build_reversal_growth_buffer_cost_experiments()
    by_id = {experiment["experiment_id"]: experiment for experiment in experiments}

    assert set(by_id) == {
        "baseline_v1_reversal_growth_buffer_80_cost_2x",
        "baseline_v1_reversal_growth_buffer_80_cost_3x",
    }
    assert by_id["baseline_v1_reversal_growth_buffer_80_cost_2x"]["selection"] == {
        "method": "rank_buffer",
        "top_n": 50,
        "entry_rank": 50,
        "exit_rank": 80,
    }
    assert by_id["baseline_v1_reversal_growth_buffer_80_cost_2x"]["cost_multiplier"] == 2.0
    assert by_id["baseline_v1_reversal_growth_buffer_80_cost_3x"]["cost_multiplier"] == 3.0


def test_rank_buffered_signal_retains_existing_holdings_inside_exit_band():
    import pandas as pd
    from ashare_alpha.analysis.weight_experiments import build_rank_buffered_signal

    score = pd.DataFrame({
        "trade_date": ["20200102"] * 6,
        "ts_code": ["A", "B", "C", "D", "E", "F"],
        "score": [6, 5, 4, 3, 2, 1],
    })
    universe = pd.DataFrame({
        "ts_code": ["A", "B", "C", "D", "E", "F"],
        "industry_code": ["I1", "I1", "I2", "I2", "I3", "I3"],
    })

    signal = build_rank_buffered_signal(
        score,
        universe,
        previous_holdings={"D", "E", "Z"},
        top_n=3,
        entry_rank=3,
        exit_rank=5,
    )

    assert signal["ts_code"].tolist() == ["A", "D", "E"]
    assert signal.set_index("ts_code").loc["D", "is_buffer_retained"]
    assert signal.set_index("ts_code").loc["E", "is_buffer_retained"]
    assert signal.set_index("ts_code").loc["A", "industry_code"] == "I1"


def test_rank_buffered_signal_without_previous_holdings_matches_top_n():
    import pandas as pd
    from ashare_alpha.analysis.weight_experiments import build_rank_buffered_signal

    score = pd.DataFrame({
        "trade_date": ["20200102"] * 4,
        "ts_code": ["A", "B", "C", "D"],
        "score": [1, 4, 3, 2],
    })
    universe = pd.DataFrame({"ts_code": ["A", "B", "C", "D"]})

    signal = build_rank_buffered_signal(
        score,
        universe,
        previous_holdings=set(),
        top_n=2,
        entry_rank=2,
        exit_rank=3,
    )

    assert signal["ts_code"].tolist() == ["B", "C"]
    assert signal["rank"].tolist() == [1, 2]


def test_scale_cost_config_scales_rates_without_mutating_other_fields():
    from ashare_alpha.analysis.weight_experiments import scale_cost_config

    cost_config = {
        "commission_rate": 0.0003,
        "stamp_tax_rate": 0.0005,
        "exchange_fee_rate": 0.0000341,
        "slippage_rate": 0.0005,
        "lot_size": 100,
        "liquidity": {"max_trade_pct_of_avg_amount_20": 0.05},
    }

    scaled = scale_cost_config(cost_config, 3.0)

    assert scaled["commission_rate"] == 0.0009
    assert scaled["stamp_tax_rate"] == 0.0015
    assert scaled["exchange_fee_rate"] == 0.0001023
    assert scaled["slippage_rate"] == 0.0015
    assert scaled["lot_size"] == 100
    assert scaled["liquidity"] == {"max_trade_pct_of_avg_amount_20": 0.05}
    assert cost_config["commission_rate"] == 0.0003


def test_portfolio_factor_exposure_is_weighted_by_target_weight():
    import pandas as pd
    from ashare_alpha.analysis.weight_experiments import (
        calc_portfolio_factor_exposure,
        summarize_portfolio_factor_exposure,
    )

    target_weights = pd.DataFrame({
        "rebalance_date": ["20200102", "20200102", "20200203"],
        "ts_code": ["A", "B", "A"],
        "target_weight": [0.25, 0.75, 1.0],
        "industry_code": ["I1", "I2", "I1"],
        "value": [1.0, 3.0, 5.0],
        "reversal": [2.0, 0.0, 4.0],
    })

    exposure = calc_portfolio_factor_exposure(target_weights, ["value", "reversal"])

    first = exposure[exposure["rebalance_date"] == "20200102"].iloc[0]
    assert first["exposure_value"] == 2.5
    assert first["exposure_reversal"] == 0.5
    assert first["max_industry_weight"] == 0.75

    summary = summarize_portfolio_factor_exposure(exposure)
    assert summary["avg_holding_count"] == 1.5
    assert summary["max_observed_industry_weight"] == 1.0


def test_rebalance_turnover_summarizes_name_retention_and_weight_turnover():
    import pandas as pd
    from ashare_alpha.analysis.weight_experiments import (
        calc_rebalance_turnover,
        summarize_rebalance_turnover,
    )

    target_weights = pd.DataFrame({
        "rebalance_date": ["20200102", "20200102", "20200203", "20200203"],
        "ts_code": ["A", "B", "A", "C"],
        "target_weight": [0.5, 0.5, 0.6, 0.4],
    })

    turnover = calc_rebalance_turnover(target_weights)
    second = turnover[turnover["rebalance_date"] == "20200203"].iloc[0]

    assert second["retained_count"] == 1
    assert second["added_count"] == 1
    assert second["removed_count"] == 1
    assert second["name_retention"] == 0.5
    assert abs(second["target_weight_turnover"] - 0.5) < 1e-12

    summary = summarize_rebalance_turnover(turnover)
    assert summary["avg_name_retention"] == 0.5
    assert summary["avg_target_weight_turnover"] == 0.5


def test_experiment_diagnostics_summarize_returns_drawdowns_and_turnover_by_year():
    import pandas as pd
    from ashare_alpha.analysis.weight_experiments import (
        calc_annual_return_diagnostics,
        calc_monthly_drawdown_diagnostics,
        summarize_annual_return_diagnostics,
        summarize_monthly_drawdown_diagnostics,
        summarize_rebalance_turnover_by_year,
    )

    nav = pd.DataFrame({
        "trade_date": ["20200102", "20200131", "20200203", "20210104", "20210129"],
        "total_value": [100.0, 110.0, 105.0, 120.0, 90.0],
    })

    annual = calc_annual_return_diagnostics(nav)
    assert annual["year"].tolist() == [2020, 2021]
    assert abs(annual.loc[annual["year"] == 2020, "return"].iloc[0] - 0.05) < 1e-12

    annual_summary = summarize_annual_return_diagnostics(annual)
    assert annual_summary["negative_year_count"] == 1

    monthly = calc_monthly_drawdown_diagnostics(nav)
    assert monthly["year_month"].tolist() == ["2020-01", "2020-02", "2021-01"]
    assert abs(monthly.loc[monthly["year_month"] == "2021-01", "monthly_return"].iloc[0] + 0.25) < 1e-12

    monthly_summary = summarize_monthly_drawdown_diagnostics(monthly)
    assert monthly_summary["worst_month"] == "2021-01"
    assert monthly_summary["worst_month_return"] == -0.25

    turnover = pd.DataFrame({
        "rebalance_date": ["20200102", "20200203", "20210104"],
        "target_weight_turnover": [None, 0.4, 0.6],
        "name_retention": [None, 0.6, 0.4],
        "added_count": [None, 20, 30],
        "removed_count": [None, 20, 30],
    })
    by_year = summarize_rebalance_turnover_by_year(turnover)

    assert by_year["year"].tolist() == [2020, 2021]
    assert by_year.loc[by_year["year"] == 2020, "rebalance_count"].iloc[0] == 2
    assert abs(by_year.loc[by_year["year"] == 2021, "avg_target_weight_turnover"].iloc[0] - 0.6) < 1e-12
