import json
import numpy as np
import pandas as pd


def parse_trade_dates(trade_date: pd.Series) -> pd.Series:
    values = trade_date.astype(str)
    parsed = pd.to_datetime(values, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(values, errors="coerce")
    return parsed.fillna(fallback)


def prepare_nav(nav: pd.DataFrame) -> pd.DataFrame:
    nav = nav.copy()
    nav = nav.sort_values("trade_date")
    nav["nav"] = nav["total_value"] / nav["total_value"].iloc[0]
    nav["daily_ret"] = nav["nav"].pct_change().fillna(0)
    return nav


def calc_annual_return(nav: pd.DataFrame) -> float:
    n = len(nav)
    total_return = nav["nav"].iloc[-1] / nav["nav"].iloc[0] - 1
    return (1 + total_return) ** (252 / n) - 1


def calc_annual_vol(nav: pd.DataFrame) -> float:
    return nav["daily_ret"].std() * np.sqrt(252)


def calc_sharpe(nav: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
    ann_ret = calc_annual_return(nav)
    ann_vol = calc_annual_vol(nav)

    if ann_vol == 0:
        return np.nan

    return (ann_ret - risk_free_rate) / ann_vol


def calc_max_drawdown(nav: pd.DataFrame) -> float:
    cummax = nav["nav"].cummax()
    drawdown = nav["nav"] / cummax - 1
    return drawdown.min()


def calc_calmar(nav: pd.DataFrame) -> float:
    ann_ret = calc_annual_return(nav)
    max_dd = abs(calc_max_drawdown(nav))

    if max_dd == 0:
        return np.nan

    return ann_ret / max_dd


def calc_win_rate(nav: pd.DataFrame) -> float:
    monthly = _calc_monthly_returns(nav)
    if monthly is None or monthly.empty:
        return float("nan")
    return (monthly["return"] > 0).sum() / len(monthly)


def _calc_monthly_returns(nav: pd.DataFrame) -> pd.DataFrame | None:
    nav = nav.copy()
    nav["trade_date"] = parse_trade_dates(nav["trade_date"])
    nav["year_month"] = nav["trade_date"].dt.to_period("M")

    monthly = nav.groupby("year_month")["nav"].agg(["first", "last"])
    monthly["return"] = monthly["last"] / monthly["first"] - 1
    return monthly.reset_index()


def calc_monthly_returns(nav: pd.DataFrame) -> pd.DataFrame:
    return _calc_monthly_returns(nav)


def calc_annual_returns(nav: pd.DataFrame) -> pd.DataFrame:
    nav = nav.copy()
    nav["trade_date"] = parse_trade_dates(nav["trade_date"])
    nav["year"] = nav["trade_date"].dt.year

    annual = nav.groupby("year")["nav"].agg(["first", "last"])
    annual["return"] = annual["last"] / annual["first"] - 1
    return annual.reset_index()


def prepare_benchmark(benchmark: pd.DataFrame) -> pd.DataFrame:
    benchmark = benchmark.copy()
    benchmark = benchmark.sort_values("trade_date")

    value_col = None
    for col in ["nav", "total_value", "adj_close", "close"]:
        if col in benchmark.columns:
            value_col = col
            break

    if value_col is None:
        raise ValueError("Benchmark data must include one of: nav, total_value, adj_close, close")

    benchmark[value_col] = pd.to_numeric(benchmark[value_col], errors="coerce")
    benchmark = benchmark.dropna(subset=[value_col])
    benchmark = benchmark[benchmark[value_col] > 0].copy()
    if benchmark.empty:
        raise ValueError("Benchmark data is empty after cleaning")

    benchmark["nav"] = benchmark[value_col] / benchmark[value_col].iloc[0]
    benchmark["daily_ret"] = benchmark["nav"].pct_change().fillna(0)
    return benchmark


def align_nav_with_benchmark(nav: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    strategy = prepare_nav(nav)[["trade_date", "nav", "daily_ret"]].copy()
    bench = prepare_benchmark(benchmark)[["trade_date", "nav", "daily_ret"]].copy()

    strategy["trade_date"] = strategy["trade_date"].astype(str)
    bench["trade_date"] = bench["trade_date"].astype(str)

    aligned = strategy.merge(
        bench,
        on="trade_date",
        how="inner",
        suffixes=("_strategy", "_benchmark"),
    ).sort_values("trade_date")

    if aligned.empty:
        return pd.DataFrame(columns=[
            "trade_date",
            "strategy_nav",
            "benchmark_nav",
            "strategy_ret",
            "benchmark_ret",
            "active_ret",
            "excess_nav",
            "excess_ret",
            "excess_drawdown",
        ])

    aligned = aligned.rename(columns={
        "nav_strategy": "strategy_nav",
        "nav_benchmark": "benchmark_nav",
        "daily_ret_strategy": "strategy_ret",
        "daily_ret_benchmark": "benchmark_ret",
    })

    aligned["strategy_nav"] = aligned["strategy_nav"] / aligned["strategy_nav"].iloc[0]
    aligned["benchmark_nav"] = aligned["benchmark_nav"] / aligned["benchmark_nav"].iloc[0]
    aligned["strategy_ret"] = aligned["strategy_nav"].pct_change().fillna(0)
    aligned["benchmark_ret"] = aligned["benchmark_nav"].pct_change().fillna(0)
    aligned["active_ret"] = aligned["strategy_ret"] - aligned["benchmark_ret"]
    aligned["excess_nav"] = aligned["strategy_nav"] / aligned["benchmark_nav"]
    aligned["excess_ret"] = aligned["excess_nav"].pct_change().fillna(0)
    aligned["excess_drawdown"] = aligned["excess_nav"] / aligned["excess_nav"].cummax() - 1

    return aligned[[
        "trade_date",
        "strategy_nav",
        "benchmark_nav",
        "strategy_ret",
        "benchmark_ret",
        "active_ret",
        "excess_nav",
        "excess_ret",
        "excess_drawdown",
    ]]


def _annualized_nav_return(nav_values: pd.Series) -> float:
    if nav_values.empty or nav_values.iloc[0] == 0:
        return float("nan")
    total_return = nav_values.iloc[-1] / nav_values.iloc[0] - 1
    return (1 + total_return) ** (252 / len(nav_values)) - 1


def calc_benchmark_performance(nav: pd.DataFrame, benchmark: pd.DataFrame) -> dict:
    aligned = align_nav_with_benchmark(nav, benchmark)

    if len(aligned) < 2:
        return {
            "n_days": int(len(aligned)),
            "benchmark_total_return": float("nan"),
            "benchmark_annual_return": float("nan"),
            "active_total_return": float("nan"),
            "excess_return": float("nan"),
            "annual_excess_return": float("nan"),
            "tracking_error": float("nan"),
            "information_ratio": float("nan"),
            "max_excess_drawdown": float("nan"),
        }

    active_std = aligned["active_ret"].std()
    tracking_error = active_std * np.sqrt(252)
    information_ratio = (
        aligned["active_ret"].mean() / active_std * np.sqrt(252)
        if active_std and not np.isnan(active_std)
        else float("nan")
    )

    return {
        "n_days": int(len(aligned)),
        "benchmark_total_return": float(aligned["benchmark_nav"].iloc[-1] - 1),
        "benchmark_annual_return": float(_annualized_nav_return(aligned["benchmark_nav"])),
        "active_total_return": float((aligned["strategy_nav"].iloc[-1] - 1) - (aligned["benchmark_nav"].iloc[-1] - 1)),
        "excess_return": float(aligned["excess_nav"].iloc[-1] - 1),
        "annual_excess_return": float(_annualized_nav_return(aligned["excess_nav"])),
        "tracking_error": float(tracking_error),
        "information_ratio": float(information_ratio),
        "max_excess_drawdown": float(aligned["excess_drawdown"].min()),
    }


def calc_turnover(trades: pd.DataFrame, nav: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0

    nav = prepare_nav(nav)

    trade_dates = trades["trade_date"].unique()
    turnovers = []

    for td in trade_dates:
        day_trades = trades[trades["trade_date"] == td]
        if day_trades.empty:
            continue

        nav_row = nav[nav["trade_date"] == td]
        if nav_row.empty:
            continue

        total_value = nav_row.iloc[0]["total_value"]
        if total_value == 0:
            continue

        buy_amount = day_trades[day_trades["side"] == "BUY"]["amount"].sum()
        sell_amount = day_trades[day_trades["side"] == "SELL"]["amount"].sum()
        turnover = (buy_amount + sell_amount) / 2 / total_value
        turnovers.append(turnover)

    if not turnovers:
        return 0.0

    return np.mean(turnovers)


def calc_performance(
    nav: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    benchmarks: dict[str, pd.DataFrame] | None = None,
) -> dict:
    nav = prepare_nav(nav)

    metrics = {
        "annual_return": calc_annual_return(nav),
        "annual_vol": calc_annual_vol(nav),
        "sharpe": calc_sharpe(nav),
        "max_drawdown": calc_max_drawdown(nav),
        "calmar": calc_calmar(nav),
        "total_return": float(nav["nav"].iloc[-1] - 1),
        "final_nav": float(nav["nav"].iloc[-1]),
        "win_rate": calc_win_rate(nav),
    }

    if trades is not None:
        metrics["turnover"] = calc_turnover(trades, nav)

    if benchmarks:
        metrics["benchmarks"] = {
            name: calc_benchmark_performance(nav, benchmark)
            for name, benchmark in benchmarks.items()
        }

    return metrics


def save_metrics(metrics: dict, path: str = "results/metrics.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
