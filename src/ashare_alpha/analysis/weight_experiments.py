from copy import deepcopy

import numpy as np
import pandas as pd


FACTOR_NAMES = ["value", "quality", "growth", "lowvol", "momentum", "reversal"]
COST_RATE_KEYS = ("commission_rate", "stamp_tax_rate", "exchange_fee_rate", "slippage_rate")


def make_equal_factor_weights(factor_names: list[str] | tuple[str, ...]) -> dict[str, float]:
    names = [str(name) for name in factor_names]
    if not names:
        raise ValueError("factor_names must contain at least one factor")
    if len(set(names)) != len(names):
        raise ValueError("factor_names must not contain duplicate factors")

    weight = 1.0 / len(names)
    return {name: weight for name in names}


def build_reversal_ablation_experiments() -> list[dict]:
    return [
        {
            "experiment_id": "baseline_v1_reversal_only",
            "weights": make_equal_factor_weights(["reversal"]),
            "is_reference": False,
            "notes": "Single-factor reversal baseline for checking whether paired-factor returns are mostly reversal-driven.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth",
            "weights": make_equal_factor_weights(["reversal", "growth"]),
            "is_reference": False,
            "notes": "Equal-weight reversal and growth to test whether growth adds payoff beyond reversal.",
        },
        {
            "experiment_id": "baseline_v1_reversal_lowvol",
            "weights": make_equal_factor_weights(["reversal", "lowvol"]),
            "is_reference": False,
            "notes": "Equal-weight reversal and lowvol to test lowvol's contribution despite payoff conflict.",
        },
        {
            "experiment_id": "baseline_v1_reversal_value",
            "weights": make_equal_factor_weights(["reversal", "value"]),
            "is_reference": False,
            "notes": "Equal-weight reversal and value to test value's contribution despite payoff conflict.",
        },
    ]


def build_reversal_growth_validation_experiments() -> list[dict]:
    reversal_growth_weights = make_equal_factor_weights(["reversal", "growth"])
    return [
        {
            "experiment_id": "baseline_v1_growth_only",
            "weights": make_equal_factor_weights(["growth"]),
            "is_reference": False,
            "notes": "Single-factor growth control to check whether reversal_growth is truly complementary.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_cost_2x",
            "weights": reversal_growth_weights,
            "cost_multiplier": 2.0,
            "is_reference": False,
            "notes": "Equal-weight reversal and growth under 2x trading cost stress.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_cost_3x",
            "weights": reversal_growth_weights,
            "cost_multiplier": 3.0,
            "is_reference": False,
            "notes": "Equal-weight reversal and growth under 3x trading cost stress.",
        },
    ]


def build_reversal_growth_turnover_control_experiments() -> list[dict]:
    reversal_growth_weights = make_equal_factor_weights(["reversal", "growth"])
    return [
        {
            "experiment_id": "baseline_v1_reversal_growth_buffer_70",
            "weights": reversal_growth_weights,
            "selection": {
                "method": "rank_buffer",
                "top_n": 50,
                "entry_rank": 50,
                "exit_rank": 70,
            },
            "is_reference": False,
            "notes": "Equal-weight reversal and growth with Top50 entry and narrower Top70 holding buffer.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_buffer_80",
            "weights": reversal_growth_weights,
            "selection": {
                "method": "rank_buffer",
                "top_n": 50,
                "entry_rank": 50,
                "exit_rank": 80,
            },
            "is_reference": False,
            "notes": "Equal-weight reversal and growth with Top50 entry and Top80 holding buffer.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_buffer_100",
            "weights": reversal_growth_weights,
            "selection": {
                "method": "rank_buffer",
                "top_n": 50,
                "entry_rank": 50,
                "exit_rank": 100,
            },
            "is_reference": False,
            "notes": "Equal-weight reversal and growth with Top50 entry and wider Top100 holding buffer.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_top60_buffer_80",
            "weights": reversal_growth_weights,
            "selection": {
                "method": "rank_buffer",
                "top_n": 60,
                "entry_rank": 60,
                "exit_rank": 80,
            },
            "is_reference": False,
            "notes": "Equal-weight reversal and growth with Top60 entry and narrower Top80 holding buffer.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_top60_buffer_90",
            "weights": reversal_growth_weights,
            "selection": {
                "method": "rank_buffer",
                "top_n": 60,
                "entry_rank": 60,
                "exit_rank": 90,
            },
            "is_reference": False,
            "notes": "Equal-weight reversal and growth with Top60 entry and Top90 holding buffer.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_top60_buffer_100",
            "weights": reversal_growth_weights,
            "selection": {
                "method": "rank_buffer",
                "top_n": 60,
                "entry_rank": 60,
                "exit_rank": 100,
            },
            "is_reference": False,
            "notes": "Equal-weight reversal and growth with Top60 entry and wider Top100 holding buffer.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_top70_buffer_100",
            "weights": reversal_growth_weights,
            "selection": {
                "method": "rank_buffer",
                "top_n": 70,
                "entry_rank": 70,
                "exit_rank": 100,
            },
            "is_reference": False,
            "notes": "Equal-weight reversal and growth with Top70 entry and Top100 holding buffer.",
        },
    ]


def build_reversal_growth_buffer_cost_experiments() -> list[dict]:
    reversal_growth_weights = make_equal_factor_weights(["reversal", "growth"])
    return [
        {
            "experiment_id": "baseline_v1_reversal_growth_buffer_80_cost_2x",
            "weights": reversal_growth_weights,
            "selection": {
                "method": "rank_buffer",
                "top_n": 50,
                "entry_rank": 50,
                "exit_rank": 80,
            },
            "cost_multiplier": 2.0,
            "is_reference": False,
            "notes": "Top50/Top80 buffered reversal_growth under 2x trading cost stress.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_buffer_80_cost_3x",
            "weights": reversal_growth_weights,
            "selection": {
                "method": "rank_buffer",
                "top_n": 50,
                "entry_rank": 50,
                "exit_rank": 80,
            },
            "cost_multiplier": 3.0,
            "is_reference": False,
            "notes": "Top50/Top80 buffered reversal_growth under 3x trading cost stress.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_top60_buffer_90_cost_2x",
            "weights": reversal_growth_weights,
            "selection": {
                "method": "rank_buffer",
                "top_n": 60,
                "entry_rank": 60,
                "exit_rank": 90,
            },
            "cost_multiplier": 2.0,
            "is_reference": False,
            "notes": "Top60/Top90 buffered reversal_growth under 2x trading cost stress.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_top60_buffer_90_cost_3x",
            "weights": reversal_growth_weights,
            "selection": {
                "method": "rank_buffer",
                "top_n": 60,
                "entry_rank": 60,
                "exit_rank": 90,
            },
            "cost_multiplier": 3.0,
            "is_reference": False,
            "notes": "Top60/Top90 buffered reversal_growth under 3x trading cost stress.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_top70_buffer_100_cost_2x",
            "weights": reversal_growth_weights,
            "selection": {
                "method": "rank_buffer",
                "top_n": 70,
                "entry_rank": 70,
                "exit_rank": 100,
            },
            "cost_multiplier": 2.0,
            "is_reference": False,
            "notes": "Top70/Top100 buffered reversal_growth under 2x trading cost stress.",
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_top70_buffer_100_cost_3x",
            "weights": reversal_growth_weights,
            "selection": {
                "method": "rank_buffer",
                "top_n": 70,
                "entry_rank": 70,
                "exit_rank": 100,
            },
            "cost_multiplier": 3.0,
            "is_reference": False,
            "notes": "Top70/Top100 buffered reversal_growth under 3x trading cost stress.",
        },
    ]


def _rank_buffer_selection_config(
    top_n: int,
    exit_rank: int,
    max_turnover_per_rebalance: float | None = None,
    retention_quality_rank: int | None = None,
) -> dict:
    selection = {
        "method": "rank_buffer",
        "top_n": top_n,
        "entry_rank": top_n,
        "exit_rank": exit_rank,
    }
    if max_turnover_per_rebalance is not None:
        selection["max_turnover_per_rebalance"] = max_turnover_per_rebalance
    if retention_quality_rank is not None:
        selection["retention_quality_rank"] = retention_quality_rank
    return selection


def build_reversal_growth_turnover_cap_experiments() -> list[dict]:
    reversal_growth_weights = make_equal_factor_weights(["reversal", "growth"])
    experiments = []
    bases = [
        ("top60_buffer_90", 60, 90, "Top60/Top90 buffered reversal_growth"),
        ("top70_buffer_100", 70, 100, "Top70/Top100 buffered reversal_growth"),
    ]
    for suffix, top_n, exit_rank, label in bases:
        for cap in [0.30, 0.40, 0.50]:
            cap_pct = int(round(cap * 100))
            experiments.append({
                "experiment_id": (
                    f"baseline_v1_reversal_growth_{suffix}_turnover_cap_{cap_pct}"
                ),
                "weights": reversal_growth_weights,
                "selection": _rank_buffer_selection_config(
                    top_n=top_n,
                    exit_rank=exit_rank,
                    max_turnover_per_rebalance=cap,
                ),
                "is_reference": False,
                "notes": (
                    f"{label} with max target name turnover capped at {cap_pct}% "
                    "per rebalance when enough prior holdings remain eligible."
                ),
            })
    return experiments


def _rank_soft_turnover_selection_config(
    top_n: int,
    retention_rank_bonus: int,
    force_exit_rank: int,
    retention_quality_rank: int | None = None,
) -> dict:
    selection = {
        "method": "rank_soft_turnover",
        "top_n": top_n,
        "retention_rank_bonus": retention_rank_bonus,
        "force_exit_rank": force_exit_rank,
    }
    if retention_quality_rank is not None:
        selection["retention_quality_rank"] = retention_quality_rank
    return selection


def build_reversal_growth_soft_turnover_experiments() -> list[dict]:
    reversal_growth_weights = make_equal_factor_weights(["reversal", "growth"])
    return [
        {
            "experiment_id": "baseline_v1_reversal_growth_top60_soft_bonus_10_exit_90",
            "weights": reversal_growth_weights,
            "selection": _rank_soft_turnover_selection_config(
                top_n=60,
                retention_rank_bonus=10,
                force_exit_rank=90,
            ),
            "is_reference": False,
            "notes": (
                "Top60 reversal_growth with a soft rank bonus of 10 for prior holdings, "
                "forcing old names out once raw rank is worse than 90."
            ),
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_top60_soft_bonus_20_exit_90",
            "weights": reversal_growth_weights,
            "selection": _rank_soft_turnover_selection_config(
                top_n=60,
                retention_rank_bonus=20,
                force_exit_rank=90,
            ),
            "is_reference": False,
            "notes": (
                "Top60 reversal_growth with a soft rank bonus of 20 for prior holdings, "
                "forcing old names out once raw rank is worse than 90."
            ),
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_top60_soft_bonus_30_exit_90",
            "weights": reversal_growth_weights,
            "selection": _rank_soft_turnover_selection_config(
                top_n=60,
                retention_rank_bonus=30,
                force_exit_rank=90,
            ),
            "is_reference": False,
            "notes": (
                "Top60 reversal_growth with a soft rank bonus of 30 for prior holdings, "
                "roughly allowing rank 90 old names to compete at the Top60 boundary."
            ),
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_top70_soft_bonus_30_exit_100",
            "weights": reversal_growth_weights,
            "selection": _rank_soft_turnover_selection_config(
                top_n=70,
                retention_rank_bonus=30,
                force_exit_rank=100,
            ),
            "is_reference": False,
            "notes": (
                "Top70 reversal_growth with a soft rank bonus of 30 for prior holdings, "
                "kept as a risk-control companion to the Top70/Top100 buffered candidate."
            ),
        },
    ]


def build_reversal_growth_soft_turnover_cost_experiments() -> list[dict]:
    reversal_growth_weights = make_equal_factor_weights(["reversal", "growth"])
    bases = [
        (
            "top60_soft_bonus_30_exit_90",
            60,
            30,
            90,
            "Top60 soft +30 / exit90 reversal_growth",
        ),
        (
            "top70_soft_bonus_30_exit_100",
            70,
            30,
            100,
            "Top70 soft +30 / exit100 reversal_growth",
        ),
    ]
    experiments = []
    for suffix, top_n, retention_rank_bonus, force_exit_rank, label in bases:
        for multiplier in [2.0, 3.0]:
            multiplier_label = int(multiplier)
            experiments.append({
                "experiment_id": (
                    f"baseline_v1_reversal_growth_{suffix}_cost_{multiplier_label}x"
                ),
                "weights": reversal_growth_weights,
                "selection": _rank_soft_turnover_selection_config(
                    top_n=top_n,
                    retention_rank_bonus=retention_rank_bonus,
                    force_exit_rank=force_exit_rank,
                ),
                "cost_multiplier": multiplier,
                "is_reference": False,
                "notes": f"{label} under {multiplier_label}x trading cost stress.",
            })
    return experiments


def build_reversal_growth_retention_quality_experiments() -> list[dict]:
    reversal_growth_weights = make_equal_factor_weights(["reversal", "growth"])
    return [
        {
            "experiment_id": "baseline_v1_reversal_growth_top60_buffer_90_quality_rank_80",
            "weights": reversal_growth_weights,
            "selection": _rank_buffer_selection_config(
                top_n=60,
                exit_rank=90,
                retention_quality_rank=80,
            ),
            "is_reference": False,
            "notes": (
                "Top60/Top90 buffered reversal_growth where old names only receive "
                "buffer retention if raw rank remains within 80."
            ),
        },
        {
            "experiment_id": "baseline_v1_reversal_growth_top70_buffer_100_quality_rank_90",
            "weights": reversal_growth_weights,
            "selection": _rank_buffer_selection_config(
                top_n=70,
                exit_rank=100,
                retention_quality_rank=90,
            ),
            "is_reference": False,
            "notes": (
                "Top70/Top100 buffered reversal_growth where old names only receive "
                "buffer retention if raw rank remains within 90."
            ),
        },
        {
            "experiment_id": (
                "baseline_v1_reversal_growth_top60_soft_bonus_30_exit_90_quality_rank_80"
            ),
            "weights": reversal_growth_weights,
            "selection": _rank_soft_turnover_selection_config(
                top_n=60,
                retention_rank_bonus=30,
                force_exit_rank=90,
                retention_quality_rank=80,
            ),
            "is_reference": False,
            "notes": (
                "Top60 soft +30 / exit90 reversal_growth where old names only receive "
                "the rank bonus if raw rank remains within 80."
            ),
        },
        {
            "experiment_id": (
                "baseline_v1_reversal_growth_top70_soft_bonus_30_exit_100_quality_rank_90"
            ),
            "weights": reversal_growth_weights,
            "selection": _rank_soft_turnover_selection_config(
                top_n=70,
                retention_rank_bonus=30,
                force_exit_rank=100,
                retention_quality_rank=90,
            ),
            "is_reference": False,
            "notes": (
                "Top70 soft +30 / exit100 reversal_growth where old names only receive "
                "the rank bonus if raw rank remains within 90."
            ),
        },
    ]


def scale_cost_config(cost_config: dict, multiplier: float) -> dict:
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")

    scaled = deepcopy(cost_config)
    for key in COST_RATE_KEYS:
        if key in scaled:
            scaled[key] = float(scaled[key]) * multiplier
    return scaled


def _merge_universe_industry(score: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    industry_cols = [c for c in ["industry_code", "industry_name"] if c in universe.columns]
    if not industry_cols:
        return score

    industry_meta = universe[["ts_code"] + industry_cols].drop_duplicates("ts_code")
    out = score.merge(industry_meta, on="ts_code", how="left", suffixes=("", "_universe"))
    for col in industry_cols:
        universe_col = f"{col}_universe"
        if universe_col in out.columns:
            out[col] = out[col].combine_first(out[universe_col])
            out = out.drop(columns=[universe_col])
    return out


def build_rank_buffered_signal(
    score: pd.DataFrame,
    universe: pd.DataFrame,
    previous_holdings: set[str] | list[str] | tuple[str, ...] | None,
    top_n: int = 50,
    entry_rank: int | None = None,
    exit_rank: int = 80,
    max_turnover_per_rebalance: float | None = None,
    retention_quality_rank: int | None = None,
) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    if entry_rank is None:
        entry_rank = top_n

    if entry_rank <= 0 or exit_rank <= 0:
        raise ValueError("entry_rank and exit_rank must be positive")
    if exit_rank < entry_rank:
        raise ValueError("exit_rank must be greater than or equal to entry_rank")
    if max_turnover_per_rebalance is not None and not (
        0 <= max_turnover_per_rebalance <= 1
    ):
        raise ValueError("max_turnover_per_rebalance must be between 0 and 1")
    if retention_quality_rank is not None and retention_quality_rank <= 0:
        raise ValueError("retention_quality_rank must be positive")

    valid_codes = universe["ts_code"].unique()
    candidates = score[score["ts_code"].isin(valid_codes)].copy()
    if candidates.empty:
        return candidates

    candidates = _merge_universe_industry(candidates, universe)
    candidates = candidates.sort_values("score", ascending=False).reset_index(drop=True)
    candidates["rank"] = range(1, len(candidates) + 1)

    previous = {str(code) for code in (previous_holdings or [])}
    if not previous:
        selected = candidates.head(top_n).copy()
        selected["is_buffer_retained"] = False
        selected["is_turnover_cap_retained"] = False
    else:
        code_str = candidates["ts_code"].astype(str)
        retention_rank_limit = int(exit_rank)
        if retention_quality_rank is not None:
            retention_rank_limit = min(retention_rank_limit, int(retention_quality_rank))
        retained = candidates[
            (code_str.isin(previous)) & (candidates["rank"] <= retention_rank_limit)
        ].copy()
        retained = retained.sort_values("rank").head(top_n)
        retained["is_buffer_retained"] = True
        retained["is_turnover_cap_retained"] = False

        if max_turnover_per_rebalance is not None:
            max_new_count = int(np.floor(max_turnover_per_rebalance * top_n + 1e-12))
            min_retained_count = max(top_n - max_new_count, 0)
            if len(retained) < min_retained_count:
                retained_codes = set(retained["ts_code"].astype(str))
                extra_retained_mask = (
                    code_str.isin(previous)
                    & (~candidates["ts_code"].astype(str).isin(retained_codes))
                )
                if retention_quality_rank is not None:
                    extra_retained_mask = extra_retained_mask & (
                        candidates["rank"] <= int(retention_quality_rank)
                    )
                extra_retained = candidates[extra_retained_mask].copy()
                extra_retained = extra_retained.sort_values("rank").head(
                    min_retained_count - len(retained)
                )
                if not extra_retained.empty:
                    extra_retained["is_buffer_retained"] = False
                    extra_retained["is_turnover_cap_retained"] = True
                    retained = pd.concat([retained, extra_retained], ignore_index=True)

        selected_codes = set(retained["ts_code"].astype(str))
        fill_pool = candidates[
            (~candidates["ts_code"].astype(str).isin(selected_codes))
            & (candidates["rank"] <= entry_rank)
        ].copy()
        fill_pool["is_buffer_retained"] = False
        fill_pool["is_turnover_cap_retained"] = False

        need = top_n - len(retained)
        selected = pd.concat([retained, fill_pool.head(max(need, 0))], ignore_index=True)

        if len(selected) < top_n:
            selected_codes = set(selected["ts_code"].astype(str))
            fallback = candidates[~candidates["ts_code"].astype(str).isin(selected_codes)].copy()
            fallback["is_buffer_retained"] = False
            fallback["is_turnover_cap_retained"] = False
            selected = pd.concat(
                [selected, fallback.head(top_n - len(selected))],
                ignore_index=True,
            )

    selected = selected.sort_values("rank").head(top_n).copy()
    selected["buffer_entry_rank"] = int(entry_rank)
    selected["buffer_exit_rank"] = int(exit_rank)
    if max_turnover_per_rebalance is not None:
        selected["max_turnover_per_rebalance"] = float(max_turnover_per_rebalance)
    if retention_quality_rank is not None:
        selected["buffer_retention_quality_rank"] = int(retention_quality_rank)
    return selected


def build_rank_soft_turnover_signal(
    score: pd.DataFrame,
    universe: pd.DataFrame,
    previous_holdings: set[str] | list[str] | tuple[str, ...] | None,
    top_n: int = 50,
    retention_rank_bonus: int = 20,
    force_exit_rank: int | None = None,
    retention_quality_rank: int | None = None,
) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if retention_rank_bonus < 0:
        raise ValueError("retention_rank_bonus must be non-negative")
    if force_exit_rank is not None and force_exit_rank <= 0:
        raise ValueError("force_exit_rank must be positive")
    if retention_quality_rank is not None and retention_quality_rank <= 0:
        raise ValueError("retention_quality_rank must be positive")

    valid_codes = universe["ts_code"].unique()
    candidates = score[score["ts_code"].isin(valid_codes)].copy()
    if candidates.empty:
        return candidates

    candidates = _merge_universe_industry(candidates, universe)
    candidates = candidates.sort_values("score", ascending=False).reset_index(drop=True)
    candidates["rank"] = range(1, len(candidates) + 1)

    previous = {str(code) for code in (previous_holdings or [])}
    code_str = candidates["ts_code"].astype(str)
    candidates["is_previous_holding"] = code_str.isin(previous)
    candidates["soft_turnover_rank_bonus"] = 0
    candidates["soft_turnover_adjusted_rank"] = candidates["rank"].astype(float)
    candidates["soft_turnover_force_exit_rank"] = (
        int(force_exit_rank) if force_exit_rank is not None else pd.NA
    )
    candidates["soft_turnover_quality_rank"] = (
        int(retention_quality_rank) if retention_quality_rank is not None else pd.NA
    )
    candidates["is_soft_forced_exit"] = False
    candidates["is_soft_quality_rejected"] = False

    if previous:
        retainable = candidates["is_previous_holding"]
        if force_exit_rank is not None:
            forced_exit = retainable & (candidates["rank"] > int(force_exit_rank))
            candidates.loc[forced_exit, "is_soft_forced_exit"] = True
            retainable = retainable & ~forced_exit
        if retention_quality_rank is not None:
            quality_rejected = retainable & (
                candidates["rank"] > int(retention_quality_rank)
            )
            candidates.loc[quality_rejected, "is_soft_quality_rejected"] = True
            retainable = retainable & ~quality_rejected

        candidates.loc[retainable, "soft_turnover_rank_bonus"] = int(retention_rank_bonus)
        candidates.loc[retainable, "soft_turnover_adjusted_rank"] = np.maximum(
            1.0,
            candidates.loc[retainable, "rank"].astype(float) - float(retention_rank_bonus),
        )
        eligible = candidates[~candidates["is_soft_forced_exit"]].copy()
    else:
        eligible = candidates.copy()

    selected = (
        eligible.sort_values(
            ["soft_turnover_adjusted_rank", "rank", "score"],
            ascending=[True, True, False],
        )
        .head(top_n)
        .copy()
    )
    selected["is_soft_retained"] = selected["is_previous_holding"]
    selected["is_soft_bonus_retained"] = (
        selected["is_previous_holding"] & (selected["rank"] > top_n)
        & (selected["soft_turnover_rank_bonus"] > 0)
    )
    selected["is_buffer_retained"] = False
    selected["is_turnover_cap_retained"] = False
    return selected


def calc_rebalance_turnover(target_weights: pd.DataFrame) -> pd.DataFrame:
    if target_weights.empty:
        return pd.DataFrame()

    date_col = "rebalance_date" if "rebalance_date" in target_weights.columns else "trade_date"
    records = []
    previous_weights: dict[str, float] | None = None

    for rebalance_date, group in target_weights.groupby(date_col, sort=True):
        current_weights = {
            str(row.ts_code): float(row.target_weight)
            for row in group[["ts_code", "target_weight"]].itertuples(index=False)
        }
        current_codes = set(current_weights)

        record = {
            "rebalance_date": rebalance_date,
            "holding_count": int(len(current_codes)),
        }

        if previous_weights is None:
            record.update({
                "retained_count": pd.NA,
                "added_count": pd.NA,
                "removed_count": pd.NA,
                "name_retention": pd.NA,
                "target_weight_turnover": pd.NA,
            })
        else:
            previous_codes = set(previous_weights)
            all_codes = previous_codes | current_codes
            retained = previous_codes & current_codes
            abs_diff = sum(
                abs(current_weights.get(code, 0.0) - previous_weights.get(code, 0.0))
                for code in all_codes
            )
            record.update({
                "retained_count": int(len(retained)),
                "added_count": int(len(current_codes - previous_codes)),
                "removed_count": int(len(previous_codes - current_codes)),
                "name_retention": (
                    float(len(retained) / len(previous_codes))
                    if previous_codes
                    else pd.NA
                ),
                "target_weight_turnover": float(0.5 * abs_diff),
            })

        records.append(record)
        previous_weights = current_weights

    return pd.DataFrame(records)


def summarize_rebalance_turnover(turnover: pd.DataFrame) -> dict:
    if turnover.empty:
        return {}

    summary = {}
    metrics = [
        "target_weight_turnover",
        "name_retention",
        "added_count",
        "removed_count",
    ]
    for metric in metrics:
        if metric in turnover.columns:
            values = pd.to_numeric(turnover[metric], errors="coerce")
            summary[f"avg_{metric}"] = float(values.mean())
    return summary


def _parse_date_series(values: pd.Series) -> pd.Series:
    as_str = values.astype(str)
    parsed = pd.to_datetime(as_str, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(as_str, errors="coerce")
    return parsed.fillna(fallback)


def _prepare_nav_series(nav: pd.DataFrame) -> pd.DataFrame:
    if nav.empty:
        return pd.DataFrame()

    out = nav.copy()
    if "nav" not in out.columns:
        if "total_value" not in out.columns:
            raise ValueError("nav must include either nav or total_value")
        out["nav"] = out["total_value"] / out["total_value"].iloc[0]

    out["trade_date_parsed"] = _parse_date_series(out["trade_date"])
    out = out.dropna(subset=["trade_date_parsed"]).sort_values("trade_date_parsed")
    return out


def calc_annual_return_diagnostics(nav: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_nav_series(nav)
    if prepared.empty:
        return pd.DataFrame()

    prepared["year"] = prepared["trade_date_parsed"].dt.year
    records = []
    for year, group in prepared.groupby("year", sort=True):
        first = group.iloc[0]
        last = group.iloc[-1]
        if first["nav"] == 0:
            annual_return = np.nan
        else:
            annual_return = float(last["nav"] / first["nav"] - 1)
        records.append({
            "year": int(year),
            "start_date": first["trade_date"],
            "end_date": last["trade_date"],
            "trading_days": int(len(group)),
            "start_nav": float(first["nav"]),
            "end_nav": float(last["nav"]),
            "return": annual_return,
        })

    return pd.DataFrame(records)


def calc_monthly_drawdown_diagnostics(nav: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_nav_series(nav)
    if prepared.empty:
        return pd.DataFrame()

    prepared["drawdown"] = prepared["nav"] / prepared["nav"].cummax() - 1
    prepared["year_month"] = prepared["trade_date_parsed"].dt.to_period("M").astype(str)

    records = []
    for year_month, group in prepared.groupby("year_month", sort=True):
        first = group.iloc[0]
        last = group.iloc[-1]
        worst_idx = group["drawdown"].idxmin()
        worst = group.loc[worst_idx]
        if first["nav"] == 0:
            monthly_return = np.nan
        else:
            monthly_return = float(last["nav"] / first["nav"] - 1)
        records.append({
            "year_month": str(year_month),
            "start_date": first["trade_date"],
            "end_date": last["trade_date"],
            "trading_days": int(len(group)),
            "monthly_return": monthly_return,
            "month_end_drawdown": float(last["drawdown"]),
            "min_drawdown": float(group["drawdown"].min()),
            "worst_drawdown_date": worst["trade_date"],
        })

    return pd.DataFrame(records)


def summarize_annual_return_diagnostics(annual_returns: pd.DataFrame) -> dict:
    if annual_returns.empty:
        return {}

    returns = pd.to_numeric(annual_returns["return"], errors="coerce")
    return {
        "min_annual_return": float(returns.min()),
        "max_annual_return": float(returns.max()),
        "negative_year_count": int((returns < 0).sum()),
    }


def summarize_monthly_drawdown_diagnostics(monthly_drawdown: pd.DataFrame) -> dict:
    if monthly_drawdown.empty:
        return {}

    monthly = monthly_drawdown.copy()
    monthly["monthly_return"] = pd.to_numeric(monthly["monthly_return"], errors="coerce")
    monthly["min_drawdown"] = pd.to_numeric(monthly["min_drawdown"], errors="coerce")

    worst_return = monthly.loc[monthly["monthly_return"].idxmin()]
    worst_drawdown = monthly.loc[monthly["min_drawdown"].idxmin()]
    return {
        "worst_month": str(worst_return["year_month"]),
        "worst_month_return": float(worst_return["monthly_return"]),
        "worst_drawdown_month": str(worst_drawdown["year_month"]),
        "worst_month_min_drawdown": float(worst_drawdown["min_drawdown"]),
    }


def summarize_rebalance_turnover_by_year(turnover: pd.DataFrame) -> pd.DataFrame:
    if turnover.empty:
        return pd.DataFrame()

    out = turnover.copy()
    out["rebalance_date_parsed"] = _parse_date_series(out["rebalance_date"])
    out = out.dropna(subset=["rebalance_date_parsed"])
    if out.empty:
        return pd.DataFrame()

    out["year"] = out["rebalance_date_parsed"].dt.year
    numeric_cols = [
        "target_weight_turnover",
        "name_retention",
        "added_count",
        "removed_count",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    aggregations = {
        "rebalance_count": ("rebalance_date", "count"),
    }
    if "target_weight_turnover" in out.columns:
        aggregations["avg_target_weight_turnover"] = ("target_weight_turnover", "mean")
        aggregations["max_target_weight_turnover"] = ("target_weight_turnover", "max")
    if "name_retention" in out.columns:
        aggregations["avg_name_retention"] = ("name_retention", "mean")
    if "added_count" in out.columns:
        aggregations["avg_added_count"] = ("added_count", "mean")
    if "removed_count" in out.columns:
        aggregations["avg_removed_count"] = ("removed_count", "mean")

    return out.groupby("year", sort=True).agg(**aggregations).reset_index()


def build_pressure_year_diagnostics(
    annual_returns: pd.DataFrame,
    monthly_drawdown: pd.DataFrame,
    turnover_by_year: pd.DataFrame,
    years: list[int] | tuple[int, ...],
) -> pd.DataFrame:
    records = []
    annual = annual_returns.copy()
    monthly = monthly_drawdown.copy()
    turnover = turnover_by_year.copy()

    if not annual.empty and "year" in annual.columns:
        annual["year"] = pd.to_numeric(annual["year"], errors="coerce")

    if not monthly.empty and "year_month" in monthly.columns:
        monthly["year"] = pd.to_numeric(
            monthly["year_month"].astype(str).str.slice(0, 4),
            errors="coerce",
        )

    if not turnover.empty and "year" in turnover.columns:
        turnover["year"] = pd.to_numeric(turnover["year"], errors="coerce")

    for year in years:
        year = int(year)
        record = {
            "year": year,
            "annual_return": pd.NA,
            "annual_start_date": pd.NA,
            "annual_end_date": pd.NA,
            "worst_month": pd.NA,
            "worst_month_return": pd.NA,
            "worst_drawdown_month": pd.NA,
            "worst_month_min_drawdown": pd.NA,
            "rebalance_count": pd.NA,
            "avg_target_weight_turnover": pd.NA,
            "max_target_weight_turnover": pd.NA,
            "avg_name_retention": pd.NA,
            "avg_added_count": pd.NA,
            "avg_removed_count": pd.NA,
        }

        if not annual.empty and "year" in annual.columns:
            annual_year = annual[annual["year"] == year]
            if not annual_year.empty:
                row = annual_year.iloc[0]
                record["annual_return"] = row.get("return", pd.NA)
                record["annual_start_date"] = row.get("start_date", pd.NA)
                record["annual_end_date"] = row.get("end_date", pd.NA)

        if not monthly.empty and "year" in monthly.columns:
            monthly_year = monthly[monthly["year"] == year].copy()
            if not monthly_year.empty:
                if "monthly_return" in monthly_year.columns:
                    monthly_year["monthly_return"] = pd.to_numeric(
                        monthly_year["monthly_return"],
                        errors="coerce",
                    )
                    valid_return = monthly_year.dropna(subset=["monthly_return"])
                    if not valid_return.empty:
                        worst_return = valid_return.loc[valid_return["monthly_return"].idxmin()]
                        record["worst_month"] = worst_return.get("year_month", pd.NA)
                        record["worst_month_return"] = worst_return.get("monthly_return", pd.NA)

                if "min_drawdown" in monthly_year.columns:
                    monthly_year["min_drawdown"] = pd.to_numeric(
                        monthly_year["min_drawdown"],
                        errors="coerce",
                    )
                    valid_drawdown = monthly_year.dropna(subset=["min_drawdown"])
                    if not valid_drawdown.empty:
                        worst_drawdown = valid_drawdown.loc[
                            valid_drawdown["min_drawdown"].idxmin()
                        ]
                        record["worst_drawdown_month"] = worst_drawdown.get("year_month", pd.NA)
                        record["worst_month_min_drawdown"] = worst_drawdown.get(
                            "min_drawdown",
                            pd.NA,
                        )

        if not turnover.empty and "year" in turnover.columns:
            turnover_year = turnover[turnover["year"] == year]
            if not turnover_year.empty:
                row = turnover_year.iloc[0]
                for col in [
                    "rebalance_count",
                    "avg_target_weight_turnover",
                    "max_target_weight_turnover",
                    "avg_name_retention",
                    "avg_added_count",
                    "avg_removed_count",
                ]:
                    record[col] = row.get(col, pd.NA)

        records.append(record)

    return pd.DataFrame(records)


def build_pressure_month_attribution(
    monthly_drawdown: pd.DataFrame,
    turnover: pd.DataFrame,
    exposure: pd.DataFrame,
    target_weights: pd.DataFrame,
    months: list[str] | tuple[str, ...],
    factor_names: list[str] | None = None,
) -> pd.DataFrame:
    if factor_names is None:
        factor_names = FACTOR_NAMES

    records = []
    monthly = monthly_drawdown.copy()
    turn = turnover.copy()
    exp = exposure.copy()
    weights = target_weights.copy()

    if not monthly.empty and "year_month" in monthly.columns:
        monthly["year_month"] = monthly["year_month"].astype(str)

    if not turn.empty and "rebalance_date" in turn.columns:
        turn["rebalance_date_parsed"] = _parse_date_series(turn["rebalance_date"])
        turn["year_month"] = turn["rebalance_date_parsed"].dt.to_period("M").astype(str)

    if not exp.empty and "rebalance_date" in exp.columns:
        exp["rebalance_date_parsed"] = _parse_date_series(exp["rebalance_date"])
        exp["year_month"] = exp["rebalance_date_parsed"].dt.to_period("M").astype(str)
        exp = exp.sort_values("rebalance_date_parsed")
        for factor_name in factor_names:
            col = f"exposure_{factor_name}"
            if col in exp.columns:
                exp[f"delta_{col}"] = pd.to_numeric(exp[col], errors="coerce").diff()
        if "max_industry_weight" in exp.columns:
            exp["delta_max_industry_weight"] = pd.to_numeric(
                exp["max_industry_weight"],
                errors="coerce",
            ).diff()

    date_col = "rebalance_date" if "rebalance_date" in weights.columns else "trade_date"
    if not weights.empty and date_col in weights.columns:
        weights["rebalance_date_parsed"] = _parse_date_series(weights[date_col])
        weights["year_month"] = weights["rebalance_date_parsed"].dt.to_period("M").astype(str)

    for month in months:
        month = str(month)
        record = {
            "year_month": month,
            "monthly_return": pd.NA,
            "min_drawdown": pd.NA,
            "worst_drawdown_date": pd.NA,
            "rebalance_date": pd.NA,
            "holding_count": pd.NA,
            "target_weight_turnover": pd.NA,
            "name_retention": pd.NA,
            "added_count": pd.NA,
            "removed_count": pd.NA,
            "max_industry_weight": pd.NA,
            "delta_max_industry_weight": pd.NA,
            "top_industry_code": pd.NA,
            "top_industry_name": pd.NA,
            "top_industry_weight": pd.NA,
            "avg_score": pd.NA,
            "median_rank": pd.NA,
            "buffer_retained_count": pd.NA,
            "turnover_cap_retained_count": pd.NA,
            "soft_retained_count": pd.NA,
            "soft_bonus_retained_count": pd.NA,
            "soft_quality_rejected_count": pd.NA,
            "retention_rank_bonus": pd.NA,
            "force_exit_rank": pd.NA,
            "retention_quality_rank": pd.NA,
        }

        if not monthly.empty and "year_month" in monthly.columns:
            month_rows = monthly[monthly["year_month"] == month]
            if not month_rows.empty:
                row = month_rows.iloc[0]
                record["monthly_return"] = row.get("monthly_return", pd.NA)
                record["min_drawdown"] = row.get("min_drawdown", pd.NA)
                record["worst_drawdown_date"] = row.get("worst_drawdown_date", pd.NA)

        if not turn.empty and "year_month" in turn.columns:
            turn_rows = turn[turn["year_month"] == month]
            if not turn_rows.empty:
                row = turn_rows.iloc[0]
                for col in [
                    "rebalance_date",
                    "holding_count",
                    "target_weight_turnover",
                    "name_retention",
                    "added_count",
                    "removed_count",
                ]:
                    record[col] = row.get(col, pd.NA)

        if not exp.empty and "year_month" in exp.columns:
            exp_rows = exp[exp["year_month"] == month]
            if not exp_rows.empty:
                row = exp_rows.iloc[0]
                record["rebalance_date"] = row.get("rebalance_date", record["rebalance_date"])
                record["holding_count"] = row.get("holding_count", record["holding_count"])
                record["max_industry_weight"] = row.get("max_industry_weight", pd.NA)
                record["delta_max_industry_weight"] = row.get(
                    "delta_max_industry_weight",
                    pd.NA,
                )
                for factor_name in factor_names:
                    exposure_col = f"exposure_{factor_name}"
                    delta_col = f"delta_{exposure_col}"
                    record[exposure_col] = row.get(exposure_col, pd.NA)
                    record[delta_col] = row.get(delta_col, pd.NA)

        if not weights.empty and "year_month" in weights.columns:
            weight_rows = weights[weights["year_month"] == month]
            if not weight_rows.empty:
                if "score" in weight_rows.columns:
                    record["avg_score"] = pd.to_numeric(
                        weight_rows["score"],
                        errors="coerce",
                    ).mean()
                if "rank" in weight_rows.columns:
                    record["median_rank"] = pd.to_numeric(
                        weight_rows["rank"],
                        errors="coerce",
                    ).median()
                if "is_buffer_retained" in weight_rows.columns:
                    record["buffer_retained_count"] = int(
                        weight_rows["is_buffer_retained"].fillna(False).astype(bool).sum()
                    )
                if "is_turnover_cap_retained" in weight_rows.columns:
                    record["turnover_cap_retained_count"] = int(
                        weight_rows["is_turnover_cap_retained"]
                        .fillna(False)
                        .astype(bool)
                        .sum()
                    )
                if "is_soft_retained" in weight_rows.columns:
                    record["soft_retained_count"] = int(
                        weight_rows["is_soft_retained"].fillna(False).astype(bool).sum()
                    )
                if "is_soft_bonus_retained" in weight_rows.columns:
                    record["soft_bonus_retained_count"] = int(
                        weight_rows["is_soft_bonus_retained"]
                        .fillna(False)
                        .astype(bool)
                        .sum()
                    )
                if "is_soft_quality_rejected" in weight_rows.columns:
                    record["soft_quality_rejected_count"] = int(
                        weight_rows["is_soft_quality_rejected"]
                        .fillna(False)
                        .astype(bool)
                        .sum()
                    )
                if "soft_turnover_rank_bonus" in weight_rows.columns:
                    bonus = pd.to_numeric(
                        weight_rows["soft_turnover_rank_bonus"],
                        errors="coerce",
                    ).max()
                    record["retention_rank_bonus"] = bonus
                if "soft_turnover_force_exit_rank" in weight_rows.columns:
                    force_exit = pd.to_numeric(
                        weight_rows["soft_turnover_force_exit_rank"],
                        errors="coerce",
                    ).max()
                    record["force_exit_rank"] = force_exit
                quality_rank_values = []
                for quality_col in [
                    "buffer_retention_quality_rank",
                    "soft_turnover_quality_rank",
                ]:
                    if quality_col in weight_rows.columns:
                        values = pd.to_numeric(
                            weight_rows[quality_col],
                            errors="coerce",
                        ).dropna()
                        if not values.empty:
                            quality_rank_values.append(values)
                if quality_rank_values:
                    record["retention_quality_rank"] = pd.concat(quality_rank_values).max()
                if "industry_code" in weight_rows.columns:
                    industry_keys = ["industry_code"]
                    if "industry_name" in weight_rows.columns:
                        industry_keys.append("industry_name")
                    industry = (
                        weight_rows.groupby(industry_keys, dropna=False)["target_weight"]
                        .sum()
                        .reset_index()
                        .sort_values("target_weight", ascending=False)
                    )
                    if not industry.empty:
                        top = industry.iloc[0]
                        record["top_industry_code"] = top.get("industry_code", pd.NA)
                        record["top_industry_name"] = top.get("industry_name", pd.NA)
                        record["top_industry_weight"] = top.get("target_weight", pd.NA)

        records.append(record)

    return pd.DataFrame(records)


def _is_true(value) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def _retention_bucket(row: pd.Series) -> str:
    if _is_true(row.get("is_soft_bonus_retained")):
        return "soft_bonus_retained"
    if _is_true(row.get("is_buffer_retained")):
        return "buffer_retained"
    if _is_true(row.get("is_turnover_cap_retained")):
        return "turnover_cap_retained"
    if _is_true(row.get("is_previous_holding")) or _is_true(row.get("is_soft_retained")):
        return "previous_kept"
    return "new_name"


def build_pressure_month_holding_contribution(
    target_weights: pd.DataFrame,
    price_panel: pd.DataFrame,
    months: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    if target_weights.empty or price_panel.empty:
        return pd.DataFrame()

    price_col = "adj_close" if "adj_close" in price_panel.columns else "close"
    required_price_cols = {"ts_code", "trade_date", price_col}
    if not required_price_cols.issubset(price_panel.columns):
        return pd.DataFrame()

    weights = target_weights.copy()
    date_col = "rebalance_date" if "rebalance_date" in weights.columns else "trade_date"
    if date_col not in weights.columns or "ts_code" not in weights.columns:
        return pd.DataFrame()

    weights["rebalance_date_parsed"] = _parse_date_series(weights[date_col])
    if "execution_date" in weights.columns:
        weights["execution_date_parsed"] = _parse_date_series(weights["execution_date"])
    else:
        weights["execution_date_parsed"] = weights["rebalance_date_parsed"]
    weights["year_month"] = weights["rebalance_date_parsed"].dt.to_period("M").astype(str)
    weights["target_weight"] = pd.to_numeric(weights.get("target_weight"), errors="coerce")

    prices = price_panel[list(required_price_cols)].copy()
    prices["trade_date_parsed"] = _parse_date_series(prices["trade_date"])
    prices[price_col] = pd.to_numeric(prices[price_col], errors="coerce")
    prices = prices.dropna(subset=["trade_date_parsed", price_col]).sort_values(
        ["ts_code", "trade_date_parsed"],
    )
    price_by_code = {ts_code: frame for ts_code, frame in prices.groupby("ts_code")}

    records = []
    for month in months:
        month = str(month)
        month_weights = weights[weights["year_month"] == month].dropna(
            subset=["target_weight", "execution_date_parsed"],
        )
        if month_weights.empty:
            continue

        month_period = pd.Period(month, freq="M")
        for _, row in month_weights.iterrows():
            code = row["ts_code"]
            code_prices = price_by_code.get(code)
            if code_prices is None or code_prices.empty:
                continue

            execution_date = row["execution_date_parsed"]
            eligible_prices = code_prices[
                (code_prices["trade_date_parsed"] >= execution_date)
                & (code_prices["trade_date_parsed"].dt.to_period("M") == month_period)
            ]
            if eligible_prices.empty:
                continue

            start = eligible_prices.iloc[0]
            end = eligible_prices.iloc[-1]
            start_price = float(start[price_col])
            end_price = float(end[price_col])
            if start_price <= 0:
                continue

            holding_return = end_price / start_price - 1.0
            target_weight = float(row["target_weight"])
            contribution = target_weight * holding_return
            record = {
                "year_month": month,
                "rebalance_date": row.get(date_col, pd.NA),
                "execution_date": row.get("execution_date", row.get(date_col, pd.NA)),
                "start_date": start["trade_date"],
                "end_date": end["trade_date"],
                "ts_code": code,
                "industry_code": row.get("industry_code", pd.NA),
                "industry_name": row.get("industry_name", pd.NA),
                "target_weight": target_weight,
                "start_price": start_price,
                "end_price": end_price,
                "holding_return": holding_return,
                "return_contribution": contribution,
                "retention_bucket": _retention_bucket(row),
                "rank": row.get("rank", pd.NA),
                "score": row.get("score", pd.NA),
                "is_previous_holding": row.get("is_previous_holding", pd.NA),
                "is_buffer_retained": row.get("is_buffer_retained", pd.NA),
                "is_turnover_cap_retained": row.get("is_turnover_cap_retained", pd.NA),
                "is_soft_retained": row.get("is_soft_retained", pd.NA),
                "is_soft_bonus_retained": row.get("is_soft_bonus_retained", pd.NA),
            }
            records.append(record)

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records)
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    return out.sort_values(
        ["year_month", "return_contribution", "target_weight"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_pressure_month_realized_pnl_contribution(
    nav: pd.DataFrame,
    positions: pd.DataFrame,
    trades: pd.DataFrame,
    target_weights: pd.DataFrame,
    months: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    if nav.empty or "total_value" not in nav.columns:
        return pd.DataFrame()

    nav_frame = nav.copy()
    nav_frame["trade_date_parsed"] = _parse_date_series(nav_frame["trade_date"])
    nav_frame["total_value"] = pd.to_numeric(nav_frame["total_value"], errors="coerce")
    nav_frame = nav_frame.dropna(subset=["trade_date_parsed", "total_value"]).sort_values(
        "trade_date_parsed",
    )
    if nav_frame.empty:
        return pd.DataFrame()

    pos = positions.copy()
    if not pos.empty and {"trade_date", "ts_code", "market_value"}.issubset(pos.columns):
        pos["trade_date_parsed"] = _parse_date_series(pos["trade_date"])
        pos["market_value"] = pd.to_numeric(pos["market_value"], errors="coerce")
        pos = pos.dropna(subset=["trade_date_parsed", "market_value"])
    else:
        pos = pd.DataFrame()

    trade = trades.copy()
    if not trade.empty and {"trade_date", "ts_code", "side", "amount", "cost"}.issubset(
        trade.columns
    ):
        trade["trade_date_parsed"] = _parse_date_series(trade["trade_date"])
        trade["amount"] = pd.to_numeric(trade["amount"], errors="coerce").fillna(0.0)
        trade["cost"] = pd.to_numeric(trade["cost"], errors="coerce").fillna(0.0)
        trade["side"] = trade["side"].astype(str).str.upper()
        trade = trade.dropna(subset=["trade_date_parsed"])
    else:
        trade = pd.DataFrame()

    weights = target_weights.copy()
    date_col = "rebalance_date" if "rebalance_date" in weights.columns else "trade_date"
    if not weights.empty and date_col in weights.columns and "ts_code" in weights.columns:
        weights["rebalance_date_parsed"] = _parse_date_series(weights[date_col])
        weights["year_month"] = weights["rebalance_date_parsed"].dt.to_period("M").astype(str)
    else:
        weights = pd.DataFrame()

    def position_values(date: pd.Timestamp) -> pd.Series:
        if pos.empty:
            return pd.Series(dtype=float)
        frame = pos[pos["trade_date_parsed"] == date]
        if frame.empty:
            return pd.Series(dtype=float)
        return frame.groupby("ts_code")["market_value"].sum()

    records = []
    for month in months:
        month = str(month)
        month_period = pd.Period(month, freq="M")
        month_nav = nav_frame[nav_frame["trade_date_parsed"].dt.to_period("M") == month_period]
        if month_nav.empty:
            continue

        first_month_date = month_nav["trade_date_parsed"].min()
        prior_nav = nav_frame[nav_frame["trade_date_parsed"] < first_month_date]
        if prior_nav.empty:
            start_row = month_nav.iloc[0]
        else:
            start_row = prior_nav.iloc[-1]
        end_row = month_nav.iloc[-1]
        start_date = start_row["trade_date_parsed"]
        end_date = end_row["trade_date_parsed"]
        start_total_value = float(start_row["total_value"])
        if start_total_value <= 0:
            continue

        start_values = position_values(start_date)
        end_values = position_values(end_date)

        if trade.empty:
            month_trades = pd.DataFrame()
        else:
            month_trades = trade[
                (trade["trade_date_parsed"] > start_date)
                & (trade["trade_date_parsed"] <= end_date)
            ].copy()

        buy_amount = pd.Series(dtype=float)
        sell_amount = pd.Series(dtype=float)
        trade_cost = pd.Series(dtype=float)
        if not month_trades.empty:
            buy_amount = (
                month_trades[month_trades["side"] == "BUY"]
                .groupby("ts_code")["amount"]
                .sum()
            )
            sell_amount = (
                month_trades[month_trades["side"] == "SELL"]
                .groupby("ts_code")["amount"]
                .sum()
            )
            trade_cost = month_trades.groupby("ts_code")["cost"].sum()

        if weights.empty:
            month_weights = pd.DataFrame()
        else:
            month_weights = weights[weights["year_month"] == month].copy()
            month_weights = month_weights.drop_duplicates("ts_code", keep="last")
            if not month_weights.empty:
                month_weights = month_weights.set_index("ts_code")

        codes = (
            set(start_values.index.astype(str))
            | set(end_values.index.astype(str))
            | set(buy_amount.index.astype(str))
            | set(sell_amount.index.astype(str))
            | set(trade_cost.index.astype(str))
        )
        for code in sorted(codes):
            start_market_value = float(start_values.get(code, 0.0))
            end_market_value = float(end_values.get(code, 0.0))
            buys = float(buy_amount.get(code, 0.0))
            sells = float(sell_amount.get(code, 0.0))
            costs = float(trade_cost.get(code, 0.0))
            gross_pnl = end_market_value - start_market_value + sells - buys
            net_pnl = gross_pnl - costs

            record = {
                "year_month": month,
                "period_start_date": start_row["trade_date"],
                "period_end_date": end_row["trade_date"],
                "ts_code": code,
                "start_total_value": start_total_value,
                "start_market_value": start_market_value,
                "end_market_value": end_market_value,
                "buy_amount": buys,
                "sell_amount": sells,
                "trade_cost": costs,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "gross_pnl_contribution": gross_pnl / start_total_value,
                "net_pnl_contribution": net_pnl / start_total_value,
                "cost_contribution": -costs / start_total_value,
                "retention_bucket": (
                    "removed_name"
                    if (start_market_value > 0 or sells > 0) and end_market_value == 0
                    else "non_target_position"
                ),
                "target_weight": pd.NA,
                "rank": pd.NA,
                "score": pd.NA,
                "industry_code": pd.NA,
                "industry_name": pd.NA,
            }

            if not month_weights.empty and code in month_weights.index:
                meta = month_weights.loc[code]
                record.update({
                    "retention_bucket": _retention_bucket(meta),
                    "target_weight": meta.get("target_weight", pd.NA),
                    "rank": meta.get("rank", pd.NA),
                    "score": meta.get("score", pd.NA),
                    "industry_code": meta.get("industry_code", pd.NA),
                    "industry_name": meta.get("industry_name", pd.NA),
                })

            records.append(record)

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records)
    for col in ["target_weight", "rank", "score"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(
        ["year_month", "net_pnl_contribution", "gross_pnl_contribution"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def derive_ic_weighted_static_weights(
    factor_summary: pd.DataFrame,
    min_mean_ic: float = 0.01,
) -> dict[str, float]:
    if factor_summary.empty:
        return {}

    required = {"factor", "mean_ic"}
    if not required.issubset(factor_summary.columns):
        missing = ", ".join(sorted(required - set(factor_summary.columns)))
        raise ValueError(f"factor_summary is missing required columns: {missing}")

    summary = factor_summary.copy()
    summary["mean_ic"] = pd.to_numeric(summary["mean_ic"], errors="coerce")
    summary = summary[summary["mean_ic"] >= min_mean_ic]
    summary = summary[summary["factor"].isin(FACTOR_NAMES)]

    total = summary["mean_ic"].sum()
    if not np.isfinite(total) or total <= 0:
        return {}

    return {
        str(row.factor): float(row.mean_ic / total)
        for row in summary.itertuples(index=False)
    }


def flatten_performance_metrics(metrics: dict) -> dict:
    keys = [
        "final_nav",
        "total_return",
        "annual_return",
        "annual_vol",
        "sharpe",
        "max_drawdown",
        "calmar",
        "turnover",
    ]
    flat = {key: metrics.get(key, np.nan) for key in keys}

    for benchmark_name, benchmark_metrics in metrics.get("benchmarks", {}).items():
        for metric_name in [
            "benchmark_total_return",
            "active_total_return",
            "excess_return",
            "annual_excess_return",
            "tracking_error",
            "information_ratio",
            "max_excess_drawdown",
        ]:
            flat[f"{benchmark_name}_{metric_name}"] = benchmark_metrics.get(metric_name, np.nan)

    return flat


def calc_portfolio_factor_exposure(
    target_weights: pd.DataFrame,
    factor_names: list[str] | None = None,
) -> pd.DataFrame:
    if factor_names is None:
        factor_names = FACTOR_NAMES

    if target_weights.empty:
        return pd.DataFrame()

    date_col = "rebalance_date" if "rebalance_date" in target_weights.columns else "trade_date"
    records = []

    for rebalance_date, group in target_weights.groupby(date_col, sort=True):
        weight_sum = group["target_weight"].sum()
        record = {
            "rebalance_date": rebalance_date,
            "holding_count": int(group["ts_code"].nunique()),
            "total_weight": float(weight_sum),
            "max_stock_weight": float(group["target_weight"].max()),
        }

        if "industry_code" in group.columns and not group.empty:
            industry_weight = group.groupby("industry_code", dropna=False)["target_weight"].sum()
            record["max_industry_weight"] = float(industry_weight.max())

        for factor_name in factor_names:
            exposure_col = f"exposure_{factor_name}"
            if factor_name not in group.columns or weight_sum <= 0:
                record[exposure_col] = np.nan
                continue
            values = pd.to_numeric(group[factor_name], errors="coerce")
            weights = group["target_weight"].where(values.notna(), 0.0)
            denom = weights.sum()
            record[exposure_col] = (
                float((values.fillna(0.0) * group["target_weight"]).sum() / denom)
                if denom > 0
                else np.nan
            )

        records.append(record)

    return pd.DataFrame(records)


def summarize_portfolio_factor_exposure(exposure: pd.DataFrame) -> dict:
    if exposure.empty:
        return {}

    summary = {
        "avg_holding_count": float(exposure["holding_count"].mean()),
        "avg_total_weight": float(exposure["total_weight"].mean()),
        "avg_max_stock_weight": float(exposure["max_stock_weight"].mean()),
    }

    if "max_industry_weight" in exposure.columns:
        summary["avg_max_industry_weight"] = float(exposure["max_industry_weight"].mean())
        summary["max_observed_industry_weight"] = float(exposure["max_industry_weight"].max())

    for col in [c for c in exposure.columns if c.startswith("exposure_")]:
        summary[f"avg_{col}"] = float(exposure[col].mean())

    return summary
