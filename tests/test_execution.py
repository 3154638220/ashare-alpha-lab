import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_is_suspended():
    import pandas as pd
    from ashare_alpha.backtest.execution import is_suspended

    suspended = pd.Series({"open": None, "close": 10.0, "vol": 1000})
    assert is_suspended(suspended)

    zero_vol = pd.Series({"open": 10.0, "close": 10.0, "vol": 0})
    assert is_suspended(zero_vol)

    normal = pd.Series({"open": 10.0, "close": 10.5, "vol": 1000})
    assert not is_suspended(normal)


def test_is_limit_up():
    import pandas as pd
    from ashare_alpha.backtest.execution import is_limit_up

    limit_up = pd.Series({"open": 11.0, "up_limit": 11.0})
    assert is_limit_up(limit_up)

    not_limit = pd.Series({"open": 10.5, "up_limit": 11.0})
    assert not is_limit_up(not_limit)

    no_data = pd.Series({"open": 10.0})
    assert not is_limit_up(no_data)


def test_is_limit_down():
    import pandas as pd
    from ashare_alpha.backtest.execution import is_limit_down

    limit_down = pd.Series({"open": 9.0, "down_limit": 9.0})
    assert is_limit_down(limit_down)

    not_limit = pd.Series({"open": 10.0, "down_limit": 9.0})
    assert not is_limit_down(not_limit)


def test_can_buy_sell():
    import pandas as pd
    from ashare_alpha.backtest.execution import can_buy, can_sell

    suspended = pd.Series({"open": None, "close": 10.0, "vol": 0})
    assert not can_buy(suspended)
    assert not can_sell(suspended)

    limit_up = pd.Series({"open": 11.0, "up_limit": 11.0, "close": 11.0, "vol": 1000})
    assert not can_buy(limit_up)
    assert can_sell(limit_up)

    limit_down = pd.Series({"open": 9.0, "down_limit": 9.0, "close": 9.0, "vol": 1000})
    assert can_buy(limit_down)
    assert not can_sell(limit_down)


def test_round_lot_shares():
    from ashare_alpha.backtest.execution import round_lot_shares

    assert round_lot_shares(150) == 100
    assert round_lot_shares(249) == 200
    assert round_lot_shares(250) == 200
    assert round_lot_shares(99) == 0
