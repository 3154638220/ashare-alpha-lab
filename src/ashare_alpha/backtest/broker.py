from .cost import calc_buy_cost, calc_sell_cost
from .execution import can_buy, can_sell, round_lot_shares

from ashare_alpha.logger import logger


class Broker:
    def __init__(self, account, cost_config):
        self.account = account
        self.cost_config = cost_config

    def buy(self, trade_date, ts_code, target_amount, row):
        if not can_buy(row):
            return None

        price = row["adj_open"]
        shares = round_lot_shares(target_amount / price)

        if shares <= 0:
            return None

        amount = shares * price
        cost = calc_buy_cost(amount, self.cost_config)
        total_cash_need = amount + cost

        if total_cash_need > self.account.cash:
            shares = round_lot_shares(
                self.account.cash / (price * 1.001)
            )
            amount = shares * price
            cost = calc_buy_cost(amount, self.cost_config)
            total_cash_need = amount + cost

        if shares <= 0:
            return None

        self.account.cash -= total_cash_need
        self.account.positions[ts_code] = (
            self.account.positions.get(ts_code, 0) + shares
        )

        return {
            "trade_date": trade_date,
            "ts_code": ts_code,
            "side": "BUY",
            "price": price,
            "shares": shares,
            "amount": amount,
            "cost": cost,
        }

    def sell(self, trade_date, ts_code, shares, row):
        if not can_sell(row):
            return None

        price = row["adj_open"]
        amount = shares * price
        cost = calc_sell_cost(amount, self.cost_config)

        self.account.cash += amount - cost
        self.account.positions[ts_code] -= shares

        if self.account.positions[ts_code] <= 0:
            del self.account.positions[ts_code]

        return {
            "trade_date": trade_date,
            "ts_code": ts_code,
            "side": "SELL",
            "price": price,
            "shares": shares,
            "amount": amount,
            "cost": cost,
        }
