import pandas as pd

from .account import Account
from .broker import Broker

from ashare_alpha.logger import logger


class BacktestEngine:
    def __init__(
        self,
        price_panel: pd.DataFrame,
        target_weights: pd.DataFrame,
        trade_dates: list[str],
        rebalance_dates: list[str],
        init_cash: float,
        cost_config: dict,
    ):
        self.price_by_date = {}
        for dt, g in price_panel.groupby("trade_date"):
            self.price_by_date[dt] = g.set_index("ts_code")

        self.target_weights = target_weights
        self.trade_dates = trade_dates
        self.rebalance_dates = set(rebalance_dates)
        self.account = Account(init_cash)
        self.broker = Broker(self.account, cost_config)

        self.nav_records = []
        self.trade_records = []
        self.position_records = []

    def run(self) -> dict:
        for trade_date in self.trade_dates:
            today_price = self.price_by_date.get(trade_date)
            if today_price is None:
                continue

            if trade_date in self.rebalance_dates:
                self._rebalance(trade_date, today_price)

            total_value = self.account.update_market_value_optimized(trade_date, today_price)

            self.nav_records.append({
                "trade_date": trade_date,
                "total_value": total_value,
                "cash": self.account.cash,
            })

            self._record_positions(trade_date, today_price)

        return {
            "nav": pd.DataFrame(self.nav_records),
            "trades": pd.DataFrame(self.trade_records) if self.trade_records else pd.DataFrame(),
            "positions": pd.DataFrame(self.position_records) if self.position_records else pd.DataFrame(),
        }

    def _rebalance(self, trade_date: str, today_price: pd.DataFrame):
        date_col = "execution_date" if "execution_date" in self.target_weights.columns else "rebalance_date"
        target = self.target_weights[
            self.target_weights[date_col] == trade_date
        ].copy()

        if target.empty:
            return

        total_value = self.account.update_market_value_optimized(trade_date, today_price)

        target_amount = {
            row["ts_code"]: row["target_weight"] * total_value
            for _, row in target.iterrows()
        }

        current_codes = set(self.account.positions.keys())
        target_codes = set(target_amount.keys())

        for ts_code in list(current_codes - target_codes):
            if ts_code not in today_price.index:
                continue

            row = today_price.loc[ts_code]
            shares = self.account.positions[ts_code]
            trade = self.broker.sell(trade_date, ts_code, shares, row)

            if trade:
                self.trade_records.append(trade)

        for ts_code, amount in target_amount.items():
            if ts_code not in today_price.index:
                continue

            row = today_price.loc[ts_code]
            price = row["adj_open"]
            current_shares = self.account.positions.get(ts_code, 0)
            current_amount = current_shares * price

            diff_amount = amount - current_amount

            if diff_amount > 0:
                trade = self.broker.buy(trade_date, ts_code, diff_amount, row)
            elif diff_amount < 0:
                sell_shares = min(current_shares, int(abs(diff_amount) / price))
                trade = self.broker.sell(trade_date, ts_code, sell_shares, row)
            else:
                trade = None

            if trade:
                self.trade_records.append(trade)

    def _record_positions(self, trade_date: str, today_price: pd.DataFrame):
        for ts_code, shares in self.account.positions.items():
            if ts_code not in today_price.index:
                continue

            price = today_price.loc[ts_code, "adj_close"]

            self.position_records.append({
                "trade_date": trade_date,
                "ts_code": ts_code,
                "shares": shares,
                "close_price": price,
                "market_value": shares * price,
            })
