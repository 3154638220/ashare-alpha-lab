def calc_buy_cost(amount: float, config: dict) -> float:
    commission = amount * config["commission_rate"]
    exchange_fee = amount * config["exchange_fee_rate"]
    slippage = amount * config["slippage_rate"]

    return commission + exchange_fee + slippage


def calc_sell_cost(amount: float, config: dict) -> float:
    commission = amount * config["commission_rate"]
    exchange_fee = amount * config["exchange_fee_rate"]
    stamp_tax = amount * config["stamp_tax_rate"]
    slippage = amount * config["slippage_rate"]

    return commission + exchange_fee + stamp_tax + slippage
