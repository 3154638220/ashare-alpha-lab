下面把它改写成**工程实现方案**。目标不是一上来做复杂 Alpha，而是先实现一个可以跑通的 MVP：

> **日频行情 + 财务指标 + 行业分类 → 股票池过滤 → 多因子打分 → 月频调仓 → 回测 → 绩效分析 → 图表报告**

第一版不做机器学习、不做另类数据、不做复杂组合优化，只做**规则清晰、可排查偏差的多因子选股系统**。

---

# 1. GitHub 项目目录结构

```text
a-share-alpha-mvp/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── config/
│   ├── data.yaml
│   ├── strategy.yaml
│   ├── cost.yaml
│   └── backtest.yaml
│
├── data/
│   ├── raw/
│   │   ├── stock_basic.parquet
│   │   ├── trade_calendar.parquet
│   │   ├── daily.parquet
│   │   ├── adj_factor.parquet
│   │   ├── daily_basic.parquet
│   │   ├── fina_indicator.parquet
│   │   ├── industry_member.parquet
│   │   └── stk_limit.parquet
│   │
│   ├── processed/
│   │   ├── price_panel.parquet
│   │   ├── valuation_panel.parquet
│   │   ├── fundamental_asof.parquet
│   │   ├── industry_asof.parquet
│   │   └── stock_status_panel.parquet
│   │
│   ├── factors/
│   │   ├── factor_panel.parquet
│   │   └── factor_score.parquet
│   │
│   └── signals/
│       ├── rebalance_dates.parquet
│       └── target_weights.parquet
│
├── results/
│   ├── nav.csv
│   ├── trades.csv
│   ├── positions.csv
│   ├── holdings.csv
│   ├── metrics.json
│   ├── factor_ic.csv
│   └── figures/
│       ├── nav_curve.png
│       ├── drawdown.png
│       ├── annual_return.png
│       ├── monthly_heatmap.png
│       └── industry_exposure.png
│
├── notebooks/
│   ├── 01_data_check.ipynb
│   ├── 02_factor_check.ipynb
│   ├── 03_backtest_report.ipynb
│   └── 04_bias_check.ipynb
│
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_build_panel.py
│   ├── 03_calc_factors.py
│   ├── 04_generate_signals.py
│   ├── 05_run_backtest.py
│   └── 06_make_report.py
│
├── src/
│   └── ashare_alpha/
│       ├── __init__.py
│       ├── settings.py
│       ├── logger.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── tushare_client.py
│       │   ├── downloader.py
│       │   ├── calendar.py
│       │   ├── schema.py
│       │   ├── preprocess.py
│       │   ├── asof.py
│       │   └── loader.py
│       │
│       ├── factors/
│       │   ├── __init__.py
│       │   ├── transform.py
│       │   ├── value.py
│       │   ├── quality.py
│       │   ├── growth.py
│       │   ├── volatility.py
│       │   ├── momentum.py
│       │   └── composite.py
│       │
│       ├── strategy/
│       │   ├── __init__.py
│       │   ├── universe.py
│       │   ├── signal.py
│       │   ├── portfolio.py
│       │   └── constraints.py
│       │
│       ├── backtest/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   ├── broker.py
│       │   ├── execution.py
│       │   ├── cost.py
│       │   ├── account.py
│       │   └── recorder.py
│       │
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── attribution.py
│       │   ├── factor_ic.py
│       │   └── risk.py
│       │
│       └── visualize/
│           ├── __init__.py
│           ├── plot_nav.py
│           ├── plot_drawdown.py
│           ├── plot_returns.py
│           ├── plot_exposure.py
│           └── report.py
│
└── tests/
    ├── test_asof.py
    ├── test_factors.py
    ├── test_universe.py
    ├── test_execution.py
    └── test_metrics.py
```

---

# 2. 每个 Python 文件的作用

## 2.1 顶层配置与工具

| 文件                             | 作用                                |
| ------------------------------ | --------------------------------- |
| `src/ashare_alpha/settings.py` | 读取 `config/*.yaml`，统一管理路径、日期、策略参数 |
| `src/ashare_alpha/logger.py`   | 初始化日志，记录下载、回测、异常股票等信息             |

核心接口：

```python
def load_config(config_dir: str = "config") -> dict:
    ...
```

---

## 2.2 数据模块

| 文件                       | 作用                   |
| ------------------------ | -------------------- |
| `data/tushare_client.py` | 初始化 Tushare Pro API  |
| `data/downloader.py`     | 下载原始数据               |
| `data/calendar.py`       | 交易日历、调仓日、上一交易日、下一交易日 |
| `data/schema.py`         | 定义标准字段名，检查数据列        |
| `data/preprocess.py`     | 复权价格、成交额、停牌、涨跌停等预处理  |
| `data/asof.py`           | 财务数据和行业数据的 as-of 处理  |
| `data/loader.py`         | 读取处理好的 panel 数据      |

核心接口：

```python
def download_all(config: dict) -> None:
    ...

def build_price_panel(config: dict) -> pd.DataFrame:
    ...

def build_fundamental_asof(config: dict) -> pd.DataFrame:
    ...

def load_panel(name: str, config: dict) -> pd.DataFrame:
    ...
```

---

## 2.3 因子模块

| 文件                      | 作用              |
| ----------------------- | --------------- |
| `factors/transform.py`  | 去极值、标准化、行业中性化   |
| `factors/value.py`      | PB、PE 价值因子      |
| `factors/quality.py`    | ROE、ROA、现金流质量因子 |
| `factors/growth.py`     | 营收增速、净利润增速      |
| `factors/volatility.py` | 低波动因子           |
| `factors/momentum.py`   | 中期动量、短期反转       |
| `factors/composite.py`  | 合成总分            |

核心接口：

```python
def calc_value_factor(df: pd.DataFrame) -> pd.DataFrame:
    ...

def calc_quality_factor(df: pd.DataFrame) -> pd.DataFrame:
    ...

def calc_growth_factor(df: pd.DataFrame) -> pd.DataFrame:
    ...

def calc_volatility_factor(price: pd.DataFrame) -> pd.DataFrame:
    ...

def calc_momentum_factor(price: pd.DataFrame) -> pd.DataFrame:
    ...

def calc_composite_score(factors: pd.DataFrame, config: dict) -> pd.DataFrame:
    ...
```

---

## 2.4 策略模块

| 文件                        | 作用              |
| ------------------------- | --------------- |
| `strategy/universe.py`    | 股票池过滤           |
| `strategy/signal.py`      | 根据因子分数生成选股信号    |
| `strategy/portfolio.py`   | 目标权重生成          |
| `strategy/constraints.py` | 行业上限、个股上限、成交额约束 |

核心接口：

```python
def build_universe(trade_date: str, data: dict, config: dict) -> pd.DataFrame:
    ...

def generate_signal(score: pd.DataFrame, universe: pd.DataFrame, config: dict) -> pd.DataFrame:
    ...

def generate_target_weights(signal: pd.DataFrame, config: dict) -> pd.DataFrame:
    ...
```

---

## 2.5 回测模块

| 文件                      | 作用                |
| ----------------------- | ----------------- |
| `backtest/engine.py`    | 主回测循环             |
| `backtest/broker.py`    | 模拟撮合、下单、成交        |
| `backtest/execution.py` | 停牌、涨跌停、成交价格、成交量限制 |
| `backtest/cost.py`      | 佣金、印花税、滑点         |
| `backtest/account.py`   | 现金、持仓、市值、净值       |
| `backtest/recorder.py`  | 记录每日净值、交易、持仓      |

核心接口：

```python
class BacktestEngine:
    def run(self) -> dict:
        ...

class Account:
    def update_market_value(self, trade_date: str, price: pd.DataFrame) -> None:
        ...

class Broker:
    def rebalance(self, trade_date: str, target_weights: pd.DataFrame) -> list:
        ...
```

---

## 2.6 绩效分析模块

| 文件                        | 作用                  |
| ------------------------- | ------------------- |
| `analysis/metrics.py`     | 年化收益、夏普、最大回撤、Calmar |
| `analysis/factor_ic.py`   | IC、RankIC、ICIR      |
| `analysis/attribution.py` | 行业暴露、收益归因           |
| `analysis/risk.py`        | 波动、回撤、换手、集中度        |

核心接口：

```python
def calc_performance(nav: pd.DataFrame, benchmark: pd.DataFrame) -> dict:
    ...

def calc_factor_ic(factor: pd.DataFrame, future_return: pd.DataFrame) -> pd.DataFrame:
    ...

def calc_industry_exposure(positions: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    ...
```

---

## 2.7 可视化模块

| 文件                           | 作用                  |
| ---------------------------- | ------------------- |
| `visualize/plot_nav.py`      | 净值曲线                |
| `visualize/plot_drawdown.py` | 回撤曲线                |
| `visualize/plot_returns.py`  | 年度收益、月度收益热力图        |
| `visualize/plot_exposure.py` | 行业暴露、市值暴露           |
| `visualize/report.py`        | 汇总生成图表和 markdown 报告 |

核心接口：

```python
def plot_nav_curve(nav: pd.DataFrame, save_path: str) -> None:
    ...

def plot_drawdown(nav: pd.DataFrame, save_path: str) -> None:
    ...

def generate_report(results_dir: str) -> None:
    ...
```

---

# 3. 数据下载模块设计

## 3.1 下载目标

第一版只下载这些表：

```text
stock_basic
trade_calendar
daily
adj_factor
daily_basic
fina_indicator
industry_member
stk_limit
```

不下载：

```text
资金流
龙虎榜
新闻
研报
社交媒体
分钟线
Tick 数据
```

---

## 3.2 原始数据字段

### `stock_basic.parquet`

```python
[
    "ts_code",
    "symbol",
    "name",
    "area",
    "industry",
    "market",
    "exchange",
    "list_date",
    "delist_date",
    "is_hs"
]
```

---

### `daily.parquet`

```python
[
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount"
]
```

---

### `adj_factor.parquet`

```python
[
    "ts_code",
    "trade_date",
    "adj_factor"
]
```

---

### `daily_basic.parquet`

```python
[
    "ts_code",
    "trade_date",
    "turnover_rate",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv"
]
```

---

### `fina_indicator.parquet`

```python
[
    "ts_code",
    "ann_date",
    "end_date",
    "roe",
    "roe_dt",
    "roa",
    "grossprofit_margin",
    "netprofit_margin",
    "debt_to_assets",
    "ocf_to_or",
    "or_yoy",
    "netprofit_yoy"
]
```

---

### `stk_limit.parquet`

```python
[
    "ts_code",
    "trade_date",
    "up_limit",
    "down_limit"
]
```

---

## 3.3 下载模块结构

`src/ashare_alpha/data/tushare_client.py`

```python
import tushare as ts


def get_tushare_client(token: str):
    ts.set_token(token)
    return ts.pro_api()
```

---

`src/ashare_alpha/data/downloader.py`

```python
from pathlib import Path
import pandas as pd


class TushareDownloader:
    def __init__(self, pro, raw_dir: str):
        self.pro = pro
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def save(self, df: pd.DataFrame, name: str) -> None:
        path = self.raw_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)

    def download_stock_basic(self) -> pd.DataFrame:
        df = self.pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,exchange,list_date,delist_date,is_hs"
        )
        self.save(df, "stock_basic")
        return df

    def download_trade_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = self.pro.trade_cal(
            exchange="SSE",
            start_date=start_date,
            end_date=end_date
        )
        self.save(df, "trade_calendar")
        return df

    def download_daily_by_date(self, trade_date: str) -> pd.DataFrame:
        return self.pro.daily(trade_date=trade_date)

    def download_adj_factor_by_date(self, trade_date: str) -> pd.DataFrame:
        return self.pro.adj_factor(trade_date=trade_date)

    def download_daily_basic_by_date(self, trade_date: str) -> pd.DataFrame:
        return self.pro.daily_basic(trade_date=trade_date)

    def download_stk_limit_by_date(self, trade_date: str) -> pd.DataFrame:
        return self.pro.stk_limit(trade_date=trade_date)

    def download_fina_indicator(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = self.pro.fina_indicator(
            start_date=start_date,
            end_date=end_date,
            fields=(
                "ts_code,ann_date,end_date,roe,roe_dt,roa,"
                "grossprofit_margin,netprofit_margin,debt_to_assets,"
                "ocf_to_or,or_yoy,netprofit_yoy"
            )
        )
        self.save(df, "fina_indicator")
        return df
```

---

## 3.4 日频数据批量下载逻辑

`scripts/01_download_data.py`

```python
from ashare_alpha.settings import load_config
from ashare_alpha.data.tushare_client import get_tushare_client
from ashare_alpha.data.downloader import TushareDownloader


def main():
    config = load_config()
    pro = get_tushare_client(config["data"]["tushare_token"])
    downloader = TushareDownloader(pro, config["data"]["raw_dir"])

    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]

    downloader.download_stock_basic()
    cal = downloader.download_trade_calendar(start_date, end_date)

    trade_dates = cal.loc[cal["is_open"] == 1, "cal_date"].tolist()

    daily_list = []
    adj_list = []
    basic_list = []
    limit_list = []

    for trade_date in trade_dates:
        daily_list.append(downloader.download_daily_by_date(trade_date))
        adj_list.append(downloader.download_adj_factor_by_date(trade_date))
        basic_list.append(downloader.download_daily_basic_by_date(trade_date))
        limit_list.append(downloader.download_stk_limit_by_date(trade_date))

    downloader.save(pd.concat(daily_list), "daily")
    downloader.save(pd.concat(adj_list), "adj_factor")
    downloader.save(pd.concat(basic_list), "daily_basic")
    downloader.save(pd.concat(limit_list), "stk_limit")

    downloader.download_fina_indicator(start_date, end_date)


if __name__ == "__main__":
    main()
```

第一版为了简单可以全量下载。后面再做增量更新。

---

# 4. 因子计算模块设计

## 4.1 因子输入表

因子计算统一使用一个宽表：

```text
factor_input
```

字段：

```python
[
    "ts_code",
    "trade_date",
    "adj_open",
    "adj_close",
    "amount",
    "turnover_rate",
    "pe_ttm",
    "pb",
    "total_mv",
    "circ_mv",
    "roe_dt",
    "roa",
    "ocf_to_or",
    "or_yoy",
    "netprofit_yoy",
    "debt_to_assets",
    "industry_code",
    "industry_name"
]
```

---

## 4.2 预处理：复权价格

`data/preprocess.py`

```python
import pandas as pd


def build_price_panel(daily: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    df = daily.merge(adj, on=["ts_code", "trade_date"], how="left")

    df["adj_open"] = df["open"] * df["adj_factor"]
    df["adj_high"] = df["high"] * df["adj_factor"]
    df["adj_low"] = df["low"] * df["adj_factor"]
    df["adj_close"] = df["close"] * df["adj_factor"]
    df["ret"] = df.groupby("ts_code")["adj_close"].pct_change()

    return df
```

---

## 4.3 财务数据 as-of

重点：防止财报未来函数。

`data/asof.py`

```python
import pandas as pd


def build_fundamental_asof(
    trade_dates: list[str],
    fina: pd.DataFrame
) -> pd.DataFrame:
    fina = fina.copy()
    fina = fina.dropna(subset=["ann_date"])
    fina = fina.sort_values(["ts_code", "ann_date", "end_date"])

    result = []

    for trade_date in trade_dates:
        available = fina[fina["ann_date"] <= trade_date]
        latest = (
            available
            .sort_values(["ts_code", "ann_date", "end_date"])
            .groupby("ts_code")
            .tail(1)
        )
        latest = latest.copy()
        latest["trade_date"] = trade_date
        result.append(latest)

    return pd.concat(result, ignore_index=True)
```

第一版这样写最直观，但速度慢。后续可以改成 `merge_asof`。

---

## 4.4 因子标准化工具

`factors/transform.py`

```python
import numpy as np
import pandas as pd


def winsorize_series(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    low = s.quantile(lower)
    high = s.quantile(upper)
    return s.clip(low, high)


def zscore_series(s: pd.Series) -> pd.Series:
    std = s.std()
    if std == 0 or np.isnan(std):
        return s * 0
    return (s - s.mean()) / std


def industry_neutral_zscore(
    df: pd.DataFrame,
    factor_col: str,
    industry_col: str = "industry_code"
) -> pd.Series:
    return (
        df.groupby(industry_col)[factor_col]
        .transform(lambda x: zscore_series(winsorize_series(x)))
    )
```

---

## 4.5 价值因子

`factors/value.py`

```python
import numpy as np
import pandas as pd
from .transform import industry_neutral_zscore


def calc_value_factor(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["ts_code", "trade_date", "industry_code", "pb", "pe_ttm"]].copy()

    out = out[(out["pb"] > 0) & (out["pe_ttm"] > 0)]

    out["raw_value_pb"] = -np.log(out["pb"])
    out["raw_value_pe"] = -np.log(out["pe_ttm"])

    out["value_pb"] = (
        out.groupby("trade_date", group_keys=False)
        .apply(lambda x: industry_neutral_zscore(x, "raw_value_pb"))
    )

    out["value_pe"] = (
        out.groupby("trade_date", group_keys=False)
        .apply(lambda x: industry_neutral_zscore(x, "raw_value_pe"))
    )

    out["value"] = 0.5 * out["value_pb"] + 0.5 * out["value_pe"]

    return out[["ts_code", "trade_date", "value"]]
```

---

## 4.6 质量因子

`factors/quality.py`

```python
import pandas as pd
from .transform import industry_neutral_zscore


def calc_quality_factor(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        [
            "ts_code",
            "trade_date",
            "industry_code",
            "roe_dt",
            "roa",
            "ocf_to_or"
        ]
    ].copy()

    for col in ["roe_dt", "roa", "ocf_to_or"]:
        out[f"z_{col}"] = (
            out.groupby("trade_date", group_keys=False)
            .apply(lambda x: industry_neutral_zscore(x, col))
        )

    out["quality"] = (
        0.4 * out["z_roe_dt"]
        + 0.3 * out["z_roa"]
        + 0.3 * out["z_ocf_to_or"]
    )

    return out[["ts_code", "trade_date", "quality"]]
```

---

## 4.7 成长因子

`factors/growth.py`

```python
import pandas as pd
from .transform import industry_neutral_zscore


def calc_growth_factor(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        [
            "ts_code",
            "trade_date",
            "industry_code",
            "or_yoy",
            "netprofit_yoy"
        ]
    ].copy()

    for col in ["or_yoy", "netprofit_yoy"]:
        out[f"z_{col}"] = (
            out.groupby("trade_date", group_keys=False)
            .apply(lambda x: industry_neutral_zscore(x, col))
        )

    out["growth"] = 0.5 * out["z_or_yoy"] + 0.5 * out["z_netprofit_yoy"]

    return out[["ts_code", "trade_date", "growth"]]
```

---

## 4.8 低波动因子

`factors/volatility.py`

```python
import pandas as pd
from .transform import industry_neutral_zscore


def calc_lowvol_factor(price: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    out = price[["ts_code", "trade_date", "industry_code", "ret"]].copy()

    out["vol60"] = (
        out.groupby("ts_code")["ret"]
        .rolling(window)
        .std()
        .reset_index(level=0, drop=True)
    )

    out["raw_lowvol"] = -out["vol60"]

    out["lowvol"] = (
        out.groupby("trade_date", group_keys=False)
        .apply(lambda x: industry_neutral_zscore(x, "raw_lowvol"))
    )

    return out[["ts_code", "trade_date", "lowvol"]]
```

---

## 4.9 动量与反转因子

`factors/momentum.py`

```python
import pandas as pd
from .transform import industry_neutral_zscore


def calc_momentum_reversal_factor(price: pd.DataFrame) -> pd.DataFrame:
    out = price[["ts_code", "trade_date", "industry_code", "adj_close"]].copy()

    g = out.groupby("ts_code")["adj_close"]

    out["mom_120_20"] = g.shift(20) / g.shift(120) - 1
    out["ret_20"] = out["adj_close"] / g.shift(20) - 1

    out["raw_momentum"] = out["mom_120_20"]
    out["raw_reversal"] = -out["ret_20"]

    out["momentum"] = (
        out.groupby("trade_date", group_keys=False)
        .apply(lambda x: industry_neutral_zscore(x, "raw_momentum"))
    )

    out["reversal"] = (
        out.groupby("trade_date", group_keys=False)
        .apply(lambda x: industry_neutral_zscore(x, "raw_reversal"))
    )

    return out[["ts_code", "trade_date", "momentum", "reversal"]]
```

---

## 4.10 综合因子

`factors/composite.py`

```python
import pandas as pd


def calc_composite_score(factors: pd.DataFrame, weights: dict) -> pd.DataFrame:
    out = factors.copy()

    out["score"] = 0.0

    for factor_name, weight in weights.items():
        out["score"] += weight * out[factor_name].fillna(0)

    return out[
        [
            "ts_code",
            "trade_date",
            "value",
            "quality",
            "growth",
            "lowvol",
            "momentum",
            "reversal",
            "score"
        ]
    ]
```

---

# 5. 股票池过滤模块设计

## 5.1 MVP 股票池过滤规则

每个调仓信号日执行：

```text
1. 只保留 A 股普通股票
2. 剔除上市不足 250 个交易日
3. 剔除 ST、*ST、退市整理
4. 剔除停牌股票
5. 剔除 PB <= 0
6. 剔除 PE_TTM <= 0
7. 剔除过去 20 日日均成交额 < 3000 万
8. 剔除过去 60 日停牌天数 > 10 天
9. 剔除调仓日开盘涨停无法买入的股票
10. 剔除关键因子缺失过多的股票
```

---

## 5.2 股票池函数设计

`strategy/universe.py`

```python
import pandas as pd


def filter_listed_days(
    df: pd.DataFrame,
    stock_basic: pd.DataFrame,
    trade_date: str,
    min_list_days: int
) -> pd.DataFrame:
    stock_basic = stock_basic.copy()
    stock_basic["list_date"] = stock_basic["list_date"].astype(str)

    listed = stock_basic[stock_basic["list_date"] <= trade_date].copy()

    # 简化处理：用自然日近似，后面可改成交易日数量
    listed["listed_days"] = (
        pd.to_datetime(trade_date) - pd.to_datetime(listed["list_date"])
    ).dt.days

    valid_codes = listed.loc[
        listed["listed_days"] >= min_list_days,
        "ts_code"
    ]

    return df[df["ts_code"].isin(valid_codes)]


def filter_st(df: pd.DataFrame, stock_basic: pd.DataFrame) -> pd.DataFrame:
    names = stock_basic[["ts_code", "name"]].copy()
    df = df.merge(names, on="ts_code", how="left")

    mask = ~df["name"].str.contains("ST|退", na=False)
    return df[mask].drop(columns=["name"])


def filter_liquidity(
    df: pd.DataFrame,
    price_panel: pd.DataFrame,
    trade_date: str,
    min_avg_amount: float,
    window: int = 20
) -> pd.DataFrame:
    hist = price_panel[price_panel["trade_date"] <= trade_date].copy()
    hist = hist.sort_values(["ts_code", "trade_date"])

    hist["avg_amount_20"] = (
        hist.groupby("ts_code")["amount"]
        .rolling(window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    latest = hist.groupby("ts_code").tail(1)
    valid_codes = latest.loc[
        latest["avg_amount_20"] >= min_avg_amount,
        "ts_code"
    ]

    return df[df["ts_code"].isin(valid_codes)]


def filter_valuation_valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["pb"] > 0)
        & (df["pe_ttm"] > 0)
    ]


def build_universe(
    trade_date: str,
    factor_input: pd.DataFrame,
    price_panel: pd.DataFrame,
    stock_basic: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    df = factor_input[factor_input["trade_date"] == trade_date].copy()

    df = filter_listed_days(
        df,
        stock_basic,
        trade_date,
        config["universe"]["min_list_days"]
    )

    df = filter_st(df, stock_basic)

    df = filter_liquidity(
        df,
        price_panel,
        trade_date,
        config["universe"]["min_avg_amount_20"]
    )

    df = filter_valuation_valid(df)

    return df
```

第一版先用股票名称过滤 ST，不完美，但可以跑通。第二版应该接入历史 ST 状态数据。

---

# 6. 回测引擎设计

## 6.1 回测核心流程

时间线必须严格这样：

```text
T 日收盘后：
    计算因子
    生成目标持仓

T+1 日开盘：
    检查停牌、涨跌停、流动性限制
    执行调仓

T+1 到下一个调仓日前：
    持仓不变
    每日根据收盘价更新净值
```

---

## 6.2 回测状态

`backtest/account.py`

```python
class Account:
    def __init__(self, init_cash: float):
        self.cash = init_cash
        self.positions = {}
        self.total_value = init_cash

    def update_market_value(self, trade_date, price_df):
        market_value = 0.0

        for ts_code, shares in self.positions.items():
            row = price_df[
                (price_df["trade_date"] == trade_date)
                & (price_df["ts_code"] == ts_code)
            ]

            if row.empty:
                continue

            close_price = row.iloc[0]["adj_close"]
            market_value += shares * close_price

        self.total_value = self.cash + market_value
        return self.total_value
```

---

## 6.3 交易成本

`backtest/cost.py`

```python
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
```

---

## 6.4 成交规则

`backtest/execution.py`

```python
import numpy as np
import pandas as pd


def is_suspended(row: pd.Series) -> bool:
    return pd.isna(row["open"]) or pd.isna(row["close"]) or row["vol"] == 0


def is_limit_up(row: pd.Series) -> bool:
    if pd.isna(row.get("up_limit")):
        return False
    return row["open"] >= row["up_limit"]


def is_limit_down(row: pd.Series) -> bool:
    if pd.isna(row.get("down_limit")):
        return False
    return row["open"] <= row["down_limit"]


def can_buy(row: pd.Series) -> bool:
    if is_suspended(row):
        return False
    if is_limit_up(row):
        return False
    return True


def can_sell(row: pd.Series) -> bool:
    if is_suspended(row):
        return False
    if is_limit_down(row):
        return False
    return True


def round_lot_shares(shares: float, lot_size: int = 100) -> int:
    return int(shares // lot_size * lot_size)
```

---

## 6.5 Broker 撮合模块

`backtest/broker.py`

```python
from .cost import calc_buy_cost, calc_sell_cost
from .execution import can_buy, can_sell, round_lot_shares


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
            shares = round_lot_shares(self.account.cash / (price * 1.01))
            amount = shares * price
            cost = calc_buy_cost(amount, self.cost_config)
            total_cash_need = amount + cost

        if shares <= 0:
            return None

        self.account.cash -= total_cash_need
        self.account.positions[ts_code] = self.account.positions.get(ts_code, 0) + shares

        return {
            "trade_date": trade_date,
            "ts_code": ts_code,
            "side": "BUY",
            "price": price,
            "shares": shares,
            "amount": amount,
            "cost": cost
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
            "cost": cost
        }
```

---

## 6.6 回测主引擎

`backtest/engine.py`

```python
import pandas as pd
from .account import Account
from .broker import Broker


class BacktestEngine:
    def __init__(
        self,
        price_panel: pd.DataFrame,
        target_weights: pd.DataFrame,
        trade_dates: list[str],
        rebalance_dates: list[str],
        init_cash: float,
        cost_config: dict
    ):
        self.price_panel = price_panel
        self.target_weights = target_weights
        self.trade_dates = trade_dates
        self.rebalance_dates = set(rebalance_dates)
        self.account = Account(init_cash)
        self.broker = Broker(self.account, cost_config)

        self.nav_records = []
        self.trade_records = []
        self.position_records = []

    def run(self):
        for trade_date in self.trade_dates:
            today_price = self.price_panel[
                self.price_panel["trade_date"] == trade_date
            ]

            if trade_date in self.rebalance_dates:
                self.rebalance(trade_date, today_price)

            total_value = self.account.update_market_value(trade_date, self.price_panel)

            self.nav_records.append({
                "trade_date": trade_date,
                "total_value": total_value,
                "cash": self.account.cash
            })

            self.record_positions(trade_date, today_price)

        return {
            "nav": pd.DataFrame(self.nav_records),
            "trades": pd.DataFrame(self.trade_records),
            "positions": pd.DataFrame(self.position_records)
        }

    def rebalance(self, trade_date: str, today_price: pd.DataFrame):
        target = self.target_weights[
            self.target_weights["rebalance_date"] == trade_date
        ].copy()

        if target.empty:
            return

        total_value = self.account.update_market_value(trade_date, self.price_panel)

        target_amount = {
            row["ts_code"]: row["target_weight"] * total_value
            for _, row in target.iterrows()
        }

        current_codes = set(self.account.positions.keys())
        target_codes = set(target_amount.keys())

        # 先卖出不在目标池里的股票
        for ts_code in list(current_codes - target_codes):
            row = today_price[today_price["ts_code"] == ts_code]
            if row.empty:
                continue

            shares = self.account.positions[ts_code]
            trade = self.broker.sell(
                trade_date,
                ts_code,
                shares,
                row.iloc[0]
            )

            if trade:
                self.trade_records.append(trade)

        # 再调整目标池中的股票
        for ts_code, amount in target_amount.items():
            row = today_price[today_price["ts_code"] == ts_code]
            if row.empty:
                continue

            price = row.iloc[0]["adj_open"]
            current_shares = self.account.positions.get(ts_code, 0)
            current_amount = current_shares * price

            diff_amount = amount - current_amount

            if diff_amount > 0:
                trade = self.broker.buy(
                    trade_date,
                    ts_code,
                    diff_amount,
                    row.iloc[0]
                )
            else:
                sell_shares = min(current_shares, int(abs(diff_amount) / price))
                trade = self.broker.sell(
                    trade_date,
                    ts_code,
                    sell_shares,
                    row.iloc[0]
                )

            if trade:
                self.trade_records.append(trade)

    def record_positions(self, trade_date: str, today_price: pd.DataFrame):
        for ts_code, shares in self.account.positions.items():
            row = today_price[today_price["ts_code"] == ts_code]
            if row.empty:
                continue

            price = row.iloc[0]["adj_close"]

            self.position_records.append({
                "trade_date": trade_date,
                "ts_code": ts_code,
                "shares": shares,
                "close_price": price,
                "market_value": shares * price
            })
```

---

# 7. 绩效分析模块设计

## 7.1 净值处理

`analysis/metrics.py`

```python
import numpy as np
import pandas as pd


def prepare_nav(nav: pd.DataFrame) -> pd.DataFrame:
    nav = nav.copy()
    nav = nav.sort_values("trade_date")
    nav["nav"] = nav["total_value"] / nav["total_value"].iloc[0]
    nav["daily_ret"] = nav["nav"].pct_change().fillna(0)
    return nav
```

---

## 7.2 年化收益

```python
def calc_annual_return(nav: pd.DataFrame) -> float:
    n = len(nav)
    total_return = nav["nav"].iloc[-1] / nav["nav"].iloc[0] - 1
    return (1 + total_return) ** (252 / n) - 1
```

---

## 7.3 年化波动

```python
def calc_annual_vol(nav: pd.DataFrame) -> float:
    return nav["daily_ret"].std() * np.sqrt(252)
```

---

## 7.4 夏普比率

```python
def calc_sharpe(nav: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
    ann_ret = calc_annual_return(nav)
    ann_vol = calc_annual_vol(nav)

    if ann_vol == 0:
        return np.nan

    return (ann_ret - risk_free_rate) / ann_vol
```

---

## 7.5 最大回撤

```python
def calc_max_drawdown(nav: pd.DataFrame) -> float:
    cummax = nav["nav"].cummax()
    drawdown = nav["nav"] / cummax - 1
    return drawdown.min()
```

---

## 7.6 Calmar

```python
def calc_calmar(nav: pd.DataFrame) -> float:
    ann_ret = calc_annual_return(nav)
    max_dd = abs(calc_max_drawdown(nav))

    if max_dd == 0:
        return np.nan

    return ann_ret / max_dd
```

---

## 7.7 汇总指标

```python
def calc_performance(nav: pd.DataFrame) -> dict:
    nav = prepare_nav(nav)

    return {
        "annual_return": calc_annual_return(nav),
        "annual_vol": calc_annual_vol(nav),
        "sharpe": calc_sharpe(nav),
        "max_drawdown": calc_max_drawdown(nav),
        "calmar": calc_calmar(nav),
        "total_return": nav["nav"].iloc[-1] - 1,
        "final_nav": nav["nav"].iloc[-1]
    }
```

---

## 7.8 因子 IC

`analysis/factor_ic.py`

```python
import pandas as pd


def calc_forward_return(
    price: pd.DataFrame,
    horizon: int = 20
) -> pd.DataFrame:
    df = price[["ts_code", "trade_date", "adj_close"]].copy()
    df = df.sort_values(["ts_code", "trade_date"])

    df["future_close"] = (
        df.groupby("ts_code")["adj_close"]
        .shift(-horizon)
    )

    df["future_return"] = df["future_close"] / df["adj_close"] - 1

    return df[["ts_code", "trade_date", "future_return"]]


def calc_rank_ic(
    factor: pd.DataFrame,
    price: pd.DataFrame,
    factor_col: str = "score",
    horizon: int = 20
) -> pd.DataFrame:
    future_ret = calc_forward_return(price, horizon)

    df = factor.merge(
        future_ret,
        on=["ts_code", "trade_date"],
        how="inner"
    )

    records = []

    for trade_date, g in df.groupby("trade_date"):
        if len(g) < 20:
            continue

        ic = g[factor_col].corr(g["future_return"], method="spearman")

        records.append({
            "trade_date": trade_date,
            "rank_ic": ic
        })

    return pd.DataFrame(records)


def calc_icir(ic_df: pd.DataFrame) -> float:
    mean_ic = ic_df["rank_ic"].mean()
    std_ic = ic_df["rank_ic"].std()

    if std_ic == 0:
        return 0

    return mean_ic / std_ic
```

---

# 8. 可视化模块设计

## 8.1 净值曲线

`visualize/plot_nav.py`

```python
import matplotlib.pyplot as plt


def plot_nav_curve(nav, save_path):
    nav = nav.copy()
    nav["nav"] = nav["total_value"] / nav["total_value"].iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(nav["trade_date"], nav["nav"])
    plt.title("Strategy NAV")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```

---

## 8.2 回撤曲线

`visualize/plot_drawdown.py`

```python
import matplotlib.pyplot as plt


def plot_drawdown(nav, save_path):
    nav = nav.copy()
    nav["nav"] = nav["total_value"] / nav["total_value"].iloc[0]
    nav["cummax"] = nav["nav"].cummax()
    nav["drawdown"] = nav["nav"] / nav["cummax"] - 1

    plt.figure(figsize=(12, 6))
    plt.plot(nav["trade_date"], nav["drawdown"])
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```

---

## 8.3 年度收益柱状图

`visualize/plot_returns.py`

```python
import matplotlib.pyplot as plt
import pandas as pd


def calc_annual_returns(nav):
    nav = nav.copy()
    nav["trade_date"] = pd.to_datetime(nav["trade_date"])
    nav["nav"] = nav["total_value"] / nav["total_value"].iloc[0]
    nav["year"] = nav["trade_date"].dt.year

    annual = nav.groupby("year")["nav"].agg(["first", "last"])
    annual["return"] = annual["last"] / annual["first"] - 1

    return annual.reset_index()


def plot_annual_return(nav, save_path):
    annual = calc_annual_returns(nav)

    plt.figure(figsize=(10, 5))
    plt.bar(annual["year"].astype(str), annual["return"])
    plt.title("Annual Return")
    plt.xlabel("Year")
    plt.ylabel("Return")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```

---

## 8.4 行业暴露

`visualize/plot_exposure.py`

```python
import matplotlib.pyplot as plt
import pandas as pd


def calc_industry_exposure(positions, industry):
    df = positions.merge(
        industry[["ts_code", "trade_date", "industry_name"]],
        on=["ts_code", "trade_date"],
        how="left"
    )

    total_mv = df.groupby("trade_date")["market_value"].transform("sum")
    df["weight"] = df["market_value"] / total_mv

    exposure = (
        df.groupby(["trade_date", "industry_name"])["weight"]
        .sum()
        .reset_index()
    )

    return exposure


def plot_industry_exposure(exposure, save_path):
    pivot = exposure.pivot(
        index="trade_date",
        columns="industry_name",
        values="weight"
    ).fillna(0)

    plt.figure(figsize=(14, 7))
    pivot.plot.area(figsize=(14, 7))
    plt.title("Industry Exposure")
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```

---

# 9. 配置文件格式

## 9.1 `config/data.yaml`

```yaml
data:
  tushare_token: "YOUR_TUSHARE_TOKEN"

  start_date: "20150101"
  end_date: "20251231"

  raw_dir: "data/raw"
  processed_dir: "data/processed"
  factor_dir: "data/factors"
  signal_dir: "data/signals"

  benchmark_code: "000905.SH"

  tables:
    stock_basic: "stock_basic.parquet"
    trade_calendar: "trade_calendar.parquet"
    daily: "daily.parquet"
    adj_factor: "adj_factor.parquet"
    daily_basic: "daily_basic.parquet"
    fina_indicator: "fina_indicator.parquet"
    industry_member: "industry_member.parquet"
    stk_limit: "stk_limit.parquet"
```

---

## 9.2 `config/strategy.yaml`

```yaml
strategy:
  name: "quality_value_lowvol_mvp"

  rebalance:
    frequency: "monthly"
    day: "first_trading_day"
    signal_lag_days: 1

  universe:
    min_list_days: 250
    min_avg_amount_20: 30000000
    max_suspend_days_60: 10
    exclude_st: true
    exclude_new_stock: true
    exclude_negative_pe: true
    exclude_negative_pb: true
    exclude_bj: true

  factors:
    weights:
      value: 0.25
      quality: 0.20
      growth: 0.15
      lowvol: 0.15
      momentum: 0.10
      reversal: 0.10
      leverage: 0.05

    winsorize:
      lower: 0.01
      upper: 0.99

    standardize:
      method: "industry_zscore"

  portfolio:
    top_n: 50
    weighting: "equal_weight"
    max_stock_weight: 0.02
    max_industry_weight: 0.20
    min_stock_weight: 0.005
```

---

## 9.3 `config/cost.yaml`

```yaml
cost:
  commission_rate: 0.0003
  stamp_tax_rate: 0.0005
  exchange_fee_rate: 0.0000341
  slippage_rate: 0.0005

  lot_size: 100

  liquidity:
    max_trade_pct_of_avg_amount_20: 0.05
```

---

## 9.4 `config/backtest.yaml`

```yaml
backtest:
  init_cash: 10000000
  start_date: "20150101"
  end_date: "20251231"

  price_type:
    buy: "adj_open"
    sell: "adj_open"
    mark_to_market: "adj_close"

  execution:
    forbid_buy_limit_up: true
    forbid_sell_limit_down: true
    forbid_trade_suspended: true
    use_lot_size: true
    allow_partial_fill: true

  output:
    result_dir: "results"
    save_trades: true
    save_positions: true
    save_nav: true
```

---

# 10. 第一版代码实现顺序

## 第 0 步：初始化项目

先建目录：

```bash
mkdir a-share-alpha-mvp
cd a-share-alpha-mvp

mkdir -p config data/raw data/processed data/factors data/signals
mkdir -p results/figures notebooks scripts tests
mkdir -p src/ashare_alpha
```

安装依赖：

```text
pandas
numpy
pyarrow
matplotlib
pyyaml
tqdm
tushare
scipy
```

`requirements.txt`：

```txt
pandas
numpy
pyarrow
matplotlib
pyyaml
tqdm
tushare
scipy
```

---

## 第 1 步：配置读取

先实现：

```text
src/ashare_alpha/settings.py
```

代码：

```python
from pathlib import Path
import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_dir: str = "config") -> dict:
    config_dir = Path(config_dir)

    config = {}

    for name in ["data", "strategy", "cost", "backtest"]:
        path = config_dir / f"{name}.yaml"
        config.update(load_yaml(path))

    return config
```

验证：

```bash
python -c "from ashare_alpha.settings import load_config; print(load_config())"
```

---

## 第 2 步：下载数据

实现：

```text
src/ashare_alpha/data/tushare_client.py
src/ashare_alpha/data/downloader.py
scripts/01_download_data.py
```

先只下载：

```text
stock_basic
trade_calendar
daily
adj_factor
daily_basic
fina_indicator
stk_limit
```

第一版可以先只跑 2020 至今，减少调试成本。

---

## 第 3 步：构建价格面板

实现：

```text
src/ashare_alpha/data/preprocess.py
scripts/02_build_panel.py
```

输入：

```text
daily.parquet
adj_factor.parquet
daily_basic.parquet
stk_limit.parquet
```

输出：

```text
price_panel.parquet
valuation_panel.parquet
```

核心字段：

```python
[
    "ts_code",
    "trade_date",
    "open",
    "close",
    "adj_open",
    "adj_close",
    "ret",
    "amount",
    "vol",
    "pe_ttm",
    "pb",
    "total_mv",
    "circ_mv",
    "up_limit",
    "down_limit"
]
```

---

## 第 4 步：财务数据 as-of

实现：

```text
src/ashare_alpha/data/asof.py
```

输入：

```text
fina_indicator.parquet
trade_calendar.parquet
```

输出：

```text
fundamental_asof.parquet
```

重点测试：

```text
任意 trade_date 的财务数据 ann_date 必须 <= trade_date
```

写测试：

```python
def test_no_financial_future_leakage():
    df = pd.read_parquet("data/processed/fundamental_asof.parquet")
    assert (df["ann_date"] <= df["trade_date"]).all()
```

---

## 第 5 步：计算因子

实现：

```text
src/ashare_alpha/factors/
scripts/03_calc_factors.py
```

输出：

```text
data/factors/factor_panel.parquet
data/factors/factor_score.parquet
```

第一版只做 6 个因子：

```text
value
quality
growth
lowvol
momentum
reversal
```

暂时不做：

```text
机器学习
动态权重
IC 滚动加权
组合优化
```

---

## 第 6 步：股票池过滤与信号生成

实现：

```text
src/ashare_alpha/strategy/universe.py
src/ashare_alpha/strategy/signal.py
src/ashare_alpha/strategy/portfolio.py
scripts/04_generate_signals.py
```

输出：

```text
data/signals/target_weights.parquet
```

字段：

```python
[
    "signal_date",
    "rebalance_date",
    "ts_code",
    "score",
    "rank",
    "target_weight",
    "industry_code",
    "industry_name"
]
```

MVP 规则：

```python
Top 50
等权
单票 2%
行业 20%
月频调仓
```

---

## 第 7 步：实现回测引擎

实现：

```text
src/ashare_alpha/backtest/
scripts/05_run_backtest.py
```

输出：

```text
results/nav.csv
results/trades.csv
results/positions.csv
```

第一版必须支持：

```text
开盘价成交
100 股整数倍
买入扣佣金 + 滑点
卖出扣佣金 + 印花税 + 滑点
停牌不可交易
涨停不可买
跌停不可卖
现金不足时缩小买入数量
```

---

## 第 8 步：绩效分析

实现：

```text
src/ashare_alpha/analysis/metrics.py
src/ashare_alpha/analysis/factor_ic.py
```

输出：

```text
results/metrics.json
results/factor_ic.csv
```

指标：

```text
年化收益
年化波动
夏普
最大回撤
Calmar
总收益
月度胜率
年度收益
换手率
RankIC
ICIR
```

---

## 第 9 步：可视化与报告

实现：

```text
src/ashare_alpha/visualize/
scripts/06_make_report.py
```

输出：

```text
results/figures/nav_curve.png
results/figures/drawdown.png
results/figures/annual_return.png
results/figures/monthly_heatmap.png
results/figures/industry_exposure.png
report/mvp_report.md
```

---

# 11. MVP 主流程命令

最终应该能这样跑：

```bash
python scripts/01_download_data.py
python scripts/02_build_panel.py
python scripts/03_calc_factors.py
python scripts/04_generate_signals.py
python scripts/05_run_backtest.py
python scripts/06_make_report.py
```

或者后面做一个统一入口：

```bash
python -m ashare_alpha.run_all
```

---

# 12. 最小可行版本和升级版本的边界

## MVP 必须完成

```text
数据下载
复权价格
财务 as-of
6 个基础因子
行业内标准化
股票池过滤
Top 50 等权
月频调仓
交易成本
涨跌停/停牌处理
净值曲线
绩效指标
RankIC
```

## MVP 暂时不做

```text
LightGBM
深度学习
分钟线择时
资金流因子
新闻情绪
研报文本
组合优化器
动态因子权重
复杂风险模型
```

## 第二版再做

```text
1. 历史 ST 状态更精确处理
2. 退市股票历史样本修正
3. 行业分类 as-of 更严格
4. benchmark 超额收益
5. 因子分组收益
6. 因子相关性矩阵
7. 滚动 IC 加权
8. 中证 500 / 中证 1000 风格归因
9. 成本压力测试
10. 参数样本外验证
```

---

# 13. 第一版开发优先级

最推荐你按这个顺序写：

```text
1. settings.py
2. tushare_client.py
3. downloader.py
4. preprocess.py
5. asof.py
6. transform.py
7. value.py
8. quality.py
9. growth.py
10. volatility.py
11. momentum.py
12. composite.py
13. universe.py
14. signal.py
15. portfolio.py
16. cost.py
17. execution.py
18. account.py
19. broker.py
20. engine.py
21. metrics.py
22. factor_ic.py
23. visualize/*.py
24. scripts/*.py
25. tests/*.py
```

最关键的是先跑通这个闭环：

```text
price_panel
+ fundamental_asof
+ factor_score
+ target_weights
+ backtest nav
```

只要这个闭环跑通，后面不管加 LightGBM、IC 加权、行业轮动，都是在同一个工程框架上升级。
