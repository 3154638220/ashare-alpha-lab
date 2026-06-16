# A 股 Alpha Lab 当前模型评估与升级计划

> 审计日期：2026-06-15
> 当前模型：`quality_value_lowvol_mvp`
> 当前定位：已跑通的 A 股多因子研究 MVP，不是可直接实盘化的成熟 Alpha 模型。
> 本文目标：基于当前代码、配置、测试和结果产物，判断模型真实能力与短板，并给出下一阶段可执行的改进计划。

---

## 1. 总体判断

当前项目已经完成了从数据、因子、股票池、信号、组合、回测到报告的研究闭环。工程上已经不是空壳，已有可复现产物：

- 数据区间：2018-01-02 至 2025-12-31。
- 股票样本：500 只股票。
- 价格面板：867,728 行，1,943 个交易日。
- 因子分数：867,728 行。
- 调仓信号：95 个调仓日，每期 50 只股票。
- 测试结果：`18 passed`。
- 策略净值：最终净值 2.2333，总收益 123.33%，年化收益 10.99%。
- 基准比较：相对中证 500 和中证 1000 均有明显超额。

但这个结果需要谨慎看待。当前数据下载逻辑使用 `benchmark_code = 000905.SH` 的指数成分股作为股票基础池，即当前样本不是全 A，也不是严格历史成分股。若用当前成分股回填 2018 年以来历史，可能存在明显的幸存者偏差和成分股前视偏差。因此，当前模型更适合被评为：

```text
研究闭环等级：L1.5

L0：只有想法和草稿
L1：可以跑通 MVP
L2：偏差控制基本可信
L3：稳定可诊断的研究模型
L4：可小资金模拟或实盘前验证

当前处于 L1 和 L2 之间：闭环完整，结果有吸引力，但偏差控制和模型诊断还不够。
```

下一阶段的核心目标不是盲目堆更多因子或上机器学习，而是先把研究可信度从“能跑出漂亮结果”提升到“能解释、能复验、能知道结果为什么有效或无效”。

---

## 2. 当前模型能力画像

### 2.1 模型结构

当前策略是一个月频多因子选股模型：

```text
AkShare 数据
→ 价格 / 估值 / 财务 / 行业 as-of 面板
→ 价值、质量、成长、低波、动量、反转因子
→ 行业内 winsorize + zscore
→ 固定权重综合打分
→ 每月首个交易日选 Top 50
→ 下一交易日开盘调仓
→ 等权持有
→ 加入佣金、印花税、交易所费用、滑点、涨跌停、停牌和整手约束
→ 生成净值、交易、持仓、绩效、IC 和图表报告
```

主要配置：

```yaml
strategy:
  rebalance:
    frequency: monthly
    day: first_trading_day
    signal_lag_days: 1

  portfolio:
    top_n: 50
    weighting: equal_weight
    max_stock_weight: 0.02
    max_industry_weight: 0.20
    min_stock_weight: 0.005
```

当前因子权重：

```yaml
value: 0.25
quality: 0.20
growth: 0.15
lowvol: 0.15
momentum: 0.10
reversal: 0.10
leverage: 0.05
```

注意：代码里目前没有实际生成 `leverage` 因子，综合打分函数会忽略不存在的列，因此实际参与打分的权重和为 0.95。排名层面这不一定改变排序，但这是配置与实现不一致，必须修正。

### 2.2 已具备的能力

当前模型已经具备以下能力：

- 能从 AkShare 下载指数成分、日线、估值、财务、行业、涨跌停和指数基准数据。
- 能构建复权价格面板和财务 as-of 面板。
- 能计算传统可解释风格因子。
- 能做行业内横截面标准化。
- 能按月生成 Top 50 股票组合。
- 能执行带 A 股基础交易限制的回测。
- 能输出净值、交易、持仓、超额净值、绩效指标、因子 IC 和图表。
- 已有基础单元测试覆盖 as-of、因子变换、股票池过滤、交易执行和绩效计算。

这是一个合格的多因子研究 MVP。

### 2.3 当前能力边界

当前模型还不能证明自己是稳定 Alpha，原因如下：

- 样本不是全 A，也不是严格历史指数成分股。
- 行业约束配置存在，但 `target_weights` 中没有 `industry_code`，实际很可能没有生效。
- `max_suspend_days_60`、`exclude_bj` 等股票池配置还没有实现。
- 固定因子权重来自人工设定，没有经过滚动验证或样本外选择。
- 动量因子当前 RankIC 为负，但配置仍给了正权重。
- 缺失值在综合分里被填成 0，容易把“缺失信息”和“中性信息”混在一起。
- 因子诊断只有 IC 摘要，缺少分组收益、单调性、分年稳定性、换手、覆盖率和衰减分析。
- 组合构建是简单 Top 50 等权，缺少行业、市值、流动性、换手和风险暴露的系统控制。
- 回测还没有成交量参与率、部分成交、退市处理、真实历史 ST 状态、历史成分股变化等更严格约束。

---

## 3. 当前结果评估

### 3.1 绩效表现

现有 `results/metrics.json` 显示：

| 指标 | 当前值 | 解读 |
| --- | ---: | --- |
| 年化收益 | 10.99% | 绝对收益尚可 |
| 年化波动 | 20.82% | 股票多头组合正常偏高 |
| 夏普比率 | 0.43 | 有收益，但风险调整后一般 |
| 最大回撤 | -33.19% | 回撤较大，不适合直接实盘 |
| Calmar | 0.33 | 回撤效率偏弱 |
| 总收益 | 123.33% | 回测区间内翻倍 |
| 月度胜率 | 50.00% | 胜率中性，主要靠盈亏幅度 |
| 平均换手 | 37.36% | 月频 Top 50 下偏高但可接受 |

分年收益：

| 年份 | 策略收益 |
| --- | ---: |
| 2018 | -30.70% |
| 2019 | 35.63% |
| 2020 | 29.91% |
| 2021 | 22.04% |
| 2022 | -6.01% |
| 2023 | 1.79% |
| 2024 | 26.37% |
| 2025 | 22.50% |

这个收益曲线说明模型有一定风格有效性，尤其在 2019、2020、2021、2024、2025 表现较好；但 2018 回撤明显，2023 基本停滞，需要做市场环境归因。

### 3.2 相对基准表现

相对中证 500：

| 指标 | 当前值 |
| --- | ---: |
| 基准总收益 | 17.90% |
| 策略相对总收益差 | 105.43% |
| 超额净值收益 | 89.43% |
| 年化超额收益 | 8.64% |
| 跟踪误差 | 11.69% |
| 信息比率 | 0.68 |
| 最大超额回撤 | -14.44% |

相对中证 1000：

| 指标 | 当前值 |
| --- | ---: |
| 基准总收益 | 7.10% |
| 策略相对总收益差 | 116.24% |
| 超额净值收益 | 108.54% |
| 年化超额收益 | 10.01% |
| 跟踪误差 | 15.00% |
| 信息比率 | 0.58 |
| 最大超额回撤 | -18.32% |

相对表现看起来不错，但必须先修正样本偏差再判断可信度。当前股票基础池来自指数成分下载逻辑，如果成分股是下载时点的静态列表，那么历史回测等于在过去使用了未来才知道的成分股名单。

### 3.3 因子 IC 表现

现有 `results/factor_ic.json`：

| 因子 | Mean RankIC | ICIR | 正 IC 比例 | 当前判断 |
| --- | ---: | ---: | ---: | --- |
| value | 0.0255 | 0.1170 | 52.06% | 有弱正贡献 |
| quality | 0.0035 | 0.0263 | 48.20% | 基本无效 |
| growth | 0.0104 | 0.0871 | 51.34% | 弱正贡献 |
| lowvol | 0.0461 | 0.2221 | 55.61% | 当前较有效 |
| momentum | -0.0126 | -0.0778 | 46.09% | 当前方向可能错误 |
| reversal | 0.0442 | 0.2877 | 58.60% | 当前最强之一 |

最重要的结论：

- 当前综合分大概率主要来自 `value`、`lowvol`、`reversal`。
- `quality` 权重 0.20，但 IC 几乎为 0，应该重新评估。
- `momentum` 的平均 RankIC 为负，继续给正权重会拖累模型。
- `reversal` 和 `lowvol` 是当前最值得优先保留和扩展诊断的因子。

### 3.4 因子相关性

现有因子 Spearman 相关性画像显示：

- `value` 与 `lowvol` 相关性约 0.497。
- `value` 与综合 `score` 相关性约 0.714。
- `lowvol` 与综合 `score` 相关性约 0.536。
- `quality` 与综合 `score` 相关性约 0.452。
- `momentum` 与综合 `score` 相关性约 0.098。

这说明综合分并不是均衡风格模型，而是偏价值、低波，同时受质量权重影响较多。由于质量当前 IC 很弱，综合分里可能存在无效噪声。

---

## 4. 关键问题清单

### P0：必须优先修复的问题

1. 样本池存在潜在前视偏差
   当前 AkShare 下载逻辑以 `000905.SH` 指数成分作为基础股票池，且样本正好 500 只。必须确认这些成分是否是历史成分；如果不是，当前历史回测存在严重成分股前视。

2. 行业约束没有真正落地
   `target_weights.parquet` 当前列为：

   ```text
   ts_code, trade_date, value, quality, growth, lowvol, momentum,
   reversal, score, rank, target_weight, rebalance_date, execution_date
   ```

   缺少 `industry_code` 和 `industry_name`，因此 `apply_position_constraints` 中的行业上限逻辑没有输入字段。

3. 配置与实现不一致
   `strategy.yaml` 中配置了 `leverage: 0.05`，但没有对应因子实现。必须删除该配置、实现该因子，或在综合打分前对实际可用因子权重重新归一化。

4. 股票池配置没有全部实现
   已配置但未实现或未严格实现：

   - `max_suspend_days_60`
   - `exclude_bj`
   - 历史 ST 状态
   - 退市股票处理

5. 因子权重没有根据有效性更新
   `momentum` 当前 RankIC 为负，但仍给正权重。`quality` 几乎无效但给 0.20 权重。

### P1：影响研究可信度的问题

1. 缺少分组收益和单调性检验。
2. 缺少分年、分市场状态、分行业、分市值段 IC 稳定性。
3. 缺少因子覆盖率和缺失值影响分析。
4. 缺少因子换手和组合换手归因。
5. 缺少交易成本压力测试。
6. 缺少参数敏感性分析，例如 Top N、调仓频率、因子窗口、权重变化。
7. 缺少样本外 walk-forward 验证。

### P2：模型增强问题

1. 固定权重过于粗糙。
2. 没有风险模型或风格暴露控制。
3. 没有组合优化目标函数。
4. 没有动态因子权重。
5. 没有机器学习排序模型。
6. 没有真实实盘约束模拟，例如成交量参与率和更细的滑点模型。

---

## 5. 改进总原则

后续升级遵循以下顺序：

```text
先修偏差
→ 再做诊断
→ 再调因子权重
→ 再升级组合构建
→ 最后考虑机器学习
```

不要在样本偏差、行业约束和因子诊断未修复前直接上 LightGBM 或复杂优化器。否则模型可能只是更复杂地过拟合。

---

## 6. 分阶段升级路线

## Phase 0：冻结当前基线

目标：把当前结果固定为 `baseline_v0`，作为后续所有改动的比较对象。

要做的事：

1. 记录当前配置快照：

   - `config/data.yaml`
   - `config/strategy.yaml`
   - `config/cost.yaml`
   - `config/backtest.yaml`

2. 记录当前产物摘要：

   - `results/metrics.json`
   - `results/factor_ic.json`
   - `results/nav.csv`
   - `results/excess_nav.csv`

3. 增加一个实验 manifest：

   ```text
   results/experiment_manifest.json
   ```

   建议字段：

   ```json
   {
     "experiment_id": "baseline_v0",
     "created_at": "2026-06-15",
     "data_start": "20180101",
     "data_end": "20251231",
     "universe": "current_csi500_constituents",
     "strategy": "quality_value_lowvol_mvp",
     "notes": "Known risk: possible constituent look-ahead bias"
   }
   ```

验收标准：

- 能一键重新生成当前结果。
- `pytest -q` 通过。
- 重新生成的关键指标与当前结果误差可解释。

---

## Phase 1：修正数据与股票池偏差

目标：让回测样本从“方便可跑”变成“研究可信”。

### 1.1 明确股票池定义

提供至少三种可切换 universe：

```yaml
universe_mode:
  - csi500_current_constituents
  - csi500_historical_constituents
  - all_a
```

优先级：

1. `all_a`：全 A 股票池，最适合做普适 Alpha 研究。
2. `csi500_historical_constituents`：用于相对中证 500 的约束组合。
3. `csi500_current_constituents`：只保留为调试模式，不作为正式回测结论。

代码切入点：

- `src/ashare_alpha/data/akshare_downloader.py`
- `scripts/01_download_data.py`
- `src/ashare_alpha/strategy/universe.py`
- `config/data.yaml`

验收标准：

- `stock_basic.parquet` 可以覆盖全 A 或明确历史成分。
- 任意回测日期只使用当日已知的股票池信息。
- 报告中明确展示 universe 类型和股票数量。

### 1.2 历史 ST、退市和上市状态

当前 `filter_st` 使用当前名称过滤 `ST|退`，这对历史回测不够严谨。

需要改成：

- 下载或维护历史 ST 状态。
- 对退市股票保留历史数据，按真实可交易日期处理。
- `list_date <= trade_date` 已有，但需要补充 `delist_date` 逻辑。

验收标准：

- 对任意 `trade_date`，股票状态来自该日或之前可获得信息。
- 测试覆盖 ST、退市、上市不足 250 天三类边界。

### 1.3 行业 as-of 修复

当前 `build_industry_asof` 使用 `in_date <= trade_date`，但没有处理 `out_date`。

需要改成：

```text
in_date <= trade_date
and (out_date is null or out_date > trade_date)
```

验收标准：

- 任意股票在同一交易日最多一个行业。
- `out_date` 后不再沿用旧行业。
- `target_weights` 必须包含 `industry_code` 和 `industry_name`。

### 1.4 股票池过滤补全

实现配置中已经存在但未落地的过滤：

- `exclude_bj`
- `max_suspend_days_60`
- `exclude_new_stock`
- 历史 ST
- 估值合法性
- 20 日成交额

验收标准：

- 每个过滤器都有独立测试。
- 每个调仓日输出过滤前后股票数量。
- 报告中展示各过滤器剔除数量。

---

## Phase 2：因子诊断体系

目标：知道每个因子为什么有效、何时失效、是否值得进入综合模型。

### 2.1 单因子报告

每个因子生成以下诊断：

- RankIC 时间序列。
- Mean IC、ICIR、正 IC 比例。
- 分年 IC。
- 分月 IC 热力图。
- 分行业 IC。
- 分市值段 IC。
- 分组收益。
- 多空收益。
- 分组单调性。
- 因子覆盖率。
- 因子换手率。
- 因子衰减：5、10、20、40、60 日 horizon。

新增输出：

```text
results/factor_report/
  value.md
  quality.md
  growth.md
  lowvol.md
  momentum.md
  reversal.md
  summary.csv
  ic_by_year.csv
  group_returns.csv
  decay.csv
  coverage.csv
```

代码切入点：

- `src/ashare_alpha/analysis/factor_ic.py`
- 新增 `src/ashare_alpha/analysis/factor_group.py`
- 新增 `src/ashare_alpha/analysis/factor_decay.py`
- `scripts/06_make_report.py`

验收标准：

- 每个因子至少有 IC、分组收益、覆盖率三类诊断。
- 综合模型不得使用诊断结果长期为负且无解释的因子。

### 2.2 缺失值策略

当前综合打分中：

```python
out["score"] += weight * out[factor_name].fillna(0)
```

这会把缺失值当作中性值。需要改成可配置策略：

```yaml
missing_policy:
  method: neutral_by_date_industry
  min_factor_count: 4
```

候选策略：

- `drop_if_missing_core_factor`
- `neutral_by_date`
- `neutral_by_date_industry`
- `median_by_date_industry`
- `penalize_missing`

验收标准：

- 每个交易日记录每只股票可用因子数量。
- 可配置最少有效因子数，例如至少 4 个因子有效才进入打分。
- 报告展示缺失率最高的因子和时间段。

### 2.3 因子相关性与冗余控制

当前 `value` 与 `lowvol` 相关性约 0.497，存在风格重叠。

需要：

- 输出滚动因子相关矩阵。
- 对高度相关因子做降权或正交化实验。
- 检查综合分是否被单一风格主导。

验收标准：

- 报告展示综合分与各因子的相关性。
- 任一单因子与综合分相关性长期超过 0.75 时触发提示。

---

## Phase 3：综合打分模型升级

目标：从人工固定权重升级到有证据支持的权重体系。

### 3.1 修正当前固定权重

先做最小改动版本 `baseline_v1_fixed_weight_clean`：

- 删除或实现 `leverage`。
- 对实际存在的因子权重重新归一化。
- 暂停或反向测试 `momentum`。
- 降低 `quality` 权重，直到诊断证明它有效。
- 保留 `value`、`lowvol`、`reversal` 作为当前主力因子。

建议先做三组实验：

```text
Experiment A：删除 momentum，权重重新归一化
Experiment B：momentum 取反，观察是否等价于中期反转
Experiment C：只用 value + lowvol + reversal
```

验收标准：

- 每组实验都输出绝对收益、超额收益、最大回撤、IR、换手、IC。
- 若收益提升但回撤或换手显著恶化，不直接采纳。

### 3.2 滚动 IC 加权

实现 `rolling_ic_weight`：

```text
过去 12 或 24 个月 RankIC
→ 只使用当期之前的数据
→ IC 均值 / 波动 得到稳定性分数
→ 负 IC 因子自动降权或反向
→ 权重平滑，限制单因子最大权重
```

配置示例：

```yaml
score_model:
  type: rolling_ic_weight
  lookback_months: 24
  min_history_months: 12
  max_factor_weight: 0.35
  weight_smoothing: 0.5
  allow_negative_weight: false
```

验收标准：

- 任意调仓日的权重只使用该日前历史 IC。
- 输出 `results/factor_weights.csv`。
- 权重变化平滑，不出现单月极端跳变。

### 3.3 线性学习模型

在滚动 IC 权重稳定后，再加入线性模型：

- Ridge 回归。
- Lasso / ElasticNet。
- Logistic top quantile 分类。
- 横截面 rank regression。

标签：

```text
未来 20 日收益
或未来 20 日行业/市值中性残差收益
或未来 20 日相对基准收益
```

验证方式：

```text
2018-2020 train
2021 validation
2022 test
然后滚动前进
```

验收标准：

- 模型训练严格使用过去数据。
- 输出样本内、验证集、样本外三段指标。
- 如果线性模型不能稳定超过清洗后的固定权重模型，则不升级。

### 3.4 机器学习排序模型

只有在数据偏差修正、因子诊断和线性模型完成后，再考虑：

- LightGBM ranker。
- XGBoost ranker。
- CatBoost ranker。

输入特征：

- 当前已有六类风格因子。
- 因子原始值与标准化值。
- 市值、流动性、波动、换手。
- 行业 one-hot 或行业内排名。
- 因子过去一段时间稳定性。

风险：

- 样本只有 500 股票时机器学习极易过拟合。
- 如果扩展到全 A，机器学习才更有意义。

验收标准：

- 必须做 walk-forward。
- 必须输出特征重要性稳定性。
- 必须与固定权重、滚动 IC 权重做同口径比较。
- 不允许只看总收益采纳模型。

---

## Phase 4：组合构建升级

目标：从 Top 50 等权升级到可控制风险、成本和暴露的组合构建。

### 4.1 修复行业和个股约束

当前行业约束未生效，应先修：

- `generate_signal` 合并行业字段。
- `generate_target_weights` 保留行业字段。
- `apply_position_constraints` 后检查是否仍有行业超限。
- 若归一化后重新超限，使用迭代 cap 或优化器。

验收标准：

- `target_weights.parquet` 包含行业字段。
- 每期行业权重不超过 `max_industry_weight`。
- 新增测试覆盖行业约束和归一化后的再次超限。

### 4.2 加入换手控制

当前平均换手约 37.36%，需要控制交易成本和组合稳定性。

方法：

- 对上期持仓保留缓冲。
- Top 50 买入、Top 80 卖出。
- 设置单次调仓最大换手。
- 目标函数加入交易成本惩罚。

配置示例：

```yaml
portfolio:
  entry_rank: 50
  exit_rank: 80
  max_turnover_per_rebalance: 0.30
```

验收标准：

- 换手下降后超额收益没有明显塌陷。
- 报告展示收益变化来自选股还是换手下降。

### 4.3 加入流动性约束

当前回测未严格限制成交金额占成交额比例。

需要：

- 每笔交易金额 <= 过去 20 日平均成交额的 5%。
- 不满足时部分成交或延迟成交。
- 报告输出未成交金额。

验收标准：

- 每笔交易有 `fill_ratio`。
- 回测输出 `unfilled_orders.csv`。

### 4.4 从等权到得分加权

候选方法：

- 等权：当前基线。
- rank weight：排名越高权重越大。
- score weight：分数归一化后给权重。
- risk adjusted score weight：分数 / 波动。
- constrained optimization：最大化分数暴露，约束行业、市值、换手、个股上限。

验收标准：

- 每种方法与等权同口径比较。
- 若得分加权增加集中度但不提升 IR，不采纳。

---

## Phase 5：回测真实性升级

目标：让回测更接近可交易结果。

### 5.1 成交模型

新增：

- 成交量参与率。
- 部分成交。
- 开盘涨跌停无法成交后的再处理。
- 停牌期间估值与持仓冻结。
- 卖出失败时后续持仓继续保留。

代码切入点：

- `src/ashare_alpha/backtest/execution.py`
- `src/ashare_alpha/backtest/broker.py`
- `src/ashare_alpha/backtest/engine.py`

验收标准：

- 每笔订单有 intended、filled、unfilled。
- 回测能区分订单和成交。

### 5.2 成本与滑点压力测试

增加参数网格：

```text
slippage: 5bp, 10bp, 20bp, 50bp
commission: 当前值、两倍当前值
participation cap: 5%, 2%, 1%
```

输出：

```text
results/stress/cost_sensitivity.csv
```

验收标准：

- 策略在成本加倍后仍有正超额，才算具备基本可交易性。

### 5.3 交易日和信号日严格隔离

当前已有 `rebalance_date` 和 `execution_date`，但需要全链路保证：

- 因子只使用 `signal_date` 当日收盘后可知数据。
- 财务数据必须满足 `ann_date <= signal_date`。
- 调仓使用下一交易日价格。
- 若下一交易日停牌或涨跌停，不能偷偷成交。

验收标准：

- 新增未来函数审计脚本。
- 任一信号行可追溯到其使用的数据时间戳。

---

## Phase 6：绩效归因与风险报告

目标：从“知道赚了多少钱”升级到“知道赚的是什么钱”。

### 6.1 分解收益来源

新增：

- 行业收益归因。
- 市值暴露归因。
- 风格暴露归因。
- 个股贡献前 20。
- 月度收益贡献。
- 调仓收益与持有收益拆分。

输出：

```text
results/attribution/
  industry_attribution.csv
  size_exposure.csv
  factor_exposure.csv
  top_contributors.csv
```

### 6.2 风险指标扩展

新增：

- 下行波动。
- Sortino。
- VaR / CVaR。
- 最大回撤持续时间。
- 回撤恢复时间。
- 最差 10 日收益。
- 连续亏损月数。
- 持仓集中度。

### 6.3 市场环境切片

按以下维度评估：

- 牛市、熊市、震荡市。
- 高波动、低波动。
- 大盘占优、小盘占优。
- 价值占优、成长占优。
- 流动性宽松、流动性收缩。

验收标准：

- 报告能解释 2018 大回撤和 2023 停滞。
- 每个年份至少有收益、超额、回撤、换手和 IC 解释。

---

## Phase 7：工程化与实验管理

目标：让研究迭代更快、更稳、更不容易混淆结果。

### 7.1 统一 CLI

新增：

```bash
python -m ashare_alpha.run_all --config config --experiment baseline_v1
```

支持：

- 下载数据。
- 构建面板。
- 计算因子。
- 生成信号。
- 回测。
- 生成报告。
- 只运行某一步。
- 强制重跑或使用缓存。

### 7.2 实验目录结构

建议：

```text
results/
  experiments/
    baseline_v0/
      config/
      metrics.json
      factor_ic.json
      nav.csv
      trades.csv
      positions.csv
      report.md
    baseline_v1_fixed_weight_clean/
    rolling_ic_v1/
```

验收标准：

- 每次实验不会覆盖上一次结果。
- 实验之间可以自动生成对比表。

### 7.3 测试扩展

新增测试：

- 行业约束必然生效。
- 股票池过滤器逐项生效。
- `leverage` 配置缺失时触发警告或自动忽略并归一化。
- 因子缺失值策略正确。
- 信号日和执行日严格错开。
- 成交量参与率约束正确。
- 历史 industry `out_date` 正确处理。

---

## 7. 近期优先任务

建议按以下顺序推进。

### 任务 1：修复行业字段和行业约束

涉及文件：

- `scripts/04_generate_signals.py`
- `src/ashare_alpha/strategy/signal.py`
- `src/ashare_alpha/strategy/portfolio.py`
- `src/ashare_alpha/strategy/constraints.py`
- `tests/test_universe.py` 或新增 `tests/test_constraints.py`

验收：

- `target_weights.parquet` 包含 `industry_code` 和 `industry_name`。
- 每期行业权重不超过 20%。
- 新测试通过。

### 任务 2：修正 `leverage` 权重不一致

可选方案：

1. 先删除配置里的 `leverage`。
2. 或实现杠杆因子：`-debt_to_assets` 标准化。
3. 或综合打分时只对存在因子重新归一化，并记录警告。

建议先选方案 3，随后把 leverage 因子作为独立实验。

验收：

- 综合打分报告中显示实际使用因子和实际权重。
- 不存在静默忽略权重的情况。

### 任务 3：补齐股票池过滤器

涉及：

- `exclude_bj`
- `max_suspend_days_60`
- 历史 ST 状态占位接口
- delist 逻辑

验收：

- 每个过滤器有测试。
- 每个调仓日有过滤漏斗统计。

### 任务 4：生成单因子诊断报告

先实现：

- 分年 IC。
- 分组收益。
- 因子覆盖率。
- 因子衰减。

验收：

- 能明确回答：`quality` 是否应该保留，`momentum` 是否应该取反或删除。

### 任务 5：重新设定固定权重基线

实验组：

- `baseline_v1_no_momentum`
- `baseline_v1_reverse_momentum`
- `baseline_v1_value_lowvol_reversal`
- `baseline_v1_ic_weighted_static`

验收：

- 与 `baseline_v0` 同口径对比。
- 不只比较收益，也比较最大回撤、IR、换手、因子暴露。

---

## 8. 推荐的下一版模型形态

下一版不要直接跳到机器学习。推荐先做：

```text
baseline_v1_clean_factor_model
```

设计：

- Universe：先明确当前 500 样本为调试池，同时准备全 A 或历史成分池。
- 因子：value、lowvol、reversal 为核心，growth 辅助，quality 和 momentum 进入观察。
- 权重：根据单因子 IC 和稳定性重新设定，实际可用因子权重归一化。
- 缺失值：行业内中位数或中性填充，并要求最少有效因子数。
- 组合：Top 50 等权保留为基线，但行业约束必须真实生效。
- 回测：加入成交量参与率压力测试。
- 报告：输出单因子诊断、行业暴露、分年收益和成本敏感性。

推荐采纳条件：

```text
1. 修正样本和约束后，年化超额仍为正。
2. 信息比率不低于 baseline_v0 的 70%。
3. 最大回撤不高于 baseline_v0。
4. 换手不显著升高。
5. 至少 3 个核心因子在分年诊断中具有稳定正贡献。
```

---

## 9. 长期目标

最终希望把项目升级成一个可持续研究平台：

```text
数据可信
→ 因子可诊断
→ 权重可解释
→ 组合可约束
→ 回测可复验
→ 报告可比较
→ 实验可追踪
```

成熟版本应具备：

- 全 A 历史样本。
- 历史股票状态与历史行业分类。
- 多 horizon 因子库。
- 单因子和组合因子诊断。
- 滚动权重或轻量学习模型。
- 可约束组合优化。
- 更真实的成交模型。
- 自动实验对比。
- 每次运行都有 manifest 和版本记录。

---

## 10. 当前结论

当前模型已经证明了项目方向可行：多因子闭环跑通，策略在当前样本上有明显超额，低波和反转等因子表现较好。

但当前结果还不能作为可靠 Alpha 结论，主要原因是股票样本可能存在历史成分前视，行业约束未实际生效，因子权重没有经过严格诊断，部分配置没有落地。

下一步最有价值的工作不是新增复杂模型，而是：

```text
修样本偏差
修约束生效
修配置一致性
做单因子诊断
重建固定权重基线
再考虑滚动 IC 权重和机器学习
```

只要这五件事完成，项目就能从“能跑通的 MVP”升级为“值得继续研究的 A 股 Alpha 实验室”。

---

## 11. 执行进度记录（2026-06-15）

本轮已完成：

- 完成近期任务 1 的核心闭环：`factor_score.parquet` 和 `target_weights.parquet` 已保留 `industry_code`、`industry_name`；`generate_signal` 也会在旧 score 缺字段时从当日 universe 补回行业字段。
- 修复行业约束生效问题：`apply_position_constraints` 不再在行业 cap 后粗暴归一化打穿上限，会只向仍有单股/行业容量的位置分配剩余权重。
- 完成近期任务 2 的建议方案 3：综合打分只对实际存在的因子权重重新归一化，缺失的 `leverage` 会触发 `RuntimeWarning`，并在分数产物中输出 `weight_*` 与 `factor_count`。
- 补充行业 as-of 的 `out_date` 处理：行业有效条件改为 `in_date <= trade_date` 且 `out_date` 为空或 `out_date > trade_date`。
- 新增/更新测试覆盖行业字段传递、行业约束、`leverage` 缺失权重归一化、行业 `out_date` 边界。

验证结果：

- `pytest -q`：`23 passed`。
- 已重跑 `scripts/03_calc_factors.py`、`scripts/04_generate_signals.py`、`scripts/05_run_backtest.py`、`scripts/06_make_report.py`。
- 新 `target_weights.parquet`：4,750 行，95 个调仓日，包含行业字段；单股最大权重 2.00%，行业最大权重约 20.00%。
- 约束真实生效后，每期目标权重总和为 46.00% 至 68.00%，剩余部分保留现金。

重算后的当前结果：

| 指标 | 约束生效后 |
| --- | ---: |
| 最终净值 | 1.6411 |
| 总收益 | 64.11% |
| 年化收益 | 6.64% |
| 年化波动 | 11.81% |
| 夏普比率 | 0.39 |
| 最大回撤 | -20.43% |
| 平均换手 | 21.08% |
| 相对中证 500 年化超额 | 4.39% |
| 相对中证 500 信息比率 | 0.19 |
| 相对中证 1000 年化超额 | 5.70% |
| 相对中证 1000 信息比率 | 0.19 |

新增发现：

- 当前 `Top 50 + 等权 + 单股上限 2% + 行业上限 20%` 在很多调仓日不可同时满仓。原因是 Top 50 等权已经把每只股票推到 2% 单股上限，行业降权后其他股票没有剩余单股容量承接权重。
- 这说明行业约束此前确实未真正落地；落地后收益和 IR 明显下降，但回撤和换手也下降。后续需要用“扩展候选池后补足权重”或“约束优化器”重建 `baseline_v1_clean_factor_model`，而不是直接把受约束后的半仓结果视为最终模型。

下一步建议：

1. 为行业约束加入替补选股机制，例如从 Top 50 扩展到 Top 100 候选，在行业/单股 cap 下填满目标仓位。
2. 将本轮结果归档为 `baseline_v1_constraints_active`，并与原 `baseline_v0` 做同口径对比。
3. 继续推进近期任务 3：补齐 `exclude_bj`、`max_suspend_days_60`、历史 ST 占位接口和 delist 逻辑。

---

## 12. 执行进度记录（2026-06-15，关闭行业硬约束）

本轮决策：

- 不再限制组合行业集中度，`config/strategy.yaml` 中 `portfolio.max_industry_weight` 已改为 `null`。
- 保留 `industry_code` 和 `industry_name` 字段，用于行业暴露统计和后续诊断。
- 代码层面支持 `max_industry_weight: null`，避免空值误触发行业约束。

完成改动：

- `src/ashare_alpha/strategy/constraints.py`：仅当 `max_industry_weight` 为有效数值时才执行行业 cap。
- `tests/test_constraints.py`：新增 `max_industry_weight: None` 的测试，确认同一行业可以持有 100% 权重。
- 重新生成信号、回测和报告。

验证结果：

- `pytest -q`：`24 passed`。
- `target_weights.parquet`：4,750 行，95 个调仓日。
- 每期目标权重总和：最小 1.00，最大 1.00。
- 单股最大权重：2.00%。
- 行业集中度不再限制：观测到的单期最大行业权重为 64.00%，每期最大行业权重均值约 51.37%。

重算后的当前结果：

| 指标 | 不限制行业集中度 |
| --- | ---: |
| 最终净值 | 2.2333 |
| 总收益 | 123.33% |
| 年化收益 | 10.99% |
| 年化波动 | 20.82% |
| 夏普比率 | 0.43 |
| 最大回撤 | -33.19% |
| 平均换手 | 37.36% |
| 相对中证 500 年化超额 | 8.64% |
| 相对中证 500 信息比率 | 0.68 |
| 相对中证 1000 年化超额 | 10.01% |
| 相对中证 1000 信息比率 | 0.58 |

当前结论更新：

- 主线基线改为“不限制行业集中度、但监控行业暴露”的版本。
- 第 11 节的行业硬约束半仓结果保留为诊断记录，不作为当前主线模型。
- 后续如果重新加入行业控制，应优先采用软约束、行业暴露提示、或基于更大候选池的组合优化，而不是在 Top 50 等权组合上硬切行业 cap。

下一步建议更新：

1. 优先补齐股票池过滤器：`exclude_bj`、`max_suspend_days_60`、历史 ST 占位接口和 delist 逻辑。
2. 生成单因子诊断报告，重点判断 `quality` 和 `momentum` 是否应该继续进入综合分。
3. 在报告中加强行业暴露展示，明确收益是否来自行业集中，而不是先做硬行业约束。

---

## 13. 执行进度记录（2026-06-16，补齐股票池过滤器）

本轮完成近期任务 3 的主体实现：

- `src/ashare_alpha/strategy/universe.py` 新增 `filter_bj`、`filter_delisted`、`filter_suspend_days`，并让 `filter_st` 支持可选历史 `stock_status` 面板。
- 历史 ST 接口采用占位兼容设计：若存在 `stock_status_panel.parquet` 或 `stock_status.parquet`，按 `trade_date <= 当前调仓日` 的最新状态过滤；若不存在，则回退到当前名称里的 `ST|退` 过滤，并记录日志。
- `exclude_new_stock` 现在真正控制上市天数过滤，`exclude_negative_pe` 和 `exclude_negative_pb` 可以分别控制估值合法性过滤。
- `build_universe` 支持 `return_filter_stats=True`，每个调仓日记录 `initial`、`listed_days`、`delisted`、`bj`、`st`、`valuation`、`liquidity`、`suspend_days_60` 的过滤前后数量。
- `scripts/04_generate_signals.py` 会保存过滤漏斗到 `data/signals/universe_filter_stats.csv`。
- `scripts/06_make_report.py` 会汇总过滤漏斗到 `results/universe_filter_stats_summary.csv`。
- `tests/test_universe.py` 新增测试覆盖北交所过滤、退市日期过滤、历史 ST 状态优先级、60 日停牌天数过滤、过滤漏斗统计。

验证结果：

- `pytest -q`：`29 passed`。
- 已重跑 `scripts/04_generate_signals.py`、`scripts/05_run_backtest.py`、`scripts/06_make_report.py`。
- `target_weights.parquet`：4,750 行，95 个调仓日。
- 每期目标权重总和：最小 1.00，最大 1.00。
- 单股最大权重：2.00%。
- `data/signals/universe_filter_stats.csv`：768 行，对应 96 个调仓日 × 8 个过滤步骤。
- `results/universe_filter_stats_summary.csv` 已生成。

过滤漏斗汇总：

| 过滤器 | 调仓日数 | 平均过滤前 | 平均过滤后 | 平均剔除 | 最大单期剔除 | 总剔除 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| initial | 96 | 446.01 | 446.01 | 0.00 | 0 | 0 |
| listed_days | 96 | 446.01 | 434.32 | 11.69 | 28 | 1122 |
| delisted | 96 | 434.32 | 434.32 | 0.00 | 0 | 0 |
| bj | 96 | 434.32 | 434.32 | 0.00 | 0 | 0 |
| st | 96 | 434.32 | 434.32 | 0.00 | 0 | 0 |
| valuation | 96 | 434.32 | 400.42 | 33.91 | 60 | 3255 |
| liquidity | 96 | 400.42 | 383.40 | 17.02 | 330 | 1634 |
| suspend_days_60 | 96 | 383.40 | 383.40 | 0.00 | 0 | 0 |

重算后的当前结果：

| 指标 | 当前值 |
| --- | ---: |
| 最终净值 | 2.2333 |
| 总收益 | 123.33% |
| 年化收益 | 10.99% |
| 年化波动 | 20.82% |
| 夏普比率 | 0.43 |
| 最大回撤 | -33.19% |
| 平均换手 | 37.36% |
| 相对中证 500 年化超额 | 8.64% |
| 相对中证 500 信息比率 | 0.68 |
| 相对中证 1000 年化超额 | 10.01% |
| 相对中证 1000 信息比率 | 0.58 |

新增发现：

- 当前 500 样本中没有实际触发 `exclude_bj`、`delisted`、历史/当前 ST、`max_suspend_days_60` 剔除；这说明接口已落地，但当前样本仍缺少更完整的全 A 或历史状态数据来检验其研究影响。
- 首个调仓日 `20180102` 因 20 日成交额窗口不足，流动性过滤后为空，因此最终仍是 95 个有效调仓日。
- 本轮过滤器落地没有改变当前主线持仓与绩效，更多是补齐偏差控制框架和诊断可见性。

下一步建议更新：

1. 推进近期任务 4：生成单因子诊断报告，先落地分年 IC、分组收益、覆盖率和衰减。
2. 准备更完整的股票状态数据或全 A 样本，让 `exclude_bj`、历史 ST、退市逻辑能在真实样本中接受压力测试。
3. 后续报告可继续加强行业暴露和过滤漏斗的可视化展示。

---

## 14. 执行进度记录（2026-06-16，生成单因子诊断报告）

本轮完成近期任务 4 的第一版落地：

- 新增 `src/ashare_alpha/analysis/factor_group.py`，用于计算单因子分组收益和 top-minus-bottom 收益。
- 新增 `src/ashare_alpha/analysis/factor_decay.py`，用于计算 5、10、20、40、60 日 horizon 的 RankIC 衰减。
- 新增 `src/ashare_alpha/analysis/factor_report.py`，统一生成单因子诊断产物。
- 扩展 `src/ashare_alpha/analysis/factor_ic.py`，支持批量 RankIC、分年 IC 和更稳健的 IC 摘要。
- `scripts/06_make_report.py` 已接入单因子诊断，每次生成报告时同步刷新 `results/factor_report/`。
- 新增 `tests/test_factor_diagnostics.py`，覆盖分组收益、IC 衰减、覆盖率和负 IC 因子的建议标记。

新增输出：

```text
results/factor_report/
  summary.csv
  ic_by_year.csv
  group_returns.csv
  decay.csv
  coverage.csv
  value.md
  quality.md
  growth.md
  lowvol.md
  momentum.md
  reversal.md
```

验证结果：

- `pytest -q`：`31 passed`。
- 已重跑 `scripts/06_make_report.py`，并生成上述 `factor_report` 产物。

核心诊断摘要：

| 因子 | 20日 Mean IC | ICIR | 正 IC 比例 | 分组多空年化 | 60日 Mean IC | 当前判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| value | 0.0255 | 0.1170 | 52.60% | -5.17% | 0.0222 | IC 弱正，但分组收益反向，需复核分组口径与因子方向 |
| quality | 0.0035 | 0.0263 | 50.38% | -2.11% | -0.0145 | 基本无效，不应继续维持 0.20 高权重 |
| growth | 0.0104 | 0.0871 | 53.72% | 6.83% | 0.0012 | 弱正，可作为辅助观察因子 |
| lowvol | 0.0461 | 0.2221 | 58.00% | -5.91% | 0.0543 | IC 稳定偏强，但分组收益反向，需优先复核 |
| momentum | -0.0126 | -0.0778 | 49.67% | -0.04% | -0.0028 | 20日方向为负，应删除或做反向实验 |
| reversal | 0.0442 | 0.2877 | 59.83% | 7.24% | 0.0318 | 当前最值得保留的核心因子 |

覆盖率发现：

- `reversal`、`growth`、`quality`、`lowvol`、`momentum` 平均覆盖率均接近或高于 98%。
- `value` 平均覆盖率约 92.43%，低于其他因子，主要受估值数据可用性影响。
- `quality` 和 `growth` 的最低覆盖率约 17.70%，说明早期财务类因子存在明显数据窗口不足，需要在报告中继续标记。

新增发现：

- `quality` 的 20 日 IC 几乎为 0，60 日 IC 转负，当前证据不支持继续给 0.20 权重。
- `momentum` 的 20 日 RankIC 为负，继续正向纳入综合分会拖累模型，应进入 `no_momentum` 和 `reverse_momentum` 实验。
- `reversal` 在 IC、ICIR、正 IC 比例和分组多空上均最稳定，适合作为下一版固定权重核心。
- `value` 和 `lowvol` 出现“RankIC 为正但五组多空收益为负”的口径冲突，后续需要检查因子方向、行业内标准化、大量 0 填充、重叠未来收益和按全交易日计算分组收益是否造成偏差。

下一步建议更新：

1. 推进近期任务 5：先做 `baseline_v1_no_momentum`、`baseline_v1_reverse_momentum`、`baseline_v1_value_lowvol_reversal`、`baseline_v1_ic_weighted_static` 四组同口径实验。
2. 在权重实验前，先复核 `value` 与 `lowvol` 的分组收益口径，必要时改为只在调仓日、股票池过滤后样本上计算分组收益。
3. 暂时将 `quality` 降为观察因子，将 `momentum` 从主线综合分中移除或单独反向测试。

---

## 15. 执行进度记录（2026-06-16，复核调仓口径因子诊断并跑固定权重实验）

本轮完成近期任务 5 的第一版同口径实验，并先修正了单因子分组收益的评估口径：

- `scripts/04_generate_signals.py` 新增保存调仓日过滤后股票池到 `data/signals/rebalance_universe.parquet`，用于后续诊断只在真实可选股票池上计算。
- `src/ashare_alpha/analysis/factor_report.py` 新增 `eligible_universe` 参数和 `filter_factor_scores_by_universe`，支持按调仓股票池过滤 IC、分组收益、覆盖率和衰减。
- `scripts/06_make_report.py` 保留原 `results/factor_report/`，并新增输出 `results/factor_report_rebalance_universe/`。
- 新增 `src/ashare_alpha/analysis/weight_experiments.py`，用于 IC 静态权重推导、绩效指标拍平和组合因子暴露汇总。
- 新增 `scripts/07_run_weight_experiments.py`，一次性运行当前基线和四组固定权重实验，产物统一保存到 `results/experiments/weight_baselines/`。
- 新增 `tests/test_weight_experiments.py`，并扩展 `tests/test_factor_diagnostics.py`，覆盖调仓股票池口径诊断和权重实验辅助函数。

新增输出：

```text
data/signals/rebalance_universe.parquet

results/factor_report_rebalance_universe/
  summary.csv
  ic_by_year.csv
  group_returns.csv
  decay.csv
  coverage.csv
  value.md
  quality.md
  growth.md
  lowvol.md
  momentum.md
  reversal.md

results/experiments/weight_baselines/
  summary.csv
  comparison.md
  baseline_v0_current_weights/
  baseline_v1_no_momentum/
  baseline_v1_reverse_momentum/
  baseline_v1_value_lowvol_reversal/
  baseline_v1_ic_weighted_static/
```

验证结果：

- `pytest -q`：`34 passed`。
- 已重跑 `scripts/04_generate_signals.py`，生成 `target_weights.parquet` 4,750 行、`universe_filter_stats.csv` 768 行、`rebalance_universe.parquet` 36,806 行。
- 已重跑 `scripts/05_run_backtest.py`，当前主线最终净值仍为 `2.2333`。
- 已重跑 `scripts/06_make_report.py`，生成全样本和调仓股票池两套因子诊断报告。
- 已运行 `scripts/07_run_weight_experiments.py`，完成当前基线加四组 v1 权重实验。

调仓股票池口径下的核心因子诊断：

| 因子 | 20日 Mean IC | ICIR | 正 IC 比例 | 分组多空年化 | 当前判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| value | 0.0147 | 0.0651 | 50.53% | -8.26% | 弱正，但分组收益仍反向 |
| quality | 0.0004 | 0.0034 | 51.09% | -4.34% | 基本无效，应降为观察因子 |
| growth | 0.0123 | 0.1005 | 52.17% | 7.43% | 弱正，可保留观察 |
| lowvol | 0.0403 | 0.1994 | 61.96% | -10.42% | IC 较强，但分组收益仍反向 |
| momentum | -0.0042 | -0.0252 | 50.56% | 3.03% | IC 为负，不宜正向纳入 |
| reversal | 0.0357 | 0.2396 | 52.63% | 6.94% | 当前最稳定核心因子 |

同口径固定权重实验结果：

| 实验 | 最终净值 | 年化收益 | 最大回撤 | 平均换手 | 中证500年化超额 | 中证500 IR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_v0_current_weights | 2.2333 | 10.99% | -33.19% | 37.36% | 8.64% | 0.68 |
| baseline_v1_no_momentum | 1.8269 | 8.13% | -33.02% | 32.23% | 5.85% | 0.45 |
| baseline_v1_reverse_momentum | 1.8614 | 8.40% | -34.47% | 30.94% | 6.11% | 0.47 |
| baseline_v1_value_lowvol_reversal | 1.9898 | 9.34% | -34.19% | 46.63% | 7.03% | 0.52 |
| baseline_v1_ic_weighted_static | 1.8641 | 8.42% | -32.03% | 55.37% | 6.13% | 0.47 |

`baseline_v1_ic_weighted_static` 使用调仓股票池口径正 IC 静态权重，剔除了 `quality` 和 `momentum`：

```json
{
  "value": 0.1426,
  "growth": 0.1194,
  "lowvol": 0.3913,
  "reversal": 0.3467
}
```

新增发现：

- 四组 v1 固定权重实验均未超过当前 `baseline_v0_current_weights`。短期不宜直接替换主线权重。
- `baseline_v1_value_lowvol_reversal` 是四组 v1 里收益最好的方案，但最大回撤略深、换手升至 46.63%，还不是更稳的主线候选。
- `baseline_v1_ic_weighted_static` 最大回撤最小，为 -32.03%，但换手升至 55.37%，说明简单按全样本静态 IC 加权会显著改变持仓并抬高交易强度。
- `no_momentum` 和 `reverse_momentum` 都降低了收益，说明当前 `momentum` 虽然单因子 IC 为负，但它可能通过与其他因子的相关结构影响组合排序，不能只按单因子结论机械删除。
- `value` 与 `lowvol` 在调仓股票池口径下仍出现“IC 为正、分组多空收益为负”的冲突，问题不只是全交易日样本口径造成的，后续需要继续复核未来收益计算、分组方向、行业中性化和重叠收益口径。

下一步建议更新：

1. 先不要替换当前主线权重，保留 `baseline_v0_current_weights` 作为继续对照的主线基线。
2. 优先排查 `value` 和 `lowvol` 的 IC 与分组收益冲突，重点检查因子方向、未来收益 horizon、调仓日执行价格、行业内标准化和分组收益年化方式。
3. 对 `baseline_v1_value_lowvol_reversal` 做换手约束或缓冲带实验，判断收益下降是否能换来更低回撤和交易成本。
4. 对 `baseline_v1_ic_weighted_static` 增加权重平滑、权重上限和换手惩罚，不宜直接采用当前静态 IC 权重。
5. 后续实验汇总统一使用 `results/experiments/weight_baselines/summary.csv` 和各实验目录下的 `experiment_manifest.json` 做可追踪对比。

---

## 16. 执行进度记录（2026-06-16，排查 IC 与分组收益冲突并增强 payoff 诊断）

本轮推进近期任务 5 的后续排查，重点复核 `value` 和 `lowvol` 的“RankIC 为正、分组多空收益为负”冲突：

- `src/ashare_alpha/analysis/factor_ic.py` 新增通用 `calc_forward_return_from_prices`，支持指定起点价格、终点价格、起点 lag 和终点 lag；原 `calc_forward_return` 保持 close-to-close 兼容口径。
- `src/ashare_alpha/analysis/factor_group.py` 新增逐期 payoff 诊断：
  - 每个因子、每个调仓日输出 RankIC、Pearson IC、标准化线性 payoff、市场平均未来收益、收益离散度、五分组高低组收益和多空收益。
  - 新增“次日开盘执行到后续收盘”的 execution payoff 口径，用于检查调仓执行价格是否解释冲突。
  - 对常数序列相关系数增加保护，避免诊断过程产生无效相关性警告。
- `src/ashare_alpha/analysis/factor_report.py` 接入 payoff 诊断，并将其写入单因子 markdown；综合 summary 中新增 Pearson、线性 payoff、多空胜率、收益幅度加权 RankIC 等字段。
- `tests/test_factor_diagnostics.py` 扩展测试，覆盖新增 payoff 产物和 execution payoff 产物。

新增输出：

```text
results/factor_report/
  payoff_by_date.csv
  payoff_summary.csv
  execution_payoff_by_date.csv
  execution_payoff_summary.csv

results/factor_report_rebalance_universe/
  payoff_by_date.csv
  payoff_summary.csv
  execution_payoff_by_date.csv
  execution_payoff_summary.csv
```

验证结果：

- `pytest -q`：`34 passed`。
- 已重跑 `scripts/06_make_report.py`，刷新全样本和调仓股票池两套因子诊断报告。
- 报告脚本仍有 Matplotlib 中文字体 glyph warning，来自本地绘图字体环境，不影响 CSV、metrics 和 markdown 产物。

调仓股票池 close-to-close payoff 核心结果：

| 因子 | Mean RankIC | Mean Pearson IC | 标准化 payoff | 多空 20日收益 | RankIC 正比例 | 多空正收益比例 | 收益幅度加权 RankIC | 当前判断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| value | 0.0147 | -0.0096 | -0.0025 | -0.6820% | 50.53% | 47.37% | 0.0008 | RankIC 与尾部分组收益冲突 |
| lowvol | 0.0403 | -0.0049 | -0.0024 | -0.8694% | 61.96% | 45.65% | 0.0003 | RankIC 与尾部分组收益冲突 |
| reversal | 0.0357 | 0.0186 | 0.0016 | 0.5338% | 52.63% | 48.42% | 0.0727 | 当前最稳核心因子 |
| growth | 0.0123 | 0.0160 | 0.0023 | 0.5701% | 52.17% | 56.52% | 0.0318 | 弱正，可观察 |
| quality | 0.0004 | -0.0063 | -0.0015 | -0.3517% | 51.09% | 48.91% | -0.0319 | 弱因子，应降权观察 |
| momentum | -0.0042 | 0.0050 | 0.0010 | 0.2371% | 50.56% | 53.93% | -0.0054 | RankIC 为负，仍不宜正向纳入 |

调仓股票池 execution payoff 核心结果：

| 因子 | Mean RankIC | Mean Pearson IC | 标准化 payoff | 多空 20日收益 | RankIC 正比例 | 多空正收益比例 | 收益幅度加权 RankIC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| value | 0.0134 | -0.0089 | -0.0027 | -0.6947% | 53.68% | 47.37% | 0.0013 |
| lowvol | 0.0339 | -0.0063 | -0.0029 | -1.0817% | 57.61% | 46.74% | -0.0138 |
| reversal | 0.0382 | 0.0208 | 0.0017 | 0.5730% | 53.68% | 50.53% | 0.0810 |
| growth | 0.0099 | 0.0159 | 0.0024 | 0.6041% | 53.26% | 56.52% | 0.0297 |

新增发现：

- `value` 和 `lowvol` 的冲突不是全交易日样本口径造成的，也不是次日开盘执行价单独造成的；在调仓股票池和 execution payoff 口径下仍然存在。
- 更准确的解释是：RankIC 是逐期同权的秩相关，而五分组多空是尾部组合收益，受收益幅度、尾部月份和线性 payoff 影响更大。`value` 和 `lowvol` 的平均 RankIC 为正，但 Pearson IC、标准化 payoff 和尾部分组多空均为负，说明它们的正 RankIC 主要不是稳定可交易的 top-minus-bottom 收益。
- `lowvol` 的 RankIC 正比例较高，但收益幅度加权 RankIC 接近 0，execution 口径下转负，说明正 IC 月份的收益幅度不足以抵消负向尾部月份。
- `reversal` 在 close-to-close 和 execution 口径下均保持正 Pearson、正线性 payoff 和正多空收益，仍是当前最可靠的核心因子。
- `growth` 的线性 payoff 和分组多空为正，但 RankIC 和稳定性偏弱，适合作为观察或辅助因子。
- `quality` 继续缺乏有效证据，应保持弱因子/降权观察标签。

下一步建议更新：

1. 暂不把 `value` 和 `lowvol` 直接作为“可交易尾部收益”核心因子，先把它们标记为 `rank_ic_group_return_conflict_review`，继续检查是否来自行业内标准化、因子非线性或极端月份暴露。
2. 优先对 `reversal + growth`、`reversal + lowvol`、`reversal + value` 做小规模消融实验，比较收益是否来自 `reversal` 单因子还是组合相关结构。
3. 对 `baseline_v1_value_lowvol_reversal` 做缓冲带或换手约束实验时，要额外观察 `value/lowvol` 的逐期 payoff 是否改善，而不是只看组合总收益。
4. 后续报告可增加 payoff by year 和极端月份归因，把 `value`、`lowvol` 的负尾部月份单独列出。
