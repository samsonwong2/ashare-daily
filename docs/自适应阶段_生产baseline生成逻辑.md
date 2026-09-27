# 自适应阶段：生产 baseline 生成逻辑

> **身份：生产说明。** 每天 `ashare-daily adaptive` 走的就是这条 B0 + `hold_up`。图7–11、六状态、防接飞刀是读图对照，不改这条买卖。
>
> 输出目录：`$PLOTLY_ROOT/{YYYYMMDD}all_adaptive_stock`（例：`20260924all_adaptive_stock`）。  
> 不含 opt-in 的 `state_space_nested_cv`、Kalman 门控等实验方案。

---

## 1. 一句话

对池内每只股票：**只用训练截止日前的历史，按 B0 在 method×enter-up 参数网格上自选一套因果上涨规则**；交易规则固定为 **只做多上涨段**（进上涨买、离上涨卖）；样本外窗口用冻结配置因果出图，不再重选。

开跑前必须已有验证目录，以及 `--as-of` 所在月的状态信号分片（如 `2026-09.signals.csv`）。文件不在就直接退出。

---

## 2. 入口与默认参数

| 项 | 默认值 | 说明 |
|---|---|---|
| 命令 | `ashare-daily adaptive` | 先检查验证目录和当月信号分片，再调用 `src/ashare_daily/plots/plot_adaptive_stage_pool.py` |
| `REGIME_MODE` | `legacy_score_method` | B0 选法（生产） |
| `TRADE_MODE` | `hold_up` | 只做多上涨段 |
| `--plot-start` / `--train-cutoff` | `2026-04-01` | 不传 `--train-cutoff` 时，训练截止等于 `--plot-start`。训练用 `< cutoff`；图窗从 start 到 `--as-of` |
| `--score-mode` | `expect_no_trade` | **仅** `bayes_rules` 模式用；`hold_up` 下不扫买卖规则 |
| 标的池 | `CLUSTER_MAPPING_SELECTED_TXT` | 聚类选出的股票。`POOL_TARGET_COUNT` 默认 100 |
| 输出 | `$PLOTLY_ROOT/{YYYYMMDD}all_adaptive_stock/` | HTML + `configs/` + trades + `batch_summary.csv`。默认同目录再写 `ticket_card.csv` |
| 训练缓存 | `$PLOTLY_ROOT/_train_cache` | 同一 cache tag 命中则不重训。`--retrain` 强制重训 |
| `--fair-path-extra-trail-years` | `1,0.5,0.25,1/12` | 图8=1年 / 图9=6个月 / 图10=3个月 / 图11=1个月（对照图7=2年）；`off` 关掉 |

`listing` 不在这条命令里。它另读本目录的 `configs/`，把上市以来 HTML 写到 `$PLOTLY_ROOT/{YYYYMMDD}_from_listing_stock`，并在那里给图7–图11 加 EWMA 轨、默认加图12。

每日命令示例：

```bash
ashare-daily adaptive --as-of 2026-09-24
# 强制重训：
ashare-daily adaptive --as-of 2026-09-24 --retrain
# 单票：
ashare-daily adaptive --as-of 2026-09-24 --codes SZ002594
# 不要多尺度图8–11：
ashare-daily adaptive --as-of 2026-09-24 --fair-path-extra-trail-years off
# 不要 ticket card：
ashare-daily adaptive --as-of 2026-09-24 --no-ticket-card
```

---

## 3. 端到端流程

```text
ashare-daily adaptive
  ├─ require 验证目录 + {YYYY-MM}.signals.csv
  └─ plot_adaptive_stage_pool.py
       Pass1  train_one_symbol
              └─ select_regime_method(..., mode=legacy_score_method)  # B0
       Pass2  冻结 REGIME_HOLD_UP_RULES → configs/{code}.json
       OOS    oos_one：因果打标签 → hold_up 模拟 → HTML / trades / batch_summary
  └─ build_daily_ticket_card.py → ticket_card.csv
```

### 时间切分（防泄漏）

| 区间 | 用途 |
|---|---|
| `as_of <= train_cutoff − 1 日` | **训练 / 选方法**（B0 打分） |
| `start_date → end_date`（默认 4/1 → AS_OF） | **样本外执行与出图**；配置已冻结，不重选 |

缓存 tag 含：`train_cutoff`、`trade_mode`、`score_mode`、`regime_mode`、`TRUTH_VERSION`、`METHOD_IMPL_VERSION`。  
`METHOD_IMPL_VERSION` 现为 `per_code_enter_up_params_v1`。改 METHODS 实现或这个版本号后，旧缓存对不上，需 `--retrain`。

---

## 4. Pass1：如何选出「这只票的上涨阶段规则」

实现：`src/ashare_daily/plots/adaptive_stage_common.py` 的 `select_regime_method_legacy`（B0）。

> **当前语义：** 每票只用自己的训练窗，在 **method × enter-up 参数网格** 上 argmax；  
> 不再用全池统一门槛 / `HOLD_UP_TIEBREAK`。胜出的 `method` + `method_params` 写入 config，OOS 冻结执行。

### 4.1 候选方法（因果）

来自 `src/ashare_daily/regime/eval_causal_regime_switch_pool.py` 的 `METHODS`：

| 方法 | 角色（简述） | 可调 enter-up 参数（网格） |
|---|---|---|
| `ma_stack_hyst` | MA20/MA60 堆叠 + 破/站 MA20 滞后确认 | `confirm`、`smooth`、`min_seg` |
| `ma_stack_struct` | 结构更粘：未破 MA60 的回撤可仍算上涨 | `confirm`、`smooth`、`min_seg` |
| `ma_stack_strict` | 更严 prior 门槛；含受控 V 反转入口 | `p60_thr`、`p20_vrev`、`confirm`、`min_seg` |
| `dual_ma_cross` | 主要看 MA20 vs MA60 | `confirm`、`smooth`、`min_seg` |
| `hybrid_ma_adx` | MA 结果再经 ADX/prior 弱化假趋势 | `confirm`、`smooth`、`min_seg` |
| `adx_di` | ADX/DI 定方向（train 分数够高可入选） | `adx_thr`、`smooth`、`min_seg` |
| `prior60_band` | prior60 + 波幅带（train 分数够高可入选） | `min_seg` |

`HOLD_UP_METHOD_EXCLUDE` 为空：`adx_di` / `prior60_band` 不再被全池排除。

所有方法经 `run_method(..., params=...)` → `finalize_regime_labels`：

1. 低历史分位波动持续时偏置为 `range`（可被强动量逃逸）
2. 过短的方向段合并（默认 `min_seg=10`；可被票级网格改成 5）

空 `method_params={}` 时行为与历史默认门槛一致（strict：`p60=6%` / `p20_vrev=6%` / `confirm=3`）。

### 4.2 训练真值（只用于选法，不当实盘信号）

- `retrospective_truth(close)`：回顾式大幅摆动切主升/主跌，其余横盘（**含训练窗内前瞻**）。
- 仅在训练条上计算，不参与 OOS 标签。

### 4.3 打分公式（B0）

对每个候选 **(method, params)** 在**全段训练条**上算：

```text
score = 0.45 · hold_up_compound
      + 0.25 · agree(labels, truth)
      + 0.15 · up_frac
      − 0.015 · n_switches
```

其中 `hold_up_compound`：训练标签上「进 up 持有、离 up 平仓」的复利收益（含段末未平视同平仓的简化版，仅用于选法）。

**注意：** B0 **不直接优化** OOS 闭合交易胜率；胜率是样本外统计结果。选参 **不看** `train_cutoff` 之后的净值。这组权重没有在个股上单独重估，生产就用代码里的这组数。

### 4.4 硬约束与回退

1. **不再**全池排除 `adx_di` / `prior60_band`；某票 train 分数最高即可入选。
2. **退化真值**：回顾真值几乎全是横盘（`range≥90%` 或方向占比 `<5%`）→ 固定回退 `ma_stack_hyst`（空 params）。上市不久、训练窗摆不动时会走到这里。
3. **否决**（对每个 (method, params)）：
   - 预测几乎全横盘：`range≥85%`；或 `range≥75%` 且比真值多出很多（真值 range≥30% 时）；
   - **方向塌缩**：真值方向占比高，但方法很少给出方向。
4. **已取消** 近平局 `HOLD_UP_TIEBREAK`（不再全池偏向 `ma_stack_strict`）。

### 4.5 训练条不足

`train_bars < MIN_TRAIN_BARS`（120）→ `skipped_train=true`，Pass2 用池内众数方法或固定 hybrid 兜底。

### 4.6 sticky-up 护栏

生产 B0 **默认不加** `apply_sticky_up_guard`。护栏只在 `regime_mode=ensemble_cv` 或选中方法名为 `ensemble` 时打开。  
OOS 标签 = 冻结 `method` + `method_params` 的因果 `run_method` 输出。

---

## 5. Pass2：冻结交易规则（hold_up）

生产不扫买卖点库，规则写死为 `REGIME_HOLD_UP_RULES`：

| 阶段 | 买 | 卖 |
|---|---|---|
| up | `进入上涨态` | `离上涨态` |
| down | `NO_TRADE` | — |
| range | `NO_TRADE` | — |

含义：

- **买入**：标签从非 up → up 的当日收盘（进入上涨态）
- **卖出**：标签从 up → 非 up 的当日（离上涨态）
- 下跌 / 横盘：**不交易**
- 样本外模拟**不画期末强平卖点**；未平仓单独标记（菱形 / `open_*`）

单边成本常量 `COST_PER_SIDE=0.001`（规则评分路径用；hold_up 主路径 edge 以模拟账面为准）。个股涨跌停和停牌不另改这条规则。

写入 `configs/{code}.json` 的关键字段：

- `method` / `method_params`（票级 enter-up 门槛；空 `{}` = 方法默认）
- `method_scores`、`method_select_meta`（含 `joint_param_search`、`grid_tried`）
- `regime_mode=legacy_score_method`、`trade_mode=hold_up`
- `train_cutoff` / `train_end` / `train_bars`
- `rules.{up,down,range}`
- OOS 后附加 `oos.{compound,bh,edge,n_closed,...}`

---

## 6. OOS：冻结执行与出图

`oos_one`：

1. 用冻结 `method` 对完整 OHLCV 因果打 `regime`（含 warmup）。K 线默认做前复权。
2. 裁到图窗 `[start_date, end_date]`。
3. `simulate_adaptive_book` 按 hold_up 规则模拟。
4. 对比列：同窗固定 `hybrid_ma_adx` + `FIXED_HYBRID_RULES`（仅对照，不改生产标签）。
5. Plotly HTML + `trades/{code}_trades.csv` / `_segments.csv`；可选早期视觉标记（`EARLY_REGIME_MOVE=6%`，**不改交易**）。

汇总：`batch_summary.csv`（每票 method、compound、bh、edge、n_closed、win、…）。  
默认同目录再写 `ticket_card.csv`（`--no-ticket-card` 可关）。

这条命令画出的图7–图11 是 OLS 路径和 expanding P95。EWMA 轨是 `listing` 后处理加上的，不在 `all_adaptive_stock` 里。见 [路径说明.md](路径说明.md)。

---

## 7. 与「不是生产」的路径对照

| 路径 | `REGIME_MODE` / 说明 | 生产默认？ |
|---|---|---|
| **B0 baseline（本文）** | `legacy_score_method` + `hold_up` | **是** |
| B4 集成 | `ensemble_cv`（标签更好，曾未过交易 edge 门） | 否 |
| 状态空间锦标赛 | `state_space_nested_cv` | 否（opt-in） |
| 旧贝叶斯买卖规则 | `TRADE_MODE=bayes_rules` | 否 |

---

## 8. 关键代码索引

| 模块 | 路径 |
|---|---|
| 每日入口 | `ashare-daily adaptive` → `src/ashare_daily/commands/adaptive.py` |
| 编排 | `src/ashare_daily/plots/plot_adaptive_stage_pool.py` |
| B0 选法 / hold_up / 模拟 | `src/ashare_daily/plots/adaptive_stage_common.py` |
| 因果 METHODS / 真值 / finalize | `src/ashare_daily/regime/eval_causal_regime_switch_pool.py` |
| 出图与 OHLCV | `src/ashare_daily/plots/plot_regime_transition_example.py` |
| ticket card | `src/ashare_daily/plots/build_daily_ticket_card.py` |

---

## 9. 设计要点（读代码时易混）

1. **选法目标 ≠ 交易胜率**：B0 优化的是训练窗上的 hold_up 复利 + 与回顾真值一致等；OOS `win` 是结果指标。
2. **回顾真值有前瞻**：只许出现在训练选法，不许当实盘标签。
3. **交易完全由标签驱动**：hold_up 下没有「绿牛/金叉」等二次买卖库；改交易表现应先改分段法或选法，而不是改 Pass2 规则。
4. **同 cutoff 可缓存**：改 METHODS 实现或 `METHOD_IMPL_VERSION` 后需 `--retrain`，否则可能复用旧 configs。
5. **读图辅助不改三角**：图7–11 checklist、六状态、防接飞刀都不覆盖绿三角 / 红三角。见 [图7-11公平路径上下轨辅助买卖.md](图7-11公平路径上下轨辅助买卖.md)。
