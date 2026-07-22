# quant_bot_v35 — Project Context for Claude Code

## What this is
MEXC crypto trading bot (BTC/USDT spot). Single file:
`quant_bot_v35.py` (~7000 lines). Flask web UI + async MEXC loop.

## Architecture

```
fetch_ohlcv_large()               ← Binance keyless → Bybit → OKX → MEXC fallback
     ↓
TabularLightGBMPipeline           ← OHLCV features → LightGBM → calibrated gate
     ↓
BarrierBacktester                 ← triple-barrier labels, execution-aligned sim
     ↓
PurgedWalkForwardEngine (nested)  ← 5-fold inner WFO + locked 20% holdout
     ↓
report: funnel / shadow_book / baseline / go-no-go
```

## Critical: LightGBM API

**MUST** use native Booster API. scikit-learn is NOT installed.

```python
# Correct
self.model = lgb.train(params, lgb.Dataset(X, label=y), ...)
proba = self.model.predict(X)          # returns 1-D float array

# Wrong — will crash at import
lgb.LGBMClassifier(...)               # requires scikit-learn
model.predict_proba(X)[:, 1]          # sklearn API
```

## Branch
Active branch: `claude/session-swf0zb`
Push: `git push -u origin claude/session-swf0zb`

## Key constants (lines ~65–250)

| Name | Value | Notes |
|------|-------|-------|
| `RESEARCH_BARS_DEFAULT` | 35040 | 1 year at 15m; env-overridable |
| `RESEARCH_BARS_MAX` | 500 000 | covers Binance 15m BTC since 2017 |
| `DEFAULT_MODEL_PARAMS["HOLD_BARS"]` | 26 | **15m horizon = 6.5 h ≈ 1σ** |
| `DEFAULT_MODEL_PARAMS["TP_PERCENT"]` | 1.0 | 1% TP, >3× roundtrip cost |
| `DEFAULT_MODEL_PARAMS["SL_PERCENT"]` | 0.5 | 0.5% SL |
| `bot_state["timeframe"]` | "15m" | UI default (15m button is `.on`) |

## Why 15m (not 1m)

1m at BTC: σ ≈ 0.05%/bar, HOLD=13 bars → TP target ≈ 5σ → label rate <2%.
LightGBM correctly learns that almost all signals are bad (p_meta ≈ 0.12 OOS).

15m: σ ≈ 0.19%/bar, HOLD=26 bars → expected move ≈ 0.97% ≈ **1σ** → label rate ~25-40%.
Also: 100k bars at 15m = 2.8 years (vs 70 days at 1m) → real regime diversity.

## Pending experiments

### Deney 1 — 15m + HOLD=26 (baseline check)
- UI: select 15m, run backtest with ≥35 000 bars
- Watch: positive label rate (want 20–45%), OOS AUC (want ≥0.53)
- If positive label rate misses: try HOLD=40 (10 h) or TP=0.8%

### Deney 2 — zero-commission simulation
- Same 15m setup, set `commission_pct = 0` in Trading Params
- Shows theoretical ceiling without exchange costs

### Deney 3 — trailing exit (if Deney 2 shows edge)
- Switch meta-labeling to expectancy target + trailing exit
- Requires code change in BarrierBacktester

## Go / No-Go criteria for live deployment

| Metric | Threshold |
|--------|-----------|
| OOS AUC | ≥ 0.53 |
| IS-OOS AUC gap | ≤ 0.08 |
| Edge percentile (vs random baseline) | ≥ 95th |
| Shadow book net | negative (rejected signals should lose) |
| Positive label rate | 20–45% |

## Running tests

```bash
python quant_bot_v35.py --test           # full suite (~60 s with LightGBM folds)
python quant_bot_v35.py --test-fast      # faster subset (skips WFO folds)
```

Test to watch after any geometry/pipeline change: `test_tabular_backtest_and_nested_wfo`

## WFO grid (PurgedWalkForwardEngine._grid)

Structures list (TP%, SL%, HOLD) — line ~3470:
```
base, (0.8,0.4,20), (1.0,0.5,30), (0.9,0.45,26),
(1.2,0.5,30), (1.4,0.6,40), (1.1,0.4,20), (0.8,0.6,40), (1.2,0.6,20)
```
`(0.9, 0.45, 26)` targets the 15m-optimal horizon.

## UI routes

| Route | Purpose |
|-------|---------|
| `/` | Main dashboard |
| `/api/backtest` | POST `{bars: N}` → runs `run_tabular_backtest` |
| `/api/run_wfo` | POST `{bars: N}` → runs `PurgedWalkForwardEngine` |
| `/api/config` | GET/POST trading config |
| `/api/promote_challenger` | Promote WFO champion to live |
| `/api/set_timeframe` | POST `{tf: "15m"}` |

## Plain-text mirror

After every significant code change, refresh the mirror:
```bash
cp quant_bot_v35.py quant_bot_v35_code.txt
```
This file is committed alongside the `.py` for diff-less review.

## Common pitfalls

- `history_exhausted` flag: if `df.attrs["history_exhausted"]` is True,
  `fetch_ohlcv_large` returns early without trying fallback providers.
- WFO needs ≥3000 bars. Nested WFO needs ≥3000 in the 80% dev split.
  Ember: `ValueError: Nested purged WFO needs at least 3000 bars`.
- Paper-entry price: `entry_price = price × (1 + (slippage_pct + spread_pct/2)/100)`.
  Tests must compute this dynamically from cfg(), not hardcode 60030.
- OBI history attach: `attach_obi_history(df, timeframe)` must be called after
  fetching but before `run_tabular_backtest`.
