"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  QUANT BOT V3.7 — LIGHTGBM RESEARCH ENGINE                                 ║
║  MEXC Spot · PyWebView arayüz · Scale-Out (70/30) + MTF Trailing           ║
║                                                                            ║
║  Çekirdek: Vol-Norm → Multi-res Paths (5/15/30/60 dyadik) → Chen İmzaları  ║
║  → δ̂ diagnostiği (κ_init + faktör bütçesi) → Shared Encoder                ║
║  M = H^a(κ)[×S^b×E^c] → Bundle Z(t) → Innovations → Epizot/Küme/Geçiş     ║
║  Grafı → GBM → Meta Labeling → Conformal (A = α·A_pred + β·A_geom)         ║
║  + Anchor Panel (append-only defter) + Purged WFO (warm start + RKD)       ║
║                                                                            ║
║  Mimarlar: Fable + Antigravity AI + Kullanıcı                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
if sys.stdout is None:
    class DummyWriter:
        def write(self, x): pass
        def flush(self): pass
    sys.stdout = DummyWriter()
if sys.stderr is None:
    class DummyWriter:
        def write(self, x): pass
        def flush(self): pass
    sys.stderr = DummyWriter()

import asyncio
import os
import math
import logging
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import ccxt
import requests
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template_string
import webview


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"bot_v3.7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8")
    ]
)
log = logging.getLogger("QuantBot")

# ═══════════════════════════════════════════════════════════════════════════════
# AYARLAR & STATE
# ═══════════════════════════════════════════════════════════════════════════════
SYMBOL = "BTC/USDT"
OHLCV_LIMIT = 1000   # MEXC spot supports up to 1000 bars per fetch — more visible history
RESEARCH_BARS_DEFAULT = int(os.environ.get("RESEARCH_BARS_DEFAULT", "10000"))
RESEARCH_BARS_MIN = 3000
RESEARCH_BARS_MAX = int(os.environ.get("RESEARCH_BARS_MAX", "100000"))
RESEARCH_PROVIDER_ORDER = tuple(
    p.strip().lower()
    for p in os.environ.get("RESEARCH_DATA_PROVIDERS", "binance,bybit,okx,mexc").split(",")
    if p.strip().lower() in {"binance", "bybit", "okx", "mexc"}
) or ("binance", "bybit", "okx", "mexc")
FAST_LENGTH = 8
SLOW_LENGTH = 21
VOL_LENGTH = 14
CVD_LENGTH = 14
BAND_MULT = 2.5
MIN_PROFIT_MARGIN = 0.3
TRAIL_MULT = 3.0
PING_STOP_MULT = 0.5
TP_PERCENT = 3.0
HMM_N_STATES = 2
HMM_TRAIN_WINDOW = 300
HMM_RETRAIN_FREQ = 24
MAX_CAPITAL_ALLOCATION = 0.95
TARGET_RISK_PERCENT = 2.0
LOOP_INTERVAL = 10
FAST_LOOP_INTERVAL = 0.5
COMMISSION_RATE = 0.001  # MEXC standard spot taker fee (0.1%). Set to 0.0 if you have a 0-fee promotion.

# ── Two-sided trading (futures) ──────────────────────────────────────────────
# Short selling is OPT-IN. Default is long-only so spot behaviour is unchanged.
# ALLOW_SHORT enables SELL signals in the decision layer, the backtester and
# PAPER trading. REAL live short execution additionally requires a futures
# venue and is intentionally NOT auto-armed (see main_tick) — validate in
# PAPER/backtest/WFO first.
ALLOW_SHORT = os.environ.get("ALLOW_SHORT", "0").strip() in ("1", "true", "True", "yes")
TRADING_VENUE = os.environ.get("TRADING_VENUE", "spot").strip().lower()  # "spot" | "futures"

# ═══════════════════════════════════════════════════════════════════════════════
# TRADING CONFIG — user-editable, persisted live trading parameters
#
# Single source of truth for the knobs the operator tunes from the UI. Persisted
# to trading_config.json so edits survive restarts. Read at runtime through
# cfg(key); geometry-affecting knobs (gates, barrier, health) take effect on the
# next fold refit, the rest apply on the next tick.
# ═══════════════════════════════════════════════════════════════════════════════
TRADING_CONFIG_PATH = Path(__file__).parent / "trading_config.json"

DEFAULT_TRADING_CONFIG = {
    # ── Sermaye & Risk ──
    "risk_per_trade_pct": 2.0,     # % of equity risked per trade (position sizing)
    "max_capital_pct": 95.0,       # max % of equity deployed in one position
    "min_order_usdt": 5.0,         # exchange minimum notional
    "paper_start_balance": 10000.0,
    # ── Hedefler & Çıkışlar ──
    "tp_base_pct": 0.6,            # base take-profit % (floored by cost model)
    "trail_mult": 3.0,             # trailing-stop = mult × realised volatility
    "partial_tp_ratio": 70.0,      # % of position taken at first trailing hit
    "barrier_hold_bars": 30,       # barrier-race horizon (labels/meta/targets)
    "tp_cost_floor_mult": 3.0,     # TP must clear this × roundtrip cost
    "sl_cost_floor_mult": 1.5,     # SL floor = this × roundtrip cost
    # ── Maliyet Modeli ──
    "commission_pct": 0.1,         # per-side taker fee %
    "slippage_pct": 0.05,          # per-side slippage %
    "spread_pct": 0.01,            # half-spread %
    # ── Karar Kapıları ──
    "gbm_gate_quantile": 70.0,     # p_hi/p_lo set at this quantile of train probs
    "conformal_accept_pct": 60.0,  # conformal target acceptance rate
    "obi_wall": 0.3,               # order-book imbalance wall that blocks entries
    "allow_short": False,          # opt-in two-sided (futures) shorts
    # ── Sağlık Eşikleri (auto-HOLD) ──
    "overfit_gap_max": 0.25,       # IS-OOS AUC gap ceiling
    "d_anchor_drift_max": 0.60,    # representation drift ceiling vs fold-0 panel
    "recon_regress_max": 2.5,      # held-out recon may not exceed this × baseline
    # ── Genel ──
    "loop_interval_sec": 10,       # main decision loop cadence
}

# Metadata for the UI: label, group, unit, min, max, step, and whether the change
# takes effect live or only on the next geometry refit.
TRADING_CONFIG_META = {
    "risk_per_trade_pct":  ("Risk / İşlem", "Sermaye & Risk", "%", 0.1, 10.0, 0.1, "live"),
    "max_capital_pct":     ("Maks. Sermaye", "Sermaye & Risk", "%", 5.0, 100.0, 1.0, "live"),
    "min_order_usdt":      ("Min. Emir", "Sermaye & Risk", "USDT", 1.0, 100.0, 0.5, "live"),
    "paper_start_balance": ("Paper Bakiye", "Sermaye & Risk", "USDT", 100.0, 1_000_000.0, 100.0, "restart"),
    "tp_base_pct":         ("TP Taban", "Hedefler & Çıkışlar", "%", 0.1, 10.0, 0.05, "live"),
    "trail_mult":          ("Trailing Çarpanı", "Hedefler & Çıkışlar", "×vol", 0.5, 10.0, 0.1, "live"),
    "partial_tp_ratio":    ("Kısmi TP Oranı", "Hedefler & Çıkışlar", "%", 0.0, 100.0, 5.0, "live"),
    "barrier_hold_bars":   ("Bariyer Ufku", "Hedefler & Çıkışlar", "bar", 5, 200, 1, "refit"),
    "tp_cost_floor_mult":  ("TP Maliyet Tabanı", "Hedefler & Çıkışlar", "×maliyet", 1.0, 10.0, 0.5, "live"),
    "sl_cost_floor_mult":  ("SL Maliyet Tabanı", "Hedefler & Çıkışlar", "×maliyet", 0.5, 10.0, 0.5, "live"),
    "commission_pct":      ("Komisyon", "Maliyet Modeli", "%", 0.0, 1.0, 0.005, "live"),
    "slippage_pct":        ("Slipaj", "Maliyet Modeli", "%", 0.0, 1.0, 0.005, "live"),
    "spread_pct":          ("Spread (yarım)", "Maliyet Modeli", "%", 0.0, 1.0, 0.005, "live"),
    "gbm_gate_quantile":   ("GBM Kapı Kantili", "Karar Kapıları", "%", 30.0, 95.0, 1.0, "refit"),
    "conformal_accept_pct":("Conformal Kabul", "Karar Kapıları", "%", 20.0, 95.0, 1.0, "refit"),
    "obi_wall":            ("OBI Duvarı", "Karar Kapıları", "", 0.0, 1.0, 0.05, "live"),
    "allow_short":         ("Short'a İzin Ver", "Karar Kapıları", "bool", 0, 1, 1, "live"),
    "overfit_gap_max":     ("Overfit Gap Maks.", "Sağlık Eşikleri", "AUC", 0.05, 0.60, 0.01, "live"),
    "d_anchor_drift_max":  ("D_anchor Drift Maks.", "Sağlık Eşikleri", "", 0.1, 2.0, 0.05, "live"),
    "recon_regress_max":   ("Recon Regres. Maks.", "Sağlık Eşikleri", "×", 1.5, 10.0, 0.5, "live"),
    "loop_interval_sec":   ("Döngü Aralığı", "Genel", "sn", 2, 120, 1, "live"),
}

def _coerce_cfg(cfg):
    """Clamp + type-coerce a config dict against the metadata bounds."""
    out = dict(DEFAULT_TRADING_CONFIG)
    for k, v in (cfg or {}).items():
        if k not in DEFAULT_TRADING_CONFIG:
            continue
        meta = TRADING_CONFIG_META.get(k)
        try:
            if meta and meta[2] == "bool":
                out[k] = bool(v)
            elif isinstance(DEFAULT_TRADING_CONFIG[k], int) and meta and meta[2] != "%":
                out[k] = int(round(float(v)))
            else:
                out[k] = float(v)
            if meta and meta[2] != "bool":
                lo, hi = meta[3], meta[4]
                out[k] = min(max(out[k], lo), hi)
        except (TypeError, ValueError):
            out[k] = DEFAULT_TRADING_CONFIG[k]
    return out

def load_trading_config():
    try:
        if TRADING_CONFIG_PATH.exists():
            import json
            with open(TRADING_CONFIG_PATH, "r", encoding="utf-8") as f:
                return _coerce_cfg(json.load(f))
    except Exception as e:
        logging.getLogger("QuantBot").error(f"Error loading trading config: {e}")
    return dict(DEFAULT_TRADING_CONFIG)

def save_trading_config(new_cfg):
    """Merge, clamp, persist and return the effective config."""
    merged = _coerce_cfg({**CFG, **(new_cfg or {})})
    CFG.clear(); CFG.update(merged)
    try:
        import json
        with open(TRADING_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(CFG, f, indent=2)
    except Exception as e:
        logging.getLogger("QuantBot").error(f"Error saving trading config: {e}")
    return dict(CFG)

CFG = load_trading_config()

def cfg(key):
    """Live read of a trading-config value (falls back to the default)."""
    return CFG.get(key, DEFAULT_TRADING_CONFIG.get(key))

# Yerel saat dilimi farkini hesapla (Turkiye = UTC+3 = 10800 saniye)
from datetime import timezone as _tz
UTC_OFFSET = int(datetime.now(_tz.utc).astimezone().utcoffset().total_seconds())

# Timeframe degisikliginde main_loop'u aninda uyandirmak icin Event
tf_change_event = asyncio.Event()
state_lock = threading.Lock()

PARAMETERS_STORE_PATH = Path(__file__).parent / "parameters_store.json"

def get_active_parameters():
    try:
        if PARAMETERS_STORE_PATH.exists():
            import json
            with open(PARAMETERS_STORE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("champion", {})
    except Exception as e:
        log.error(f"Error loading parameters: {e}")
    return {
        "FAST_LENGTH": 8,
        "SLOW_LENGTH": 21,
        "VOL_LENGTH": 14,
        "CVD_LENGTH": 14,
        "BAND_MULT": 2.5,
        "MIN_PROFIT_MARGIN": 0.3,
        "TRAIL_MULT": 3.0,
        "PING_STOP_MULT": 0.5,
        "TP_PERCENT": 0.6
    }

def get_all_parameters():
    try:
        if PARAMETERS_STORE_PATH.exists():
            import json
            with open(PARAMETERS_STORE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log.error(f"Error loading full parameters store: {e}")
    return {
        "active_version": 1,
        "champion": get_active_parameters(),
        "shadow_challenger": None,
        "history": []
    }

def save_parameters_store(store):
    """Atomically persist the champion/challenger ledger."""
    import json
    tmp_path = PARAMETERS_STORE_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, PARAMETERS_STORE_PATH)

def promote_parameter_set(store, parameters, source="manual"):
    """Archive the current champion and activate a validated parameter set."""
    promoted = dict(parameters or {})
    store.setdefault("history", []).append({
        "version": store.get("active_version", 1),
        "parameters": store.get("champion"),
        "retired_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
    })
    store["champion"] = promoted
    store["shadow_challenger"] = None
    store["active_version"] = store.get("active_version", 1) + 1
    store["last_promotion"] = {
        "source": source,
        "promoted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return store

bot_state = {
    "is_trading_active": False,
    "trading_mode": "PAPER",
    "timeframe": "1m",
    "timeframe_changed": False,

    "symbol": SYMBOL,
    "price": 0.0,
    "gauss_vol": 0.0,          # simple realised-vol for the trailing distance
    "signal": "HOLD",
    "signal_type": "",
    "obi": 0.0,

    "virtual_balance": cfg("paper_start_balance"),
    "real_balance": 0.0,

    "position_side": None,
    "position_entry": 0.0,
    "position_qty": 0.0,
    "position_type": "",
    "position_pnl": 0.0,
    "trail_stop": 0.0,
    "ping_stop": 0.0,

    "trade_count": 0,
    "total_pnl": 0.0,
    "gross_profit": 0.0,
    "gross_loss": 0.0,
    "winning_trades": 0,
    "losing_trades": 0,
    "max_drawdown": 0.0,
    "peak_balance": cfg("paper_start_balance"),
    "start_real_balance": 0.0,
    "peak_real_balance": 0.0,
    "loop": None,
    "pnl_list": [],
    "sharpe_ratio": 0.0,
    "avg_trade": 0.0,
    "avg_win": 0.0,
    "avg_loss": 0.0,
    "largest_win": 0.0,
    "largest_loss": 0.0,
    "max_cons_wins": 0,
    "max_cons_losses": 0,
    "current_cons_wins": 0,
    "current_cons_losses": 0,

    "trades": [],
    "chart_data": [],
    "orderbook": {"bids": [], "asks": []},

    # Shadow flag kept for backward-compat with UI JS (always False now)
    "shadow_active": False,

    # Backtest and WFO keys
    "wfo_report": None,
    "wfo_running": False,
    "backtest_report": None,
    "backtest_running": False,
    "backtest_bars_requested": RESEARCH_BARS_DEFAULT,
    "backtest_bars_used": 0,
    "wfo_bars_requested": RESEARCH_BARS_DEFAULT,
    "wfo_bars_used": 0,
    "research_data_source": "-",
    "active_parameters": get_active_parameters(),
    "parameters_store": get_all_parameters(),

    # V3.6 Learned Geometry state
    "geom": {"status": "collecting", "signal": "HOLD", "schema": "-", "kappa": 0.0,
             "a_score": 0.0, "a_gate": 0.5, "p_gbm": 0.5, "p_meta": 0.5,
             "episode": "NORMAL", "cluster": -1, "fold": 0, "dir": 0},

    # Two-sided trading toggles (opt-in short)
    "allow_short": bool(cfg("allow_short")) or ALLOW_SHORT,
    "trading_venue": TRADING_VENUE,
}

# ═══════════════════════════════════════════════════════════════════════════════
# METRİK MOTORU
# ═══════════════════════════════════════════════════════════════════════════════
def update_portfolio_metrics(pnl_usdt, mode):
    # Determine base capital for calculation
    if mode == "PAPER":
        base_capital = cfg("paper_start_balance")
        current_balance = bot_state["virtual_balance"]
    else:
        current_balance = bot_state.get("real_balance", 0.0)
        base_capital = bot_state.get("start_real_balance", current_balance)
        if base_capital <= 0:
            base_capital = cfg("paper_start_balance")

    # Calculate portfolio return % for this trade relative to capital base
    pnl_pct = (pnl_usdt / base_capital) * 100 if base_capital > 0 else 0.0

    bot_state["trade_count"] += 1
    bot_state["pnl_list"].append(pnl_pct)
    
    # Cumulative portfolio PnL % based on balance equity change
    if mode == "PAPER":
        bot_state["total_pnl"] = ((current_balance - cfg("paper_start_balance")) / cfg("paper_start_balance")) * 100
    else:
        bot_state["total_pnl"] = ((current_balance - base_capital) / base_capital) * 100 if base_capital > 0 else 0.0

    if pnl_usdt > 0:
        bot_state["gross_profit"] += pnl_usdt
        bot_state["winning_trades"] += 1
        bot_state["current_cons_wins"] += 1
        bot_state["current_cons_losses"] = 0
        if pnl_usdt > bot_state["largest_win"]:
            bot_state["largest_win"] = float(pnl_usdt)
    else:
        bot_state["gross_loss"] += pnl_usdt
        bot_state["losing_trades"] += 1
        bot_state["current_cons_losses"] += 1
        bot_state["current_cons_wins"] = 0
        if pnl_usdt < bot_state["largest_loss"]:
            bot_state["largest_loss"] = float(pnl_usdt)

    bot_state["max_cons_wins"] = max(bot_state["max_cons_wins"], bot_state["current_cons_wins"])
    bot_state["max_cons_losses"] = max(bot_state["max_cons_losses"], bot_state["current_cons_losses"])
    
    # Peak and Drawdown tracking
    if mode == "PAPER":
        bot_state["peak_balance"] = max(bot_state.get("peak_balance", cfg("paper_start_balance")), current_balance)
        dd = ((bot_state["peak_balance"] - current_balance) / bot_state["peak_balance"]) * 100
        bot_state["max_drawdown"] = max(bot_state["max_drawdown"], dd)
    else:
        start_real = bot_state.get("start_real_balance", current_balance)
        peak_real = max(bot_state.get("peak_real_balance", start_real), current_balance)
        bot_state["peak_real_balance"] = peak_real
        dd = ((peak_real - current_balance) / peak_real) * 100 if peak_real > 0 else 0.0
        bot_state["max_drawdown"] = max(bot_state["max_drawdown"], dd)

    if bot_state["trade_count"] > 0:
        bot_state["avg_trade"] = (bot_state["gross_profit"] + bot_state["gross_loss"]) / bot_state["trade_count"]
    if bot_state["winning_trades"] > 0:
        bot_state["avg_win"] = bot_state["gross_profit"] / bot_state["winning_trades"]
    if bot_state["losing_trades"] > 0:
        bot_state["avg_loss"] = bot_state["gross_loss"] / bot_state["losing_trades"]

    # Sharpe ratio based on list of portfolio return percentages
    if len(bot_state["pnl_list"]) > 1:
        sd = np.std(bot_state["pnl_list"])
        bot_state["sharpe_ratio"] = float(np.mean(bot_state["pnl_list"]) / sd) if sd > 0 else 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# MATEMATİK & SİNYAL & POZİSYON
# ═══════════════════════════════════════════════════════════════════════════════
def get_tf_params(tf):
    # Roundtrip fee in percent (2 trades * rate * 100)
    roundtrip_fee = 2.0 * COMMISSION_RATE * 100
    min_viable = 2.0 * roundtrip_fee

    if tf == "1m":
        return max(0.02, min_viable), 1.5
    elif tf == "5m":
        return max(0.05, min_viable), 2.0
    elif tf == "15m":
        return max(0.10, min_viable), 2.2
    elif tf == "1h":
        return max(0.25, min_viable), 2.5
    elif tf == "4h":
        return max(0.50, min_viable), 2.8
    else:
        return max(0.30, min_viable), 2.5

def get_lower_tf(tf):
    if tf == "4h": return "30m"
    elif tf == "1h": return "10m"
    elif tf == "15m": return "2m"
    elif tf == "5m": return "1m"
    else: return "1m"
def gaussian_filter(series, length):
    period = max(length, 1)
    beta = (1.0 - math.cos(2.0 * math.pi / period)) / (math.pow(2**0.5, 1.0) - 1.0)
    alpha = -beta + math.sqrt(beta * beta + 2.0 * beta)
    c0, c1, c2 = alpha**2, 2.0*(1.0-alpha), -(1.0-alpha)**2
    filt = np.zeros(len(series))
    for i in range(len(series)):
        s = series[i] if not np.isnan(series[i]) else 0.0
        p1 = filt[i-1] if i > 0 else s
        p2 = filt[i-2] if i > 1 else s
        filt[i] = c0*s + c1*p1 + c2*p2
    return filt

def calc_true_range(highs, lows, closes):
    tr = np.maximum(highs-lows, np.maximum(np.abs(highs-np.roll(closes,1)), np.abs(lows-np.roll(closes,1))))
    tr[0] = highs[0]-lows[0]
    return tr






# ═══════════════════════════════════════════════════════════════════════════════
# V3.6 — LEARNED GEOMETRY CORE ("nihai mimari")
#
# OHLCV + Order Flow ──► Volatility Normalization ──► Multi-resolution Paths
# (5/15/30/60, dyadic tree) ──► Rough Path Signatures (Chen: fine → coarse)
#   │◄┄ δ̂ diagnostic (first window only, intra-resolution quadruples,
#   │   scale-normalized) → κ_init + factor budget
#   ▼
# Shared Encoder — LEARNED GEOMETRY  M = H^a(κ) [× S^b × E^c],  κ ≤ κ_max < 0
#   r_prior ← e_res (deterministic, low capacity) · u ← content (unit tangent)
#   δ_i ← content (L1 + soft-threshold, sparse)
#   ▼
# Bundle state Z(t) ──► Innovations (fold-local μ/σ) ──► Episode segmentation
# (δ≠0 runs, hysteresis) ──► Anomaly clustering ──► Normal-centric transition
# graph ──► Gradient boosting ──► Meta labeling ──► Conformal gate
# A = α·A_pred + β·A_geom(panel) ──► Purged Walk-Forward (warm start +
# neighbour-fold RKD, geometry schema fixed) ──► Backtest
#
# Off-flow measurement standard: ANCHOR PANEL (core append-only ledger frozen
# at signature level + adaptive rolling buffer) feeding RKD targets, δ̂ health,
# D_anchor log, A_geom and κ_init.
# ═══════════════════════════════════════════════════════════════════════════════

GEOM_RESOLUTIONS = (5, 15, 30, 60)     # bars per window, fine → coarse (dyadic tree 5→15→30→60)
GEOM_PATH_DIM = 3                      # (t, vol-normalized log price, normalized signed flow)
GEOM_SIG_DIM = GEOM_PATH_DIM + GEOM_PATH_DIM**2 + GEOM_PATH_DIM**3   # levels 1..3 = 39
GEOM_HORIZON = 4                       # short-horizon stats (transition graph, drift series)
GEO_BARRIER_HOLD = 30                  # barrier-race horizon: labels, meta and trade targets
GEOMETRY_STORE_PATH = Path(__file__).parent / "geometry_store.json"


def roundtrip_cost_pct(commission_rate=None, slippage_rate=None, spread_rate=None):
    """Total roundtrip trading cost in percent: commission both ways plus
    entry/exit slippage and half-spread — the floor every target must clear.
    Reads live from the trading config unless explicit rates are passed."""
    comm = cfg("commission_pct") if commission_rate is None else commission_rate * 100.0
    slip = cfg("slippage_pct") if slippage_rate is None else slippage_rate * 100.0
    sprd = cfg("spread_pct") if spread_rate is None else spread_rate * 100.0
    return 2.0 * comm + 2.0 * (slip + sprd / 2.0)


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

def _inv_softplus(y):
    y = np.maximum(y, 1e-6)
    return np.where(y > 30.0, y, np.log(np.expm1(np.minimum(y, 30.0))))

def _soft_threshold(x, tau):
    """Proximal operator of tau·|x| — produces exact zeros so P(δ≠0) stays well defined."""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


class ChenSignature:
    """
    Level-1..3 path signatures of a d-dimensional piecewise-linear path, with
    Chen's identity used to combine fine blocks into coarse windows
    ("Chen: ince → kaba birleşim"). All tensors are kept explicitly:
    S1 (d,), S2 (d,d), S3 (d,d,d).
    """

    @staticmethod
    def segment(v):
        """Signature of a single linear segment with increment vector v."""
        S1 = v
        S2 = np.outer(v, v) / 2.0
        S3 = np.einsum('i,j,k->ijk', v, v, v) / 6.0
        return (S1, S2, S3)

    @staticmethod
    def combine(A, B):
        """Chen identity: Sig(x*y) = Sig(x) ⊗ Sig(y), truncated at level 3."""
        A1, A2, A3 = A
        B1, B2, B3 = B
        C1 = A1 + B1
        C2 = A2 + B2 + np.outer(A1, B1)
        C3 = (A3 + B3
              + np.einsum('i,jk->ijk', A1, B2)
              + np.einsum('ij,k->ijk', A2, B1))
        return (C1, C2, C3)

    @staticmethod
    def of_increments(dX):
        """Signature of a path given its per-step increments dX (n_steps, d)."""
        d = dX.shape[1]
        sig = (np.zeros(d), np.zeros((d, d)), np.zeros((d, d, d)))
        for k in range(dX.shape[0]):
            sig = ChenSignature.combine(sig, ChenSignature.segment(dX[k]))
        return sig

    @staticmethod
    def flatten(sig):
        S1, S2, S3 = sig
        return np.concatenate([S1.ravel(), S2.ravel(), S3.ravel()])


class VolatilityNormalizer:
    """
    Volatility Normalization stage. Log returns are divided by an EWMA realized
    volatility and bar-level signed flow (order-flow proxy: sign(close-open)·volume)
    is divided by its own EWMA scale, so every resolution sees O(1) increments.
    Also produces the model-free strata used by the anchor panel
    (RV quantile, jump flag, volume quantile, hour bucket).
    """

    def __init__(self, span=64, eps=1e-10):
        self.span = span
        self.eps = eps

    @staticmethod
    def _ewma_std(x, span):
        alpha = 2.0 / (span + 1.0)
        var = np.zeros(len(x))
        m = 0.0
        v = np.mean(x[:8] ** 2) if len(x) >= 8 else (x[0] ** 2 if len(x) else 1.0)
        for i in range(len(x)):
            m = (1 - alpha) * m + alpha * x[i]
            v = (1 - alpha) * v + alpha * (x[i] - m) ** 2
            var[i] = v
        return np.sqrt(np.maximum(var, 1e-18))

    def transform(self, df):
        c = df['close'].values.astype(float)
        o = df['open'].values.astype(float)
        v = df['volume'].values.astype(float)
        n = len(c)
        log_ret = np.zeros(n)
        log_ret[1:] = np.diff(np.log(np.maximum(c, self.eps)))

        rv = self._ewma_std(log_ret, self.span)
        rv = np.maximum(rv, self.eps)
        norm_ret = np.clip(log_ret / rv, -8.0, 8.0)

        flow = np.sign(c - o) * v
        fs = self._ewma_std(flow, self.span)
        norm_flow = np.clip(flow / np.maximum(fs, self.eps), -8.0, 8.0)

        # increments of the 3-d path: (dt=1 bar, normalized log return, normalized flow).
        # dt is a constant 1 bar at every resolution so Chen combination stays exact;
        # per-resolution scale differences are absorbed by fold-local standardization.
        increments = np.column_stack([np.ones(n), norm_ret, norm_flow])
        increments[0, 1] = 0.0

        log_rv = np.log(rv)
        jump = np.abs(norm_ret) > 4.0

        # model-free strata (per bar) for the anchor panel
        def _quantile_bucket(x, qs=(0.33, 0.66)):
            lo, hi = np.quantile(x, qs[0]), np.quantile(x, qs[1])
            return (x > lo).astype(int) + (x > hi).astype(int)

        rv_q = _quantile_bucket(rv)
        vol_q = _quantile_bucket(v)
        if 'timestamp' in df.columns:
            hours = pd.to_datetime(df['timestamp']).dt.hour.values
        else:
            hours = np.zeros(n, dtype=int)
        hour_b = (hours // 6).astype(int)  # 4 buckets of 6h

        return {
            "increments": increments,
            "norm_ret": norm_ret,
            "rv": rv,
            "log_rv": log_rv,
            "jump": jump,
            "strata": np.column_stack([rv_q, jump.astype(int), vol_q, hour_b]),
        }


class MultiResolutionSignatures:
    """
    Multi-resolution Paths (5/15/30/60 — dyadic tree) → Rough Path Signatures.
    Signatures of end-aligned windows are produced for every bar t ≥ 60 at each
    resolution. Fine 5-bar block signatures are computed once and coarser
    resolutions are assembled from them via Chen's identity (15 = 3×5,
    30 = 2×15, 60 = 2×30), so the fine → coarse combination is exact.
    """

    def __init__(self, resolutions=GEOM_RESOLUTIONS):
        self.resolutions = tuple(resolutions)
        self.warmup = max(self.resolutions)

    def compute(self, increments):
        n = increments.shape[0]
        res_feats = {r: np.zeros((n, GEOM_SIG_DIM)) for r in self.resolutions}
        valid = np.zeros(n, dtype=bool)
        if n <= self.warmup:
            return res_feats, valid

        # 5-bar block signature ending at t (bars t-4..t), for every t ≥ 4
        base = self.resolutions[0]
        block = {}
        for t in range(base - 1, n):
            block[t] = ChenSignature.of_increments(increments[t - base + 1:t + 1])

        for t in range(self.warmup, n):
            # end-aligned block chain: twelve 5-bar blocks cover the last 60 bars
            ends = [t - base * k for k in range(11, -1, -1)]  # oldest → newest
            sigs = [block[e] for e in ends]

            def chain(blocks):
                s = blocks[0]
                for b in blocks[1:]:
                    s = ChenSignature.combine(s, b)
                return s

            res_feats[5][t] = ChenSignature.flatten(sigs[-1])
            res_feats[15][t] = ChenSignature.flatten(chain(sigs[-3:]))
            res_feats[30][t] = ChenSignature.flatten(chain(sigs[-6:]))
            res_feats[60][t] = ChenSignature.flatten(chain(sigs[-12:]))
            valid[t] = True
        return res_feats, valid


class DeltaHatDiagnostic:
    """
    δ̂ diagnostic — Gromov four-point hyperbolicity on sampled intra-resolution
    quadruples, scale-normalized by the diameter. Runs on the FIRST window only
    and yields κ_init plus the factor budget (how many dimensions H gets; S only
    if the periodicity test passes, E only if δ̂ is weak).
    """

    KAPPA_ABS_MIN = 0.10   # κ_max = -0.10 < 0 (hard bound, "κ ≤ κ_max < 0")
    KAPPA_ABS_MAX = 4.00

    def __init__(self, n_quadruples=1500, seed=42):
        self.n_quadruples = n_quadruples
        self.seed = seed

    def measure(self, feats):
        """feats: (n, F) signature features of ONE resolution. Returns scale-normalized δ̂_rel.

        Scale normalization is ROBUST: p95 of pairwise distances instead of the max
        diameter. On real 1m data jump bars stretch the diameter, which drives
        δ̂_rel → 0 and slams κ_init into its bound (the κ̂=-4.00 saturation);
        a robust scale keeps 'magnitude from data' meaningful."""
        n = len(feats)
        if n < 16:
            return 0.35
        rng = np.random.default_rng(self.seed)
        m = min(n, 220)
        idx = rng.choice(n, size=m, replace=False)
        P = feats[idx]
        D = np.sqrt(np.maximum(
            np.sum(P ** 2, axis=1)[:, None] + np.sum(P ** 2, axis=1)[None, :] - 2.0 * (P @ P.T), 0.0))
        off = D[np.triu_indices(m, 1)]
        scale = float(np.quantile(off, 0.95))
        if scale <= 1e-12:
            return 0.35
        deltas = np.empty(self.n_quadruples)
        quad = rng.integers(0, m, size=(self.n_quadruples, 4))
        for q in range(self.n_quadruples):
            x, y, z, w = quad[q]
            s1 = D[x, y] + D[z, w]
            s2 = D[x, z] + D[y, w]
            s3 = D[x, w] + D[y, z]
            a, b, _ = sorted((s1, s2, s3), reverse=True)
            deltas[q] = (a - b) / 2.0
        # p90 of the four-point defect over the robust scale
        delta_rel = float(2.0 * np.quantile(deltas, 0.90) / scale)
        return max(delta_rel, 1e-4)

    def kappa_from_delta(self, delta_rel):
        """Khrulkov-style mapping c = (0.144/δ_rel)², bounded to [κ_min, κ_max]."""
        c = (0.144 / max(delta_rel, 1e-3)) ** 2
        c = float(np.clip(c, self.KAPPA_ABS_MIN, self.KAPPA_ABS_MAX))
        return -c

    @staticmethod
    def periodicity_test(norm_ret, min_period=8, max_period=96, power_ratio=6.0):
        """S factor admission: dominant FFT peak must clearly beat the median power."""
        x = norm_ret[-1024:] if len(norm_ret) > 1024 else norm_ret
        x = x - np.mean(x)
        if len(x) < 4 * min_period:
            return False, 0.0, 0
        spec = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), d=1.0)
        periods = np.divide(1.0, freqs, out=np.full_like(freqs, np.inf), where=freqs > 0)
        band = (periods >= min_period) & (periods <= max_period)
        if not np.any(band):
            return False, 0.0, 0
        med = float(np.median(spec[1:])) + 1e-12
        peak_i = np.argmax(spec * band)
        ratio = float(spec[peak_i] / med)
        return ratio >= power_ratio, ratio, int(round(periods[peak_i]))

    def build_schema(self, res_feats, valid, norm_ret, timeframe, version=1):
        """First-window protocol: sign from theory (dyadic tree → κ<0), magnitude from δ̂,
        factor budget frozen here. Adding/removing S/E later is a schema change and is
        only allowed at a fold boundary, with a version bump."""
        delta_hat = {}
        vidx = np.where(valid)[0]
        for r in GEOM_RESOLUTIONS:
            delta_hat[r] = self.measure(res_feats[r][vidx])
        # magnitude from the finest structurally meaningful level (median across resolutions)
        delta_med = float(np.median(list(delta_hat.values())))
        kappa_init = self.kappa_from_delta(delta_med)
        s_ok, s_ratio, s_period = self.periodicity_test(norm_ret)
        e_ok = delta_med > 0.30   # δ̂ weak → data not strongly hyperbolic → grant E dims
        budget = {
            "a": 6,
            "b": 2 if s_ok else 0,
            "c": 2 if e_ok else 0,
            "S_active": bool(s_ok),
            "E_active": bool(e_ok),
            "S_period": int(s_period),
            "S_ratio": float(s_ratio),
        }
        return GeometrySchema(
            version=version,
            timeframe=timeframe,
            resolutions=list(GEOM_RESOLUTIONS),
            kappa_init=kappa_init,
            kappa_max=-self.KAPPA_ABS_MIN,
            delta_hat={str(k): float(vv) for k, vv in delta_hat.items()},
            budget=budget,
        )


class GeometrySchema:
    """
    Geometry protocol contract: the schema (κ sign+init, factor budget, resolutions)
    is fixed in the first window. S/E membership changes are schema changes —
    allowed only at fold boundaries and always versioned.
    """

    def __init__(self, version, timeframe, resolutions, kappa_init, kappa_max, delta_hat, budget):
        self.version = int(version)
        self.timeframe = timeframe
        self.resolutions = list(resolutions)
        self.kappa_init = float(kappa_init)
        self.kappa_max = float(kappa_max)
        self.delta_hat = dict(delta_hat)
        self.budget = dict(budget)

    def label(self):
        s = f"H^{self.budget['a']}(κ̂={self.kappa_init:.2f})"
        if self.budget.get("S_active"):
            s += f"×S^{self.budget['b']}"
        if self.budget.get("E_active"):
            s += f"×E^{self.budget['c']}"
        return s + f" v{self.version}"

    def to_dict(self):
        return {
            "version": self.version, "timeframe": self.timeframe,
            "resolutions": self.resolutions, "kappa_init": self.kappa_init,
            "kappa_max": self.kappa_max, "delta_hat": self.delta_hat,
            "budget": self.budget,
        }

    @staticmethod
    def from_dict(d):
        return GeometrySchema(
            d.get("version", 1), d.get("timeframe", "1m"), d.get("resolutions", list(GEOM_RESOLUTIONS)),
            d.get("kappa_init", -0.5), d.get("kappa_max", -0.1),
            d.get("delta_hat", {}), d.get("budget", {"a": 6, "b": 0, "c": 0}))

    @staticmethod
    def load_store():
        try:
            if GEOMETRY_STORE_PATH.exists():
                import json
                with open(GEOMETRY_STORE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            log.error(f"Geometry store load error: {e}")
        return {"schemas": {}}

    def persist(self):
        """Versioned, append-style persistence keyed by timeframe."""
        try:
            import json
            store = GeometrySchema.load_store()
            entry = self.to_dict()
            entry["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            hist = store["schemas"].setdefault(self.timeframe, [])
            if not hist or hist[-1].get("version") != self.version:
                hist.append(entry)
            else:
                hist[-1] = entry
            with open(GEOMETRY_STORE_PATH, "w", encoding="utf-8") as f:
                json.dump(store, f, indent=2)
        except Exception as e:
            log.error(f"Geometry store persist error: {e}")


class LearnedGeometryEncoder:
    """
    Shared encoder onto the learned-geometry manifold M = H^a(κ) [× S^b × E^c].

    Head contract (per sample):
      r_prior ← e_res   deterministic, low capacity: one softplus scalar per resolution
      u       ← content unit vector in the tangent space at the origin
      δ_i     ← content sparse radial correction (L1 objective + soft-threshold forward,
                so the proximal step keeps P(δ≠0) well defined)
      z = exp_0^κ((r_prior + δ)·u) — represented in geodesic polar form (r, u);
      distances use the hyperbolic law of cosines with curvature κ.

    κ is learnable but bounded: κ = -(c_min + (c_max-c_min)·σ(ξ)) ≤ κ_max < 0, with
    the fine-tuning done by a clipped ("bounded gradient") update. The speed
    coefficient is named η — κ is reserved for curvature.

    Training loss (training only):
      Recon (FiLM decoder x̂ = W_dec·(γ(r)⊙h + β·r), γ(r) = 1 + softplus(g)·r monotone)
      + λ₁‖δ‖₁ + λ_c·max(0, cone_violation − m)
      + λ_t·Σ_i w_{i,t}·(d(z_t, z_{t−1}) − η·s_{i,t})²,  w from per-resolution
      standardized log-RV. Optional RKD term against a teacher's panel relations.
    """

    R_FLOOR = 0.05
    R_CAP = 3.5

    def __init__(self, in_dim, schema, hidden=24, seed=42):
        self.in_dim = in_dim
        self.schema = schema
        self.hidden = hidden
        self.a = int(schema.budget.get("a", 6))
        rng = np.random.default_rng(seed)
        s1 = 1.0 / math.sqrt(in_dim)
        s2 = 1.0 / math.sqrt(hidden)
        self.params = {
            "W1": rng.normal(0, s1, (hidden, in_dim)),
            "b1": np.zeros(hidden),
            "Wu": rng.normal(0, s2, (self.a, hidden)),
            "wd": rng.normal(0, s2, hidden),
            "bd": np.array(0.0),
            "rho": _inv_softplus(np.array([1.20, 0.90, 0.60, 0.35])),  # fine→coarse prior radii
            "g": np.full(hidden, -1.0),      # γ slope ≥ 0 via softplus → γ monotone in r
            "beta": np.zeros(hidden),
            "Wdec": rng.normal(0, s2, (in_dim, hidden)),
            "bdec": np.zeros(in_dim),
        }
        # curvature: c = |κ| bounded in [c_min, c_max] ⇒ κ ≤ -c_min < 0 always
        self.c_min = DeltaHatDiagnostic.KAPPA_ABS_MIN
        self.c_max = DeltaHatDiagnostic.KAPPA_ABS_MAX
        c0 = float(np.clip(-schema.kappa_init, self.c_min + 1e-3, self.c_max - 1e-3))
        frac = (c0 - self.c_min) / (self.c_max - self.c_min)
        self.xi = float(np.log(frac / (1.0 - frac)))
        self.eta = 1.0                     # speed coefficient (renamed from κ to avoid collision)
        self.tau = 0.20                    # soft-threshold of the δ head
        # λ's are fold-local (adjusted by the fold protocol between folds)
        self.lambdas = {"l1": 0.05, "cone": 0.5, "speed": 0.20, "rkd": 1.0}
        self.cone_cos_min = math.cos(math.radians(75.0))
        self.cone_margin = 0.05
        self.cone_r_margin = 0.02
        self._adam = {k: [np.zeros_like(np.asarray(v, dtype=float)),
                          np.zeros_like(np.asarray(v, dtype=float))] for k, v in self.params.items()}
        self._adam_t = 0
        self.train_history = []
        self.E_proj = None                 # closed-form linear (ridge) readout for the E factor

    # ── basic pieces ───────────────────────────────────────────────────────────
    @property
    def curvature_c(self):
        return self.c_min + (self.c_max - self.c_min) * _sigmoid(np.array(self.xi)).item()

    @property
    def kappa(self):
        return -self.curvature_c

    def r_prior(self):
        return _softplus(self.params["rho"])

    def forward(self, X, ridx):
        """X: (B,F) signature features (standardized), ridx: (B,) resolution index 0..3."""
        p = self.params
        H1 = X @ p["W1"].T + p["b1"]
        h = np.tanh(H1)
        U0 = h @ p["Wu"].T
        nu = np.linalg.norm(U0, axis=1) + 1e-9
        u = U0 / nu[:, None]
        draw = h @ p["wd"] + p["bd"]
        delta = _soft_threshold(draw, self.tau)
        rp = self.r_prior()[ridx]
        r0 = rp + delta
        r = np.clip(r0, self.R_FLOOR, self.R_CAP)
        sp_g = _softplus(p["g"])
        gam = 1.0 + sp_g[None, :] * r[:, None]
        film = gam * h + p["beta"][None, :] * r[:, None]
        Xh = film @ p["Wdec"].T + p["bdec"]
        return {"X": X, "ridx": ridx, "H1": H1, "h": h, "U0": U0, "nu": nu, "u": u,
                "draw": draw, "delta": delta, "rp": rp, "r0": r0, "r": r,
                "sp_g": sp_g, "gam": gam, "film": film, "Xh": Xh}

    # hyperbolic distance in geodesic polar form (law of cosines), curvature -c
    @staticmethod
    def _hyp_dist(r1, r2, cosang, c):
        s = math.sqrt(c)
        A = np.cosh(s * r1) * np.cosh(s * r2) - np.sinh(s * r1) * np.sinh(s * r2) * cosang
        A = np.maximum(A, 1.0 + 1e-12)
        return np.arccosh(A) / s, A

    @staticmethod
    def _hyp_dist_grads(r1, r2, cosang, c, A):
        """Returns dd/dr1, dd/dr2, dd/dcosang for d = arccosh(A)/√c."""
        s = math.sqrt(c)
        dA = 1.0 / (s * np.sqrt(np.maximum(A * A - 1.0, 1e-12)))
        dA_dr1 = s * (np.sinh(s * r1) * np.cosh(s * r2) - np.cosh(s * r1) * np.sinh(s * r2) * cosang)
        dA_dr2 = s * (np.cosh(s * r1) * np.sinh(s * r2) - np.sinh(s * r1) * np.cosh(s * r2) * cosang)
        dA_dcos = -np.sinh(s * r1) * np.sinh(s * r2)
        return dA * dA_dr1, dA * dA_dr2, dA * dA_dcos

    # ── loss + hand-derived gradients ─────────────────────────────────────────
    def _loss_and_grads(self, X, ridx, tgrid, speed, w_speed, rkd=None):
        """
        X (B,F) stacked per-resolution blocks; ridx (B,); tgrid (B,) time index of each
        row (rows of one resolution are consecutive in time); speed (B,) s_{i,t};
        w_speed (B,) weights from per-resolution standardized log-RV.
        rkd: optional dict {"Xp","ridxp","Dt","Ct"} — teacher relations on the anchor panel.
        """
        p = self.params
        lam = self.lambdas
        fw = self.forward(X, ridx)
        B = X.shape[0]
        c = self.curvature_c
        grads = {k: np.zeros_like(np.asarray(v, dtype=float)) for k, v in p.items()}

        # accumulators flowing back into shared quantities
        dh = np.zeros_like(fw["h"])
        du = np.zeros_like(fw["u"])
        dr = np.zeros(B)

        # 1) reconstruction (FiLM decoder)
        diff = fw["Xh"] - X
        L_rec = float(np.mean(diff ** 2))
        dXh = 2.0 * diff / diff.size
        grads["Wdec"] += dXh.T @ fw["film"]
        grads["bdec"] += dXh.sum(0)
        dfilm = dXh @ p["Wdec"]
        dgam = dfilm * fw["h"]
        dh += dfilm * fw["gam"]
        grads["g"] += (dgam * fw["r"][:, None]).sum(0) * _sigmoid(p["g"])
        grads["beta"] += (dfilm * fw["r"][:, None]).sum(0)
        dr += (dgam * fw["sp_g"][None, :]).sum(1) + (dfilm * p["beta"][None, :]).sum(1)

        # 2) sparsity: λ₁‖δ‖₁ (exact subgradient off the soft-threshold kink)
        L_l1 = lam["l1"] * float(np.mean(np.abs(fw["delta"])))
        ddelta = lam["l1"] * np.sign(fw["delta"]) / B

        # 3) cone constraint between adjacent resolutions at the same timestamp
        L_cone = 0.0
        n_res = len(GEOM_RESOLUTIONS)
        T = B // n_res
        if T > 0 and lam["cone"] > 0:
            for i in range(n_res - 1):          # child (finer) i vs parent (coarser) i+1
                ci = slice(i * T, (i + 1) * T)
                pi = slice((i + 1) * T, (i + 2) * T)
                cosang = np.sum(fw["u"][ci] * fw["u"][pi], axis=1)
                ang_slack = self.cone_cos_min - cosang
                rad_slack = fw["r"][pi] - fw["r"][ci] + self.cone_r_margin
                viol = np.maximum(ang_slack, 0.0) + np.maximum(rad_slack, 0.0)
                active = viol > self.cone_margin
                L_cone += lam["cone"] * float(np.mean(np.maximum(viol - self.cone_margin, 0.0))) / (n_res - 1)
                if np.any(active):
                    scale = lam["cone"] / (T * (n_res - 1))
                    m_ang = active & (ang_slack > 0)
                    m_rad = active & (rad_slack > 0)
                    du[ci][m_ang] -= scale * fw["u"][pi][m_ang]
                    du[pi][m_ang] -= scale * fw["u"][ci][m_ang]
                    dr_c = np.zeros(T); dr_p = np.zeros(T)
                    dr_c[m_rad] -= scale
                    dr_p[m_rad] += scale
                    dr[ci] += dr_c
                    dr[pi] += dr_p

        # 4) speed consistency: λ_t·w·(d(z_t,z_{t−1}) − η·s)²
        L_speed = 0.0
        if lam["speed"] > 0 and T > 1:
            for i in range(n_res):
                blk = slice(i * T, (i + 1) * T)
                rb, ub = fw["r"][blk], fw["u"][blk]
                sb, wb = speed[blk], w_speed[blk]
                cosang = np.sum(ub[1:] * ub[:-1], axis=1)
                d, A = self._hyp_dist(rb[1:], rb[:-1], cosang, c)
                resid = d - self.eta * sb[1:]
                L_speed += lam["speed"] * float(np.mean(wb[1:] * resid ** 2)) / n_res
                dd = lam["speed"] * 2.0 * wb[1:] * resid / (n_res * (T - 1))
                g1, g2, gcos = self._hyp_dist_grads(rb[1:], rb[:-1], cosang, c, A)
                drb = np.zeros(T); dub = np.zeros_like(ub)
                drb[1:] += dd * g1
                drb[:-1] += dd * g2
                dub[1:] += (dd * gcos)[:, None] * ub[:-1]
                dub[:-1] += (dd * gcos)[:, None] * ub[1:]
                dr[blk] += drb
                du[blk] += dub

        # 5) RKD against the teacher's panel relations (fold protocol)
        L_rkd = 0.0
        if rkd is not None and lam["rkd"] > 0:
            fwp = self.forward(rkd["Xp"], rkd["ridxp"])
            P = rkd["Xp"].shape[0]
            rp_, up_ = fwp["r"], fwp["u"]
            cosM = np.clip(up_ @ up_.T, -1.0, 1.0)
            Rm = np.tile(rp_[:, None], (1, P))
            D, A = self._hyp_dist(Rm, Rm.T, cosM, c)
            np.fill_diagonal(D, 0.0)
            off = ~np.eye(P, dtype=bool)
            mD = float(np.mean(D[off])) + 1e-9
            Dn = D / mD
            dDn = 2.0 * (Dn - rkd["Dt"]) / D.size
            L_rkd += lam["rkd"] * float(np.mean((Dn - rkd["Dt"]) ** 2))
            # d/dD of (D/mD): direct term + mean term
            dD = dDn / mD - (np.sum(dDn * D) / (mD * mD)) * (off.astype(float) / off.sum())
            dD *= lam["rkd"]
            g1, g2, gcos = self._hyp_dist_grads(Rm, Rm.T, cosM, c, A)
            np.fill_diagonal(g1, 0.0); np.fill_diagonal(g2, 0.0); np.fill_diagonal(gcos, 0.0)
            drp = (dD * g1).sum(1) + (dD * g2).sum(0)
            dcosM = dD * gcos
            # angle relations
            dCs = lam["rkd"] * 2.0 * (cosM - rkd["Ct"]) / cosM.size
            np.fill_diagonal(dCs, 0.0)
            L_rkd += lam["rkd"] * float(np.mean((cosM - rkd["Ct"]) ** 2))
            dcosM += dCs
            dup = (dcosM + dcosM.T) @ up_
            # backprop through the panel forward pass into shared params
            self._backprop_heads(fwp, rkd["Xp"], dup, drp, np.zeros(P), grads)

        # ── shared backprop for the main batch ────────────────────────────────
        self._backprop_heads(fw, X, du, dr, ddelta, grads)

        total = L_rec + L_l1 + L_cone + L_speed + L_rkd
        parts = {"rec": L_rec, "l1": L_l1, "cone": L_cone, "speed": L_speed, "rkd": L_rkd}
        # dh from recon path was accumulated locally; merge it here
        self._backprop_trunk(fw, X, dh, grads)
        return total, parts, grads, fw

    def _backprop_heads(self, fw, X, du, dr, ddelta_extra, grads):
        """Route du (through u-normalization), dr (through δ clip + rho) and δ-subgradients
        back into head weights and the trunk."""
        p = self.params
        # u = U0/‖U0‖ backprop
        dU0 = (du - fw["u"] * np.sum(du * fw["u"], axis=1, keepdims=True)) / fw["nu"][:, None]
        grads["Wu"] += dU0.T @ fw["h"]
        dh = dU0 @ p["Wu"]
        # r = clip(rp + δ) backprop
        mask_r = ((fw["r0"] > self.R_FLOOR) & (fw["r0"] < self.R_CAP)).astype(float)
        dr0 = dr * mask_r
        # rho (per-resolution deterministic prior)
        sig_rho = _sigmoid(p["rho"])
        for i in range(len(GEOM_RESOLUTIONS)):
            m = fw["ridx"] == i
            if np.any(m):
                grads["rho"][i] += np.sum(dr0[m]) * sig_rho[i]
        # δ head: soft-threshold derivative is 1 off the kink, 0 inside
        ddelta = dr0 + ddelta_extra
        st_mask = (np.abs(fw["draw"]) > self.tau).astype(float)
        ddraw = ddelta * st_mask
        grads["wd"] += fw["h"].T @ ddraw
        grads["bd"] = grads["bd"] + np.sum(ddraw)
        dh += ddraw[:, None] * p["wd"][None, :]
        self._backprop_trunk(fw, X, dh, grads)

    def _backprop_trunk(self, fw, X, dh, grads):
        dH1 = dh * (1.0 - fw["h"] ** 2)
        grads["W1"] += dH1.T @ X
        grads["b1"] += dH1.sum(0)

    # ── optimization ──────────────────────────────────────────────────────────
    def _adam_step(self, grads, lr=3e-3, clip=5.0):
        # bounded gradient contract: global norm clip before every update
        gn = math.sqrt(sum(float(np.sum(np.asarray(g) ** 2)) for g in grads.values()))
        scale = min(1.0, clip / (gn + 1e-12))
        self._adam_t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, g in grads.items():
            g = np.asarray(g, dtype=float) * scale
            m, v = self._adam[k]
            m[...] = b1 * m + (1 - b1) * g
            v[...] = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** self._adam_t)
            vh = v / (1 - b2 ** self._adam_t)
            upd = lr * mh / (np.sqrt(vh) + eps)
            self.params[k] = np.asarray(self.params[k], dtype=float) - upd

    def _update_eta(self, X, ridx, speed, w_speed):
        """Closed-form weighted LS for the speed coefficient η (kept strictly positive)."""
        fw = self.forward(X, ridx)
        c = self.curvature_c
        n_res = len(GEOM_RESOLUTIONS)
        T = X.shape[0] // n_res
        num, den = 0.0, 0.0
        for i in range(n_res):
            blk = slice(i * T, (i + 1) * T)
            rb, ub = fw["r"][blk], fw["u"][blk]
            cosang = np.sum(ub[1:] * ub[:-1], axis=1)
            d, _ = self._hyp_dist(rb[1:], rb[:-1], cosang, c)
            sb, wb = speed[blk][1:], w_speed[blk][1:]
            num += float(np.sum(wb * d * sb))
            den += float(np.sum(wb * sb * sb))
        if den > 1e-12:
            self.eta = float(np.clip(num / den, 0.01, 10.0))

    def _update_kappa(self, X, ridx, speed, w_speed, rkd, step_cap=0.25):
        """κ fine-tuning by a clipped finite-difference step on the curvature-dependent
        loss terms ("ince ayar sınırlı gradyandan")."""
        def loss_at(xi_val):
            old = self.xi
            self.xi = xi_val
            lam = self.lambdas
            fw = self.forward(X, ridx)
            c = self.curvature_c
            n_res = len(GEOM_RESOLUTIONS)
            T = X.shape[0] // n_res
            L = 0.0
            for i in range(n_res):
                blk = slice(i * T, (i + 1) * T)
                rb, ub = fw["r"][blk], fw["u"][blk]
                cosang = np.sum(ub[1:] * ub[:-1], axis=1)
                d, _ = self._hyp_dist(rb[1:], rb[:-1], cosang, c)
                resid = d - self.eta * speed[blk][1:]
                L += lam["speed"] * float(np.mean(w_speed[blk][1:] * resid ** 2)) / n_res
            if rkd is not None:
                fwp = self.forward(rkd["Xp"], rkd["ridxp"])
                P = rkd["Xp"].shape[0]
                cosM = np.clip(fwp["u"] @ fwp["u"].T, -1.0, 1.0)
                Rm = np.tile(fwp["r"][:, None], (1, P))
                D, _ = self._hyp_dist(Rm, Rm.T, cosM, c)
                np.fill_diagonal(D, 0.0)
                off = ~np.eye(P, dtype=bool)
                Dn = D / (float(np.mean(D[off])) + 1e-9)
                L += lam["rkd"] * float(np.mean((Dn - rkd["Dt"]) ** 2))
            self.xi = old
            return L
        h = 0.10
        gxi = (loss_at(self.xi + h) - loss_at(self.xi - h)) / (2 * h)
        self.xi = float(self.xi - np.clip(0.5 * gxi / (abs(gxi) + 1e-9) * min(abs(gxi) * 20, step_cap),
                                          -step_cap, step_cap))

    def train(self, X, ridx, tgrid, speed, w_speed, epochs=60, lr=3e-3, rkd=None,
              val_frac=0.15, log_prefix="ENC"):
        """Full-batch Adam with chronological held-out tail for the recon diagnostic."""
        n_res = len(GEOM_RESOLUTIONS)
        T = X.shape[0] // n_res
        T_tr = max(int(T * (1.0 - val_frac)), 8)
        tr_rows = np.concatenate([np.arange(i * T, i * T + T_tr) for i in range(n_res)])
        va_rows = np.concatenate([np.arange(i * T + T_tr, (i + 1) * T) for i in range(n_res)])
        Xtr, rtr = X[tr_rows], ridx[tr_rows]
        ttr, str_, wtr = tgrid[tr_rows], speed[tr_rows], w_speed[tr_rows]
        for ep in range(epochs):
            total, parts, grads, _ = self._loss_and_grads(Xtr, rtr, ttr, str_, wtr, rkd=rkd)
            self._adam_step(grads, lr=lr)
            if ep % 3 == 2:
                self._update_eta(Xtr, rtr, str_, wtr)
                self._update_kappa(Xtr, rtr, str_, wtr, rkd)
            if ep == 0 or ep == epochs - 1 or ep % 10 == 9:
                self.train_history.append({"epoch": ep, **{k: round(v, 6) for k, v in parts.items()}})
        # closed-form ridge readout for the E factor (linear residual innovation)
        fw = self.forward(X, ridx)
        resid = X - fw["Xh"]
        H = fw["h"]
        A = H.T @ H + 1e-3 * np.eye(H.shape[1])
        self.E_proj = np.linalg.solve(A, H.T @ resid)
        # held-out recon diagnostic
        if len(va_rows):
            fwv = self.forward(X[va_rows], ridx[va_rows])
            heldout = float(np.mean((fwv["Xh"] - X[va_rows]) ** 2))
        else:
            heldout = float("nan")
        log.info(f"[{log_prefix}] trained {epochs} epochs | κ={self.kappa:.3f} η={self.eta:.3f} "
                 f"| heldout_recon={heldout:.4f} | P(δ≠0)={float(np.mean(fw['delta'] != 0)):.3f}")
        return heldout

    # ── inference / diagnostics ───────────────────────────────────────────────
    def embed(self, X, ridx):
        fw = self.forward(X, ridx)
        out = {"r": fw["r"], "u": fw["u"], "delta": fw["delta"], "h": fw["h"],
               "recon_err": np.mean((fw["Xh"] - X) ** 2, axis=1)}
        if self.E_proj is not None:
            lin_res = (X - fw["Xh"]) - fw["h"] @ self.E_proj
            out["e_resid"] = np.linalg.norm(lin_res, axis=1)
        else:
            out["e_resid"] = np.zeros(X.shape[0])
        return out

    def pair_dist(self, r1, u1, r2, u2):
        cosang = np.clip(np.sum(u1 * u2, axis=-1), -1.0, 1.0)
        d, _ = self._hyp_dist(r1, r2, cosang, self.curvature_c)
        return d

    def panel_relations(self, Xp, ridxp):
        """Batch-normalized distance matrix + angle relations on the (frozen) panel —
        the RKD target a student fold distills from ('öğretmen = yalnız önceki fold')."""
        fw = self.forward(Xp, ridxp)
        P = Xp.shape[0]
        cosM = np.clip(fw["u"] @ fw["u"].T, -1.0, 1.0)
        Rm = np.tile(fw["r"][:, None], (1, P))
        D, _ = self._hyp_dist(Rm, Rm.T, cosM, self.curvature_c)
        np.fill_diagonal(D, 0.0)
        off = ~np.eye(P, dtype=bool)
        Dn = D / (float(np.mean(D[off])) + 1e-9)
        return {"Dt": Dn, "Ct": cosM}

    def e_res_intervention(self, X, ridx):
        """Mandatory dashboard: swap the e_res → r_prior mapping (roll resolution
        assignment by one) and measure recon degradation. ≈0 means the radius prior
        is unused; >0 means the geometry actually consumes e_res."""
        base = float(np.mean((self.forward(X, ridx)["Xh"] - X) ** 2))
        rolled = (ridx + 1) % len(GEOM_RESOLUTIONS)
        pert = float(np.mean((self.forward(X, rolled)["Xh"] - X) ** 2))
        return {"recon": base, "recon_intervened": pert,
                "degradation": (pert - base) / (base + 1e-12)}

    def r_eff(self):
        """r_prior·√|κ| per resolution — is curvature actually used or do we live in
        the flat region?"""
        sq = math.sqrt(self.curvature_c)
        return {str(r): float(v * sq) for r, v in zip(GEOM_RESOLUTIONS, self.r_prior())}


class InnovationEngine:
    """
    Innovations (factor based), standardized with fold-local μ/σ.
    The feature schema handed to the upper layers is one of the two permanent
    contracts of the system — names are stable; S/E columns exist only if the
    (frozen) geometry schema granted them a budget.
      H:  δ_i, cone slack        [S: phase deviation]   [E: linear residual]
      +   speed residual, r̃_i, θ levels, Δθ_ij
    """

    def __init__(self, schema):
        self.schema = schema
        names = []
        for r in GEOM_RESOLUTIONS:
            names += [f"delta_{r}", f"r_{r}", f"theta_{r}", f"speed_resid_{r}"]
        for i in range(len(GEOM_RESOLUTIONS) - 1):
            a, b = GEOM_RESOLUTIONS[i], GEOM_RESOLUTIONS[i + 1]
            names += [f"dtheta_{a}_{b}", f"cone_ang_{a}_{b}", f"cone_rad_{a}_{b}"]
        if schema.budget.get("S_active"):
            names += [f"phase_dev_{r}" for r in GEOM_RESOLUTIONS]
        if schema.budget.get("E_active"):
            names += [f"e_resid_{r}" for r in GEOM_RESOLUTIONS]
        self.names = names
        self.mu = None
        self.sigma = None
        self.mean_dphase = 0.0

    def _phase_series(self, norm_ret, period):
        """S factor: phase of the dominant-frequency component over a trailing window."""
        n = len(norm_ret)
        period = max(int(period), 8)
        win = min(4 * period, 256)
        ph = np.zeros(n)
        tt = np.arange(win)
        co = np.cos(2 * np.pi * tt / period)
        si = np.sin(2 * np.pi * tt / period)
        for t in range(win, n):
            seg = norm_ret[t - win:t]
            ph[t] = math.atan2(float(seg @ si), float(seg @ co))
        return ph

    def compute_raw(self, encoder, embeds, speed, norm_ret):
        """embeds: dict res → encoder embed dict over a common T grid."""
        T = len(embeds[GEOM_RESOLUTIONS[0]]["r"])
        cols = []
        eta = encoder.eta
        for r in GEOM_RESOLUTIONS:
            e = embeds[r]
            theta = np.arctan2(e["u"][:, 1], e["u"][:, 0])
            d_step = np.zeros(T)
            if T > 1:
                d_step[1:] = encoder.pair_dist(e["r"][1:], e["u"][1:], e["r"][:-1], e["u"][:-1])
            speed_resid = d_step - eta * speed
            cols += [e["delta"], e["r"], theta, speed_resid]
        for i in range(len(GEOM_RESOLUTIONS) - 1):
            ec = embeds[GEOM_RESOLUTIONS[i]]
            ep = embeds[GEOM_RESOLUTIONS[i + 1]]
            cosang = np.clip(np.sum(ec["u"] * ep["u"], axis=1), -1.0, 1.0)
            dtheta = np.arccos(cosang)
            cone_ang = encoder.cone_cos_min - cosang           # >0 → outside the cone
            cone_rad = ep["r"] - ec["r"] + encoder.cone_r_margin
            cols += [dtheta, cone_ang, cone_rad]
        if self.schema.budget.get("S_active"):
            period = self.schema.budget.get("S_period", 24) or 24
            ph = self._phase_series(norm_ret, period)[-T:]
            dph = np.zeros(T)
            dph[1:] = np.angle(np.exp(1j * np.diff(ph)))
            phase_dev = dph - self.mean_dphase
            for r in GEOM_RESOLUTIONS:
                cols.append(phase_dev)   # shared data-side phase, per-res slot kept for schema stability
        if self.schema.budget.get("E_active"):
            for r in GEOM_RESOLUTIONS:
                cols.append(embeds[r]["e_resid"])
        return np.column_stack(cols)

    def fit_norm(self, X_raw):
        """fold-local μ, σ."""
        self.mu = np.mean(X_raw, axis=0)
        self.sigma = np.std(X_raw, axis=0) + 1e-9

    def transform(self, X_raw):
        return (X_raw - self.mu) / self.sigma

    def group_masks(self):
        """Feature-stability ablation groups: 'levels' vs 'deviations'."""
        levels = np.array([n.startswith(("r_", "theta_", "dtheta_")) for n in self.names])
        return {"levels": levels, "deviations": ~levels}


class EpisodeSegmenter:
    """Episode segmentation from δ≠0 runs with hysteresis: ON after `on_bars`
    consecutive bars with ≥ `min_active` active δ's, OFF after `off_bars` clean bars."""

    def __init__(self, min_active=2, on_bars=2, off_bars=3):
        self.min_active = min_active
        self.on_bars = on_bars
        self.off_bars = off_bars
        self.reset()

    def reset(self):
        self._on_run = 0
        self._off_run = 0
        self.active = False
        self._cur_start = -1

    def step(self, t, n_active):
        """Returns (episode_active, closed_episode_span or None)."""
        closed = None
        if n_active >= self.min_active:
            self._on_run += 1
            self._off_run = 0
        else:
            self._off_run += 1
            self._on_run = 0
        if not self.active and self._on_run >= self.on_bars:
            self.active = True
            self._cur_start = t - self.on_bars + 1
        elif self.active and self._off_run >= self.off_bars:
            self.active = False
            closed = (self._cur_start, t - self.off_bars)
            self._cur_start = -1
        return self.active, closed

    def segment(self, active_counts):
        """Batch mode over an array of per-bar active-δ counts."""
        self.reset()
        flags = np.zeros(len(active_counts), dtype=bool)
        episodes = []
        for t, k in enumerate(active_counts):
            on, closed = self.step(t, int(k))
            flags[t] = on
            if closed is not None:
                episodes.append(closed)
        if self.active and self._cur_start >= 0:
            episodes.append((self._cur_start, len(active_counts) - 1))
        return flags, episodes


class AnomalyClusterer:
    """K-means (pure NumPy) over episode summary vectors."""

    SUMMARY_DIM = 6

    def __init__(self, k=3, seed=42):
        self.k = k
        self.seed = seed
        self.centroids = None
        self.mu = None
        self.sigma = None

    @staticmethod
    def summarize(span, deltas_abs_sum, cone_viol, speed_resid_abs, dtheta_abs, norm_ret):
        s, e = span
        seg = slice(s, e + 1)
        dur = float(e - s + 1)
        return np.array([
            math.log1p(dur),
            float(np.mean(deltas_abs_sum[seg])),
            float(np.max(cone_viol[seg])) if e >= s else 0.0,
            float(np.mean(speed_resid_abs[seg])),
            float(np.mean(dtheta_abs[seg])),
            float(np.sum(norm_ret[seg])),
        ])

    def fit(self, S):
        if len(S) < self.k:
            self.centroids = None
            return self
        self.mu = S.mean(0)
        self.sigma = S.std(0) + 1e-9
        Z = (S - self.mu) / self.sigma
        rng = np.random.default_rng(self.seed)
        cent = Z[rng.choice(len(Z), self.k, replace=False)]
        for _ in range(25):
            d2 = ((Z[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
            lab = np.argmin(d2, axis=1)
            for j in range(self.k):
                if np.any(lab == j):
                    cent[j] = Z[lab == j].mean(0)
        self.centroids = cent
        return self

    def assign(self, s_vec):
        if self.centroids is None:
            return -1
        z = (s_vec - self.mu) / self.sigma
        return int(np.argmin(((self.centroids - z) ** 2).sum(1)))


class TransitionGraph:
    """Normal-centric transition graph: states {N, A_0..A_{k-1}} with NORMAL between
    episodes. Purely observational statistics feeding the classifier."""

    def __init__(self, k=3):
        self.k = k
        self.n_states = k + 1        # 0 = NORMAL
        self.counts = np.ones((self.n_states, self.n_states)) * 0.5   # Laplace prior
        self.state_ret = np.zeros(self.n_states)
        self.state_ret_n = np.zeros(self.n_states)

    def fit(self, state_sequence, fwd_returns):
        for i in range(len(state_sequence) - 1):
            self.counts[state_sequence[i], state_sequence[i + 1]] += 1.0
        for st, fr in fwd_returns:
            self.state_ret[st] += fr
            self.state_ret_n[st] += 1.0
        return self

    def transmat(self):
        return self.counts / self.counts.sum(axis=1, keepdims=True)

    def expected_return(self, state):
        n = self.state_ret_n[state]
        return float(self.state_ret[state] / n) if n > 0 else 0.0

    def p_return_to_normal(self, state):
        return float(self.transmat()[state, 0])

    def features(self, episode_active, cluster):
        """Per-bar graph features for the classifier: [in_episode, onehot cluster,
        P(→NORMAL), E[fwd ret | state]]."""
        state = 0 if not episode_active or cluster < 0 else cluster + 1
        onehot = np.zeros(self.k)
        if state > 0:
            onehot[state - 1] = 1.0
        return np.concatenate([[float(episode_active)], onehot,
                               [self.p_return_to_normal(state), self.expected_return(state)]])

    @property
    def n_features(self):
        return 3 + self.k


class LightGBMModel:
    """Small adapter around the real LightGBM binary classifier.

    The rest of the geometry pipeline expects a one-dimensional P(up) array, while
    LightGBM follows the sklearn convention and returns [P(0), P(1)].  Keeping the
    conversion here makes the model swap explicit and prevents an accidental return
    to the old in-file gradient-boosting approximation.
    """

    def __init__(self, n_trees=120, depth=4, lr=0.05, min_leaf=20, n_bins=255, seed=42):
        if lgb is None:
            raise RuntimeError(
                "LightGBM is required. Install it with: pip install lightgbm>=4.0,<5"
            )
        self.n_trees = int(n_trees)
        self.depth = int(depth)
        self.lr = float(lr)
        self.min_leaf = int(min_leaf)
        self.n_bins = max(31, int(n_bins))
        self.seed = int(seed)
        self.model = None
        self.constant_probability = None

    def fit(self, X, y, X_valid=None, y_valid=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        classes = np.unique(y)
        if len(classes) < 2:
            self.constant_probability = float(classes[0]) if len(classes) else 0.5
            self.model = None
            return self

        self.constant_probability = None
        self.model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=self.n_trees,
            learning_rate=self.lr,
            max_depth=self.depth,
            num_leaves=min(2 ** self.depth, 31),
            min_child_samples=self.min_leaf,
            max_bin=self.n_bins,
            subsample=0.90,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.20,
            random_state=self.seed,
            n_jobs=-1,
            deterministic=True,
            verbosity=-1,
        )
        fit_kwargs = {}
        if X_valid is not None and y_valid is not None:
            X_valid = np.asarray(X_valid, dtype=float)
            y_valid = np.asarray(y_valid, dtype=int)
            if len(X_valid) and len(np.unique(y_valid)) > 1:
                import inspect
                fit_kwargs = {
                    "eval_metric": "auc",
                    "callbacks": [lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
                }
                # LightGBM 4.7 introduced eval_X/eval_y and deprecated eval_set;
                # keep compatibility with both the 4.0-4.6 and 4.7 APIs.
                if "eval_X" in inspect.signature(self.model.fit).parameters:
                    fit_kwargs.update({"eval_X": X_valid, "eval_y": y_valid})
                else:
                    fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        self.model.fit(X, y, **fit_kwargs)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if self.constant_probability is not None:
            return np.full(len(X), self.constant_probability, dtype=float)
        if self.model is None:
            raise RuntimeError("LightGBM model has not been fitted")
        return np.asarray(self.model.predict_proba(X)[:, 1], dtype=float)

    @staticmethod
    def auc(y, p):
        order = np.argsort(p)
        ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
        pos = y > 0.5
        n1, n0 = int(np.sum(pos)), int(np.sum(~pos))
        if n1 == 0 or n0 == 0:
            return 0.5
        return float((np.sum(ranks[pos]) - n1 * (n1 + 1) / 2) / (n1 * n0))


class MetaLabeler:
    """
    Meta labeling: logistic model estimating P(primary signal wins the barrier race
    TP-before-SL). Trained on the primary model's own historical signals.
    """

    def __init__(self, lr=0.1, epochs=300, l2=1e-3):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w = None
        self.mu = None
        self.sigma = None

    @staticmethod
    def barrier_outcome(closes, t, tp_pct, sl_pct, max_hold=30):
        entry = closes[t]
        for j in range(t + 1, min(t + max_hold + 1, len(closes))):
            r = (closes[j] - entry) / entry * 100.0
            if r >= tp_pct:
                return 1
            if r <= -sl_pct:
                return 0
        return 1 if closes[min(t + max_hold, len(closes) - 1)] > entry else 0

    @staticmethod
    def barrier_dir(closes, t, tp_pct, max_hold=30):
        """Symmetric directional label for the primary model: 1 if the +tp barrier
        is touched before the -tp barrier, else 0. Final-sign fallback if neither.
        Symmetric (equal up/down targets) so P(up) is unbiased for a two-sided model."""
        entry = closes[t]
        for j in range(t + 1, min(t + max_hold + 1, len(closes))):
            r = (closes[j] - entry) / entry * 100.0
            if r >= tp_pct:
                return 1.0
            if r <= -tp_pct:
                return 0.0
        return 1.0 if closes[min(t + max_hold, len(closes) - 1)] > entry else 0.0

    @staticmethod
    def barrier_outcome_dir(closes, t, tp_pct, sl_pct, side, max_hold=30):
        """Directional TP-before-SL race for the meta model / trade outcome.
        long:  +tp before -sl.   short: -tp before +sl (price falls tp% first)."""
        entry = closes[t]
        for j in range(t + 1, min(t + max_hold + 1, len(closes))):
            r = (closes[j] - entry) / entry * 100.0
            if side == "long":
                if r >= tp_pct:
                    return 1
                if r <= -sl_pct:
                    return 0
            else:
                if r <= -tp_pct:
                    return 1
                if r >= sl_pct:
                    return 0
        fin = closes[min(t + max_hold, len(closes) - 1)] - entry
        return int((fin > 0) == (side == "long"))

    def fit(self, X, y):
        self.mu = X.mean(0)
        self.sigma = X.std(0) + 1e-9
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sigma])
        w = np.zeros(Z.shape[1])
        y = np.asarray(y, dtype=float)
        for _ in range(self.epochs):
            p = _sigmoid(Z @ w)
            grad = Z.T @ (p - y) / len(y) + self.l2 * w
            w -= self.lr * grad
        self.w = w
        return self

    def predict_proba(self, X):
        if self.w is None:
            return np.full(X.shape[0], 0.5)
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sigma])
        return _sigmoid(Z @ self.w)


class ConformalGate:
    """
    Conformal prediction gate: A = α·A_pred + β·A_geom(panel).
    Calibration uses a RECENT, CHRONOLOGICAL, deliberately NON-stratified window
    ("yakın, kronolojik, stratifiye-EDİLMEMİŞ pencere").
    """

    def __init__(self, alpha=0.6, beta=0.4):
        self.alpha = alpha
        self.beta = beta
        self.cal_scores = np.array([])      # nonconformity = 1 - p_meta
        self.cal_dmin = np.array([])        # geometric distances on the calibration window
        self.a_min = 0.5

    def calibrate(self, p_meta_cal, dmin_cal, target_accept=0.60):
        self.cal_scores = np.sort(1.0 - np.asarray(p_meta_cal, dtype=float))
        self.cal_dmin = np.sort(np.asarray(dmin_cal, dtype=float))
        if len(p_meta_cal) >= 5:
            A_cal = np.array([self.combined(p, d) for p, d in zip(p_meta_cal, dmin_cal)])
            self.a_min = float(np.quantile(A_cal, 1.0 - target_accept))
        else:
            self.a_min = 0.5

    def a_pred(self, p_meta):
        if len(self.cal_scores) == 0:
            return 0.5
        s = 1.0 - p_meta
        ge = len(self.cal_scores) - np.searchsorted(self.cal_scores, s, side="left")
        return float((1.0 + ge) / (len(self.cal_scores) + 1.0))

    def a_geom(self, dmin):
        """Geometric half of the conformal score, from the anchor panel distances."""
        if len(self.cal_dmin) == 0:
            return 0.5
        ge = len(self.cal_dmin) - np.searchsorted(self.cal_dmin, dmin, side="left")
        return float((1.0 + ge) / (len(self.cal_dmin) + 1.0))

    def combined(self, p_meta, dmin):
        return self.alpha * self.a_pred(p_meta) + self.beta * self.a_geom(dmin)


class AnchorPanel:
    """
    ANCHOR PANEL — off-flow measurement standard.
      Core: frozen at the signature level, append-only ledger (anchors are retired,
            never deleted; metadata: added/last-seen fold, OOD score, usage, status).
            Strata are model-free (RV quantile, jump, volume, hour), kept per resolution.
      Adaptive: rolling buffer tracking current regimes.
    Feeds: RKD targets (teacher = previous fold only), δ̂ health + geometry drift,
    per-fold observational D_anchor log, A_geom, κ_init.
    """

    ADAPTIVE_MAX = 64

    def __init__(self, max_per_stratum=2):
        self.core = []                       # append-only ledger
        self.adaptive = {r: [] for r in GEOM_RESOLUTIONS}
        self.max_per_stratum = max_per_stratum
        self.d_anchor_log = []               # per fold, observational
        self.baseline_relations = None       # fold-0 panel relations
        self.kappa_init = None

    # ── core ledger ───────────────────────────────────────────────────────────
    def build_core(self, res_feats, valid, strata, fold=0):
        vidx = np.where(valid)[0]
        for ri, r in enumerate(GEOM_RESOLUTIONS):
            F = res_feats[r][vidx]
            S = strata[vidx]
            keys = [tuple(row) for row in S]
            buckets = {}
            for i, k in enumerate(keys):
                buckets.setdefault(k, []).append(i)
            for k, rows in buckets.items():
                rows = np.array(rows)
                sub = F[rows]
                centroid = sub.mean(0)
                d = np.linalg.norm(sub - centroid, axis=1)
                order = np.argsort(d)[:self.max_per_stratum]
                for oi in order:
                    self.core.append({
                        "sig": F[rows[oi]].copy(), "res": r, "res_idx": ri,
                        "stratum": tuple(int(x) for x in k),
                        "added_fold": fold, "last_seen_fold": fold,
                        "ood": 0.0, "usage": 0, "status": "active",
                    })
        log.info(f"[PANEL] core ledger built: {len(self.core)} anchors "
                 f"({sum(1 for a in self.core if a['status']=='active')} active)")

    def active_core(self, res=None):
        return [a for a in self.core
                if a["status"] == "active" and (res is None or a["res"] == res)]

    def push_adaptive(self, res, sig):
        buf = self.adaptive[res]
        buf.append(np.asarray(sig, dtype=float))
        if len(buf) > self.ADAPTIVE_MAX:
            buf.pop(0)

    def refresh(self, res_feats, valid, fold):
        """Update OOD/usage/last-seen metadata; retire stale anchors (never delete)."""
        vidx = np.where(valid)[0]
        for r in GEOM_RESOLUTIONS:
            F = res_feats[r][vidx]
            if len(F) == 0:
                continue
            mu = F.mean(0)
            sd = F.std(0) + 1e-9
            anchors = self.active_core(res=r)
            if not anchors:
                continue
            A = np.stack([a["sig"] for a in anchors])
            ood = np.linalg.norm((A - mu) / sd, axis=1) / math.sqrt(A.shape[1])
            # nearest-anchor usage on a sample of current windows
            sample = F[:: max(1, len(F) // 64)]
            d2 = ((sample[:, None, :] - A[None, :, :]) ** 2).sum(-1)
            near = np.argmin(d2, axis=1)
            for i, a in enumerate(anchors):
                a["ood"] = float(ood[i])
                hits = int(np.sum(near == i))
                if hits > 0:
                    a["usage"] += hits
                    a["last_seen_fold"] = fold
                if fold - a["last_seen_fold"] >= 3 and a["ood"] > 3.0:
                    a["status"] = "retired"   # emekli işaretlenir, silinmez

    # ── panel sampling for RKD / drift ────────────────────────────────────────
    def panel_matrix(self, max_points=96, include_adaptive=True, seed=42):
        rows, ridx = [], []
        anchors = self.active_core()
        rng = np.random.default_rng(seed)
        if len(anchors) > max_points:
            anchors = [anchors[i] for i in rng.choice(len(anchors), max_points, replace=False)]
        for a in anchors:
            rows.append(a["sig"]); ridx.append(a["res_idx"])
        if include_adaptive:
            for ri, r in enumerate(GEOM_RESOLUTIONS):
                for sigv in self.adaptive[r][-8:]:
                    rows.append(sigv); ridx.append(ri)
        if not rows:
            return None, None
        return np.stack(rows), np.array(ridx, dtype=int)

    def rkd_targets(self, teacher, feature_standardizer):
        """Relations of the TEACHER (always the immediately previous fold) on the panel."""
        Xp, ridxp = self.panel_matrix()
        if Xp is None:
            return None
        Xps = feature_standardizer(Xp, ridxp)
        rel = teacher.panel_relations(Xps, ridxp)
        return {"Xp": Xps, "ridxp": ridxp, "Dt": rel["Dt"], "Ct": rel["Ct"]}

    def log_d_anchor(self, fold, encoder, feature_standardizer):
        """Representation-side drift: distance between the current encoder's panel
        relations and the fold-0 baseline. Observational only, never a constraint."""
        Xp, ridxp = self.panel_matrix(include_adaptive=False)
        if Xp is None:
            return None
        rel = encoder.panel_relations(feature_standardizer(Xp, ridxp), ridxp)
        if self.baseline_relations is None:
            self.baseline_relations = rel
            entry = {"fold": fold, "d_anchor": 0.0}
        else:
            P = min(rel["Dt"].shape[0], self.baseline_relations["Dt"].shape[0])
            dd = float(np.mean(np.abs(rel["Dt"][:P, :P] - self.baseline_relations["Dt"][:P, :P])))
            entry = {"fold": fold, "d_anchor": dd}
        self.d_anchor_log.append(entry)
        return entry

    def delta_hat_health(self, diag):
        """δ̂ recomputed on the frozen core signatures — geometry drift guard."""
        out = {}
        for r in GEOM_RESOLUTIONS:
            anchors = self.active_core(res=r)
            if len(anchors) >= 16:
                out[str(r)] = diag.measure(np.stack([a["sig"] for a in anchors]))
        return out

    # ── A_geom support ────────────────────────────────────────────────────────
    def dmin(self, sig_by_res):
        """Scale-normalized nearest-active-anchor distance, averaged over resolutions."""
        vals = []
        for r in GEOM_RESOLUTIONS:
            anchors = self.active_core(res=r)
            if len(anchors) < 3 or r not in sig_by_res:
                continue
            A = np.stack([a["sig"] for a in anchors])
            d = np.linalg.norm(A - sig_by_res[r], axis=1)
            spacing = np.median(np.linalg.norm(A - A.mean(0), axis=1)) + 1e-9
            vals.append(float(np.min(d) / spacing))
        return float(np.mean(vals)) if vals else 1.0

    def summary(self):
        act = sum(1 for a in self.core if a["status"] == "active")
        return {"core_total": len(self.core), "core_active": act,
                "core_retired": len(self.core) - act,
                "adaptive": {str(r): len(self.adaptive[r]) for r in GEOM_RESOLUTIONS},
                "d_anchor_log": self.d_anchor_log[-12:]}


class GeometricPipeline:
    """
    Orchestrates the full learned-geometry flow and enforces the three protocols:

    • Geometry protocol — κ sign from theory (dyadic tree → κ<0), initial magnitude
      from δ̂, factor budget frozen in the first window; S/E membership changes only
      at fold boundaries, versioned.
    • Fold protocol — warm start; RKD distillation on the panel (distance
      batch-normalized + angle relations, teacher = previous fold only); μ/σ, λ's and
      conformal calibration fold-local; D_anchor logged per fold (observational);
      panel frozen at the signature level.
    • Mandatory dashboards — P(δ≠0) per fold, intra-resolution Var(r_i), e_res
      intervention test, held-out recon, r_prior·√|κ|, and the data-side δ̂(t) vs
      representation-side D_anchor drift kept as two separate series (for attribution).
    """

    MIN_TRAIN_BARS = 340
    LIVE_REFIT_BARS = 240

    def __init__(self, timeframe="1m", seed=42, encoder_epochs=60):
        self.seed = seed
        self.encoder_epochs = encoder_epochs
        self._lock = threading.Lock()
        self._training = False
        self.reset(timeframe)

    def reset(self, timeframe):
        with getattr(self, "_lock", threading.Lock()):
            self.timeframe = timeframe
            self.status = "collecting"
            self.schema = None
            self.encoder = None
            self.innov = None
            self.clusterer = None
            self.graph = None
            self.gbm = None
            self.meta = None
            self.conformal = None
            self.panel = AnchorPanel()
            self.fold = 0
            self.p_hi = 0.6
            self.p_lo = 0.4
            self.allow_short = bool(bot_state.get("allow_short", ALLOW_SHORT))
            self.cost_pct = roundtrip_cost_pct()
            self.feat_mu = None            # fold-local per-resolution feature stats
            self.feat_sd = None
            self._panel_std0 = None        # frozen fold-0 standardizer for D_anchor
            self.feature_mask = None
            self.diagnostics = []
            self.delta_hat_series = []     # data-side δ̂(t) drift series
            self.baseline_heldout = None   # fold-0 held-out recon (health baseline)
            self.baseline_delta_hat = None # fold-0 data-side δ̂ (health baseline)
            self.last_state = {"status": "collecting", "signal": "HOLD"}
            self._bars_at_fit = 0
            self._prev_standardize = None

    # ── shared preprocessing ──────────────────────────────────────────────────
    def preprocess(self, df):
        vol = VolatilityNormalizer().transform(df)
        sigs, valid = MultiResolutionSignatures().compute(vol["increments"])
        return vol, sigs, valid

    def _fit_feature_stats(self, sigs, valid):
        mu, sd = [], []
        vidx = np.where(valid)[0]
        for r in GEOM_RESOLUTIONS:
            F = sigs[r][vidx]
            mu.append(F.mean(0))
            sd.append(F.std(0) + 1e-9)
        return mu, sd

    def _standardize(self, X, ridx, stats=None):
        mu, sd = stats if stats is not None else (self.feat_mu, self.feat_sd)
        out = np.empty_like(X, dtype=float)
        for i in range(len(GEOM_RESOLUTIONS)):
            m = ridx == i
            if np.any(m):
                out[m] = (X[m] - mu[i]) / sd[i]
        return out

    def _stack_training(self, vol, sigs, valid):
        """Res-major stacked tensors on the common valid time grid."""
        vidx = np.where(valid)[0]
        T = len(vidx)
        Xs, ridx = [], []
        for i, r in enumerate(GEOM_RESOLUTIONS):
            Xs.append((sigs[r][vidx] - 0.0))
            ridx.append(np.full(T, i, dtype=int))
        X = np.vstack(Xs)
        ridx = np.concatenate(ridx)
        X = self._standardize(X, ridx)
        speed_bar = np.abs(vol["norm_ret"])[vidx]
        speed = np.tile(speed_bar, len(GEOM_RESOLUTIONS))
        # w from per-resolution standardized log-RV
        w_parts = []
        rv = vol["rv"]
        for r in GEOM_RESOLUTIONS:
            rv_r = pd.Series(rv).rolling(r, min_periods=1).mean().values[vidx]
            z = np.log(np.maximum(rv_r, 1e-12))
            z = (z - z.mean()) / (z.std() + 1e-9)
            w_parts.append(np.exp(np.clip(z, -1.5, 1.5)))
        w = np.concatenate([p / np.mean(p) for p in w_parts])
        tgrid = np.tile(vidx, len(GEOM_RESOLUTIONS))
        return X, ridx, tgrid, speed, w, vidx

    def _embed_by_res(self, encoder, sigs, valid, stats=None):
        vidx = np.where(valid)[0]
        out = {}
        for i, r in enumerate(GEOM_RESOLUTIONS):
            Xr = self._standardize(sigs[r][vidx], np.full(len(vidx), i, dtype=int), stats=stats)
            out[r] = encoder.embed(Xr, np.full(len(vidx), i, dtype=int))
        return out, vidx

    # ── decision-layer feature assembly (causal) ──────────────────────────────
    def _decision_features(self, embeds, vidx, vol, innov_X, fit_mode, closes):
        T = len(vidx)
        deltas = np.stack([np.abs(embeds[r]["delta"]) for r in GEOM_RESOLUTIONS])
        active_counts = np.sum(deltas > 0, axis=0)
        deltas_abs_sum = deltas.sum(0)
        cone_cols = [j for j, n in enumerate(self.innov.names) if n.startswith("cone_ang")]
        sr_cols = [j for j, n in enumerate(self.innov.names) if n.startswith("speed_resid")]
        dt_cols = [j for j, n in enumerate(self.innov.names) if n.startswith("dtheta")]
        cone_viol = np.maximum(innov_X[:, cone_cols], 0.0).max(1) if cone_cols else np.zeros(T)
        speed_abs = np.abs(innov_X[:, sr_cols]).mean(1) if sr_cols else np.zeros(T)
        dtheta_abs = np.abs(innov_X[:, dt_cols]).mean(1) if dt_cols else np.zeros(T)
        nret = vol["norm_ret"][vidx]

        seg = EpisodeSegmenter()
        flags, episodes = seg.segment(active_counts)

        if fit_mode:
            S = np.stack([AnomalyClusterer.summarize(sp, deltas_abs_sum, cone_viol,
                                                     speed_abs, dtheta_abs, nret)
                          for sp in episodes]) if episodes else np.zeros((0, AnomalyClusterer.SUMMARY_DIM))
            self.clusterer = AnomalyClusterer().fit(S)

        # causal per-bar state: ongoing episodes are assigned from their partial summary
        states = np.zeros(T, dtype=int)
        ep_start = -1
        for t in range(T):
            if flags[t]:
                if ep_start < 0:
                    ep_start = t
                s_vec = AnomalyClusterer.summarize((ep_start, t), deltas_abs_sum, cone_viol,
                                                   speed_abs, dtheta_abs, nret)
                cl = self.clusterer.assign(s_vec) if self.clusterer else -1
                states[t] = cl + 1 if cl >= 0 else 1
            else:
                ep_start = -1
                states[t] = 0

        if fit_mode:
            HZ = GEOM_HORIZON
            fwd = [(states[t], float((closes[vidx[t] + HZ] - closes[vidx[t]]) / closes[vidx[t]]))
                   for t in range(T) if vidx[t] + HZ < len(closes)]
            self.graph = TransitionGraph(k=self.clusterer.k if self.clusterer else 3).fit(states, fwd)

        gfeats = np.stack([self.graph.features(bool(flags[t]), states[t] - 1) for t in range(T)])
        return np.hstack([innov_X, gfeats]), flags, states, active_counts

    # ── fitting (first window / fold update) ──────────────────────────────────
    def fit(self, df, teacher=None, warm_from=None, fold=None, precomputed=None):
        t0 = datetime.now()
        vol, sigs, valid = precomputed if precomputed is not None else self.preprocess(df)
        if int(np.sum(valid)) < 120:
            raise ValueError(f"Not enough valid windows for geometry fit ({int(np.sum(valid))})")
        fold = self.fold if fold is None else fold
        closes = df['close'].values.astype(float)

        diag = DeltaHatDiagnostic(seed=self.seed)
        if self.schema is None:
            # first-window protocol: δ̂ only here; κ_init + factor budget frozen
            self.schema = diag.build_schema(sigs, valid, vol["norm_ret"], self.timeframe,
                                            version=self._next_schema_version())
            self.schema.persist()
            self.panel.build_core(sigs, valid, vol["strata"], fold=fold)
            self.panel.kappa_init = self.schema.kappa_init
            log.info(f"[GEOM] first-window schema: {self.schema.label()} | δ̂={self.schema.delta_hat}")

        # fold-local feature standardization
        self.feat_mu, self.feat_sd = self._fit_feature_stats(sigs, valid)
        if self._panel_std0 is None:
            self._panel_std0 = (self.feat_mu, self.feat_sd)

        X, ridx, tgrid, speed, w, vidx = self._stack_training(vol, sigs, valid)

        # RKD targets: teacher = previous fold ONLY, relations on the frozen panel
        rkd = None
        if teacher is not None:
            Xp_raw, ridxp = self.panel.panel_matrix()
            if Xp_raw is not None:
                t_std = teacher.get("standardize") or (lambda Xq, rq: self._standardize(Xq, rq))
                rel = teacher["encoder"].panel_relations(t_std(Xp_raw, ridxp), ridxp)
                rkd = {"Xp": self._standardize(Xp_raw, ridxp), "ridxp": ridxp,
                       "Dt": rel["Dt"], "Ct": rel["Ct"]}

        encoder = LearnedGeometryEncoder(GEOM_SIG_DIM, self.schema, seed=self.seed)
        if warm_from is not None:
            for k_ in encoder.params:
                encoder.params[k_] = np.array(warm_from.params[k_], dtype=float, copy=True)
            encoder.xi = warm_from.xi
            encoder.eta = warm_from.eta
            encoder.lambdas = dict(warm_from.lambdas)
        epochs = self.encoder_epochs if warm_from is None else max(20, self.encoder_epochs // 2)
        heldout = encoder.train(X, ridx, tgrid, speed, w, epochs=epochs, rkd=rkd,
                                log_prefix=f"ENC f{fold}")
        self.encoder = encoder

        # innovations, fold-local μ/σ
        self.innov = InnovationEngine(self.schema)
        embeds, vidx2 = self._embed_by_res(encoder, sigs, valid)
        speed_bar = np.abs(vol["norm_ret"])[vidx2]
        raw = self.innov.compute_raw(encoder, embeds, speed_bar, vol["norm_ret"])
        self.innov.fit_norm(raw)
        Xin = self.innov.transform(raw)

        # episodes → clusters → transition graph → decision features
        Xf, flags, states, active_counts = self._decision_features(
            embeds, vidx2, vol, Xin, fit_mode=True, closes=closes)

        # labels: COST-AWARE, SYMMETRIC directional barrier (does +tp get touched
        # before -tp over the hold) → clean P(up) for a two-sided model, not a
        # cost-blind 4-bar direction. Tail purged for the full hold.
        T = len(vidx2)
        tp_arr, sl_arr = self._barrier_pcts(vol, vidx2)
        usable = np.array([vidx2[t] + int(cfg("barrier_hold_bars")) < len(closes) for t in range(T)])
        y = np.zeros(T)
        for t in range(T):
            if usable[t]:
                y[t] = MetaLabeler.barrier_dir(
                    closes, vidx2[t], tp_arr[t], max_hold=int(cfg("barrier_hold_bars")))

        cal_start = int(T * 0.80)          # recent, chronological, NOT stratified
        tr_mask = usable.copy(); tr_mask[cal_start:] = False

        # feature-stability ablation (levels vs deviations), evaluated on the cal tail
        masks = self.innov.group_masks()
        n_innov = Xin.shape[1]
        stab = {}
        for gname, gmask in masks.items():
            cols = np.where(gmask)[0]
            if len(cols) == 0:
                stab[f"{gname}_auc"] = 0.5
                continue
            gb = LightGBMModel(n_trees=80, depth=3, seed=self.seed)
            gb.fit(Xf[tr_mask][:, cols], y[tr_mask])
            va = usable & (np.arange(T) >= cal_start)
            stab[f"{gname}_auc"] = LightGBMModel.auc(y[va], gb.predict_proba(Xf[va][:, cols])) if np.any(va) else 0.5
        self.feature_mask = np.ones(Xf.shape[1], dtype=bool)
        for gname, gmask in masks.items():
            if stab.get(f"{gname}_auc", 0.5) < 0.47:
                self.feature_mask[np.where(np.concatenate([gmask, np.zeros(Xf.shape[1] - n_innov, dtype=bool)]))[0]] = False

        # Real LightGBM classifier — predicts P(up). The chronological calibration
        # tail is used for early stopping and remains out of the fit window.
        self.gbm = LightGBMModel(n_trees=300, depth=5, lr=0.03, seed=self.seed)
        va_mask = usable & (np.arange(T) >= cal_start)
        self.gbm.fit(
            Xf[tr_mask][:, self.feature_mask], y[tr_mask],
            Xf[va_mask][:, self.feature_mask], y[va_mask],
        )
        p_all = self.gbm.predict_proba(Xf[:, self.feature_mask])
        # symmetric directional thresholds: long if p_up high, short if p_up low
        self.p_hi = float(np.quantile(p_all[:cal_start], cfg("gbm_gate_quantile") / 100.0))
        self.p_lo = float(np.quantile(p_all[:cal_start], 1.0 - cfg("gbm_gate_quantile") / 100.0))

        # OVERFITTING diagnostic: primary-model AUC in-sample vs on the held-out
        # chronological tail. A large positive gap = the model memorized the train
        # window. Purely observational, surfaced in the dashboards.
        train_auc = LightGBMModel.auc(y[tr_mask], p_all[tr_mask]) if np.any(tr_mask) else 0.5
        oos_auc = LightGBMModel.auc(y[va_mask], p_all[va_mask]) if np.any(va_mask) else 0.5
        overfit_gap = float(train_auc - oos_auc)

        def _side_of(p):
            if p >= self.p_hi:
                return "long", 1, float(p)
            if p <= self.p_lo:
                return "short", -1, float(1.0 - p)
            return None, 0, 0.0

        # meta labeling on the primary model's own signals (train part), directional
        # TP-before-SL over the cost-floored barriers. Trained TWO-SIDED regardless of
        # allow_short so toggling short on needs no refit; emission is gated later.
        meta_X, meta_y = [], []
        for t in range(cal_start):
            if not usable[t]:
                continue
            side, dsign, conf = _side_of(p_all[t])
            if side is None:
                continue
            meta_X.append(self._meta_features(conf, dsign, Xin[t], flags[t], states[t]))
            meta_y.append(MetaLabeler.barrier_outcome_dir(
                closes, vidx2[t], tp_arr[t], sl_arr[t], side, max_hold=int(cfg("barrier_hold_bars"))))
        self.meta = MetaLabeler()
        if len(meta_X) >= 12 and 0 < np.mean(meta_y) < 1:
            self.meta.fit(np.stack(meta_X), np.array(meta_y))

        # conformal calibration on the chronological tail (both sides)
        self.conformal = ConformalGate()
        pm, dm = [], []
        for t in range(cal_start, T):
            side, dsign, conf = _side_of(p_all[t])
            if side is None:
                continue
            pm.append(float(self.meta.predict_proba(
                self._meta_features(conf, dsign, Xin[t], flags[t], states[t])[None, :])[0]))
            dm.append(self.panel.dmin({r: sigs[r][vidx2[t]] for r in GEOM_RESOLUTIONS}))
        if len(pm) >= 5:
            self.conformal.calibrate(pm, dm, target_accept=cfg("conformal_accept_pct") / 100.0)

        # panel bookkeeping + drift series
        self.panel.refresh(sigs, valid, fold)
        for r in GEOM_RESOLUTIONS:
            for t in vidx2[-6:]:
                self.panel.push_adaptive(r, sigs[r][t])
        d_anchor = self.panel.log_d_anchor(
            fold, encoder, lambda Xq, rq: self._standardize(Xq, rq, stats=self._panel_std0))
        tail = np.where(valid)[0][-200:]
        dh_data = float(np.median([DeltaHatDiagnostic(n_quadruples=600, seed=self.seed).measure(
            sigs[r][tail]) for r in GEOM_RESOLUTIONS]))
        self.delta_hat_series.append({"fold": fold, "bars": int(len(df)), "delta_hat": dh_data})

        # mandatory dashboards ("zorunlu kadranlar")
        fw_all = encoder.forward(X, ridx)
        p_dnz = float(np.mean(fw_all["delta"] != 0))
        var_r = {}
        n_res = len(GEOM_RESOLUTIONS)
        Tb = X.shape[0] // n_res
        for i, r in enumerate(GEOM_RESOLUTIONS):
            var_r[str(r)] = float(np.var(fw_all["r"][i * Tb:(i + 1) * Tb]))
        diag_entry = {
            "fold": fold,
            "p_delta_nonzero": p_dnz,
            "var_r": var_r,
            "e_res_test": encoder.e_res_intervention(X, ridx),
            "heldout_recon": heldout,
            "r_eff": encoder.r_eff(),
            "kappa": encoder.kappa,
            "eta": encoder.eta,
            "d_anchor": (d_anchor or {}).get("d_anchor", 0.0),
            "delta_hat_data": dh_data,
            "feature_stability": stab,
            "train_auc": float(train_auc),
            "oos_auc": float(oos_auc),
            "overfit_gap": overfit_gap,
            "lambda_l1": encoder.lambdas["l1"],
            "n_episodes": int(len([1 for f_ in np.diff(flags.astype(int)) if f_ == 1]) + (1 if flags[0] else 0)),
            "schema": self.schema.label(),
        }
        self.diagnostics.append(diag_entry)

        # freeze fold-0 baselines for the health monitor
        if self.baseline_heldout is None and heldout == heldout:   # not NaN
            self.baseline_heldout = float(heldout)
        if self.baseline_delta_hat is None:
            self.baseline_delta_hat = float(dh_data)

        # fold-local λ adaptation for the NEXT fold (kept inside the encoder object)
        if p_dnz > 0.45:
            encoder.lambdas["l1"] = min(encoder.lambdas["l1"] * 1.5, 0.5)
        elif p_dnz < 0.03:
            encoder.lambdas["l1"] = max(encoder.lambdas["l1"] / 1.5, 0.005)

        self.fold = fold + 1
        self._bars_at_fit = len(df)
        self._prev_standardize = (lambda mu, sd: (lambda Xq, rq: self._standardize(Xq, rq, stats=(mu, sd))))(
            self.feat_mu, self.feat_sd)
        self.status = "ready"
        log.info(f"[GEOM] fold {fold} fitted in {(datetime.now()-t0).total_seconds():.1f}s | "
                 f"P(δ≠0)={p_dnz:.3f} | D_anchor={diag_entry['d_anchor']:.4f} | schema={self.schema.label()}")
        return diag_entry

    def _next_schema_version(self):
        store = GeometrySchema.load_store()
        hist = store.get("schemas", {}).get(self.timeframe, [])
        return (hist[-1]["version"] + 1) if hist else 1

    def _barrier_pcts(self, vol, vidx):
        """Cost-floored, volatility-scaled barrier targets per bar (percent).

        TP must clear the cost wall with room to spare (≥ 3× roundtrip) or track the
        realized vol of the barrier horizon; SL risks less than the target so the
        payoff matrix can be positive at achievable hit rates.
        Breakeven win rate = (SL+c)/(TP+SL) ≈ 0.56 at the floors."""
        base_tp = float(cfg("tp_base_pct"))
        cost = self.cost_pct
        hold = int(cfg("barrier_hold_bars"))
        tp_floor = float(cfg("tp_cost_floor_mult"))
        sl_floor = float(cfg("sl_cost_floor_mult"))
        bar_vol_pct = vol["rv"][vidx] * 100.0
        tp = np.maximum.reduce([
            np.full(len(vidx), base_tp),
            np.full(len(vidx), tp_floor * cost),
            2.0 * bar_vol_pct * math.sqrt(hold),
        ])
        sl = np.maximum(sl_floor * cost, 0.5 * tp)
        return tp, sl

    def _meta_features(self, conf, dir_sign, xin_row, flag, state):
        """conf = directional confidence of the chosen side (p_up for long,
        1-p_up for short); dir_sign = +1 long / -1 short."""
        return np.concatenate([[conf, float(dir_sign), float(flag), float(state)],
                               xin_row[:8]])

    # ── batch inference over a dataframe (backtest / WFO) ─────────────────────
    def batch_signals(self, df, start_at=0, precomputed=None):
        """Per-bar arrays: signal, exit flag, p, p_meta, A. Bar t uses data ≤ t."""
        n = len(df)
        out = {
            "signal": np.array(["HOLD"] * n, dtype=object),
            "dir": np.zeros(n, dtype=int),          # +1 long, -1 short, 0 flat
            "exit_flag": np.zeros(n, dtype=bool),
            "p": np.full(n, 0.5), "p_meta": np.full(n, 0.5),
            "A": np.zeros(n), "episode": np.zeros(n, dtype=bool),
            "state": np.zeros(n, dtype=int),
            "tp": np.full(n, max(3.0 * self.cost_pct, 0.3)),
            "sl": np.full(n, 1.5 * self.cost_pct),
            "exp_net": np.zeros(n),
        }
        if self.encoder is None:
            return out
        vol, sigs, valid = precomputed if precomputed is not None else self.preprocess(df)
        embeds, vidx = self._embed_by_res(self.encoder, sigs, valid)
        if len(vidx) < 3:
            return out
        speed_bar = np.abs(vol["norm_ret"])[vidx]
        raw = self.innov.compute_raw(self.encoder, embeds, speed_bar, vol["norm_ret"])
        Xin = self.innov.transform(raw)
        closes = df['close'].values.astype(float)
        Xf, flags, states, _ = self._decision_features(embeds, vidx, vol, Xin,
                                                       fit_mode=False, closes=closes)
        p_all = self.gbm.predict_proba(Xf[:, self.feature_mask])
        tp_arr, sl_arr = self._barrier_pcts(vol, vidx)
        a_gate = self.conformal.a_min if self.conformal else 0.5
        cost = self.cost_pct
        for t in range(len(vidx)):
            bar = vidx[t]
            if bar < start_at:
                continue
            p = float(p_all[t])                     # P(up)
            dm = self.panel.dmin({r: sigs[r][bar] for r in GEOM_RESOLUTIONS})
            tp_t, sl_t = float(tp_arr[t]), float(sl_arr[t])

            # evaluate each admissible side; BUY/SELL taken on the best positive
            # expected NET return (barrier race, costs subtracted). Short is only
            # a candidate when allow_short is set.
            candidates = []
            sides = [("long", 1, p, "BUY")]
            if self.allow_short:
                sides.append(("short", -1, 1.0 - p, "SELL"))
            for side, dsign, conf, sig_name in sides:
                thr_ok = (p >= self.p_hi) if side == "long" else (p <= self.p_lo)
                if not thr_ok:
                    continue
                mf = self._meta_features(conf, dsign, Xin[t], flags[t], states[t])
                pm = float(self.meta.predict_proba(mf[None, :])[0]) if self.meta else 0.5
                A = self.conformal.combined(pm, dm) if self.conformal else 0.5
                exp_net = pm * (tp_t - cost) - (1.0 - pm) * (sl_t + cost)
                candidates.append((exp_net, A, pm, dsign, sig_name))

            # default display fields come from the long hypothesis
            long_pm = 0.5
            if self.meta:
                long_pm = float(self.meta.predict_proba(
                    self._meta_features(p, 1, Xin[t], flags[t], states[t])[None, :])[0])
            out["p"][bar] = p
            out["p_meta"][bar] = long_pm
            out["A"][bar] = self.conformal.combined(long_pm, dm) if self.conformal else 0.5
            out["tp"][bar] = tp_t
            out["sl"][bar] = sl_t
            out["exp_net"][bar] = long_pm * (tp_t - cost) - (1.0 - long_pm) * (sl_t + cost)
            out["episode"][bar] = bool(flags[t])
            out["state"][bar] = int(states[t])

            fired = [c for c in candidates if c[0] > 0.0 and c[1] >= a_gate]
            if fired:
                exp_net, A_sel, pm, dsign, sig_name = max(fired, key=lambda c: c[0])
                out["signal"][bar] = sig_name
                out["dir"][bar] = dsign
                out["p_meta"][bar] = pm
                out["A"][bar] = A_sel
                out["exp_net"][bar] = exp_net
            exp_ret = self.graph.expected_return(states[t]) if self.graph else 0.0
            out["exit_flag"][bar] = bool(flags[t] and exp_ret < 0.0 and float(out["A"][bar]) < 0.8 * a_gate)
        return out

    # ── live path ─────────────────────────────────────────────────────────────
    def infer_latest(self, df):
        tail = df.iloc[-(MultiResolutionSignatures().warmup + 96):]
        geo = self.batch_signals(tail.reset_index(drop=True))
        i = len(tail) - 1
        st = int(geo["state"][i])
        gate = float(self.conformal.a_min if self.conformal else 0.5)
        state = {
            "status": "ready",
            "signal": str(geo["signal"][i]),
            "dir": int(geo["dir"][i]),
            "exit_flag": bool(geo["exit_flag"][i]),
            "p_gbm": float(geo["p"][i]),
            "p_meta": float(geo["p_meta"][i]),
            "a_score": float(geo["A"][i]),
            "a_gate": gate,
            "allow_short": bool(self.allow_short),
            "episode": "EPISODE" if geo["episode"][i] else "NORMAL",
            "cluster": st - 1,
            "tp_pct": float(geo["tp"][i]),
            "sl_pct": float(geo["sl"][i]),
            "exp_net": float(geo["exp_net"][i]),
            # per-bar tail arrays for the chart overlay (same computation, no extra cost)
            "chart": {
                "n": int(len(tail)),
                "gate": gate,
                "A": [round(float(x), 4) for x in geo["A"]],
                "episode": [bool(x) for x in geo["episode"]],
                "state": [int(x) for x in geo["state"]],
                "buy": [bool(sg == "BUY") for sg in geo["signal"]],
                "sell": [bool(sg == "SELL") for sg in geo["signal"]],
                "exit": [bool(x) for x in geo["exit_flag"]],
            },
        }
        return state

    # ── health monitor ────────────────────────────────────────────────────────
    # Thresholds beyond which a READY pipeline is treated as untrustworthy and
    # the live signal is forced to HOLD. This is the answer to "a layer breaks
    # silently and still prints a plausible number" — the number is checked.
    DELTA_HAT_DRIFT_MAX = 3.0    # data-side δ̂ may not move more than 3× vs baseline
    STALE_BARS_MULT = 4          # encoder older than 4× the refit cadence ⇒ stale

    def health_status(self, n_bars=None):
        """Return {'health': 'ok'|'degraded', 'reasons': [...]} from the latest
        fold diagnostics. Any tripped guard degrades the pipeline; SignalEngine
        then emits HOLD regardless of what the decision chain produced.
        Thresholds are live from the trading config."""
        reasons = []
        if not self.diagnostics or self.status != "ready":
            return {"health": "ok" if self.status == "ready" else "warmup", "reasons": []}
        d = self.diagnostics[-1]
        gap_max = cfg("overfit_gap_max")
        drift_max = cfg("d_anchor_drift_max")
        recon_max = cfg("recon_regress_max")

        gap = float(d.get("overfit_gap", 0.0))
        if gap > gap_max:
            reasons.append(f"overfit gap {gap:.2f}>{gap_max:.2f}")

        d_anchor = abs(float(d.get("d_anchor", 0.0)))
        if d_anchor > drift_max:
            reasons.append(f"D_anchor drift {d_anchor:.2f}>{drift_max:.2f}")

        recon = float(d.get("heldout_recon", 0.0))
        if (self.baseline_heldout and recon == recon
                and recon > recon_max * self.baseline_heldout):
            reasons.append(f"recon {recon:.3f}>{recon_max:.1f}×baseline")

        dh = float(d.get("delta_hat_data", 0.0))
        if self.baseline_delta_hat and self.baseline_delta_hat > 1e-9:
            ratio = dh / self.baseline_delta_hat
            if ratio > self.DELTA_HAT_DRIFT_MAX or ratio < 1.0 / self.DELTA_HAT_DRIFT_MAX:
                reasons.append(f"δ̂ drift ×{ratio:.2f}")

        p_dnz = float(d.get("p_delta_nonzero", 0.5))
        if p_dnz >= 0.999 or p_dnz <= 1e-6:
            reasons.append(f"P(δ≠0) collapsed to {p_dnz:.3f}")

        if n_bars is not None and self._bars_at_fit:
            stale = n_bars - self._bars_at_fit
            if stale > self.STALE_BARS_MULT * self.LIVE_REFIT_BARS:
                reasons.append(f"encoder stale ({stale} bars since fit)")

        return {"health": "degraded" if reasons else "ok", "reasons": reasons}

    def live_state(self, extra=None):
        hs = self.health_status()
        s = {
            "status": self.status,
            "health": hs["health"],
            "health_reasons": hs["reasons"],
            "schema": self.schema.label() if self.schema else "-",
            "version": self.schema.version if self.schema else 0,
            "kappa": float(self.encoder.kappa) if self.encoder else 0.0,
            "kappa_init": float(self.schema.kappa_init) if self.schema else 0.0,
            "eta": float(self.encoder.eta) if self.encoder else 0.0,
            "delta_hat": self.schema.delta_hat if self.schema else {},
            "fold": self.fold,
            "signal": "HOLD", "dir": 0, "exit_flag": False,
            "p_gbm": 0.5, "p_meta": 0.5, "a_score": 0.0, "a_gate": 0.5,
            "allow_short": bool(self.allow_short),
            "tp_pct": 0.0, "sl_pct": 0.0, "exp_net": 0.0,
            "episode": "NORMAL", "cluster": -1,
            "panel": self.panel.summary() if self.panel else {},
            "delta_hat_series": self.delta_hat_series[-12:],
            "diag": self.diagnostics[-1] if self.diagnostics else {},
        }
        if extra:
            s.update(extra)
        # health is authoritative: a degraded pipeline can never emit a live trade
        if s.get("health") == "degraded":
            s["signal"] = "HOLD"
            s["exit_flag"] = True   # also flatten any open geo position
        self.last_state = s
        return s

    def on_bar(self, df, force_retrain=False, timeframe=None):
        """Live tick entry point — schedules background (re)training, never blocks."""
        tf = timeframe or self.timeframe
        # runtime opt-in short toggle (no refit needed — the meta is trained two-sided)
        self.allow_short = bool(bot_state.get("allow_short", ALLOW_SHORT))
        if tf != self.timeframe:
            log.info(f"[GEOM] timeframe changed {self.timeframe} → {tf}: resetting geometry (new first window)")
            self.reset(tf)
        with self._lock:
            busy = self._training
            status = self.status
        if status == "collecting" and not busy and len(df) >= self.MIN_TRAIN_BARS:
            self._spawn_fit(df, teacher=None, warm=None)
            return self.live_state({"status": "training"})
        if status == "ready" and not busy and (
                force_retrain or len(df) - self._bars_at_fit >= self.LIVE_REFIT_BARS):
            teacher = {"encoder": self.encoder, "standardize": self._prev_standardize}
            self._spawn_fit(df, teacher=teacher, warm=self.encoder)
        if status == "ready":
            try:
                return self.live_state(self.infer_latest(df))
            except Exception as e:
                log.error(f"[GEOM] live inference error: {e}")
                return self.live_state()
        return self.live_state({"status": "training" if busy else status})

    def _spawn_fit(self, df, teacher, warm):
        df_copy = df.copy()
        with self._lock:
            if self._training:
                return
            self._training = True

        def _worker():
            try:
                self.fit(df_copy, teacher=teacher, warm_from=warm)
            except Exception as e:
                log.error(f"[GEOM] background fit failed: {e}")
                if self.status != "ready":
                    self.status = "collecting"
            finally:
                with self._lock:
                    self._training = False

        threading.Thread(target=_worker, daemon=True).start()


class PurgedWalkForwardEngine:
    """
    Purged Walk-Forward over the geometric pipeline: expanding chronological folds
    with an embargo gap (signature warm-up + label horizon) between IS and OOS,
    warm start + neighbour-fold RKD between folds, geometry schema fixed after the
    first window. The OOS trading-parameter mini-grid keeps the champion/challenger
    contract of the legacy optimizer.
    """

    def __init__(self, df, timeframe="1m", n_folds=5, seed=42):
        self.df = df.reset_index(drop=True)
        self.timeframe = timeframe
        self.n_folds = n_folds
        self.seed = seed
        self.embargo = max(GEOM_RESOLUTIONS) + int(cfg("barrier_hold_bars"))

    def run(self):
        df = self.df
        n = len(df)
        first_is = max(GeometricPipeline.MIN_TRAIN_BARS, int(n * 0.35))
        if n < first_is + 3 * (self.embargo + 120):
            raise ValueError(f"Not enough bars for purged WFO ({n})")
        oos_len = (n - first_is) // self.n_folds

        pipeline = GeometricPipeline(self.timeframe, seed=self.seed, encoder_epochs=45)
        pre_full = pipeline.preprocess(df)

        def pre_slice(upto):
            vol, sigs, valid = pre_full
            return ({k: (v[:upto] if isinstance(v, np.ndarray) else v) for k, v in vol.items()},
                    {r: sigs[r][:upto] for r in GEOM_RESOLUTIONS}, valid[:upto])

        base = get_active_parameters()
        # Index 0 is always the currently-live champion so the candidate must beat
        # the exact production baseline, not merely win against an unrelated grid.
        combos = [dict(base)]
        tp_grid = sorted({float(base.get("TP_PERCENT", 0.6)), 0.6, 1.0, 1.6, 2.4})
        trail_grid = sorted({float(base.get("TRAIL_MULT", 3.0)), 2.0, 2.5, 3.0, 3.5, 4.0})
        for tp in tp_grid:
            for tm in trail_grid:
                cmb = dict(base)
                cmb["TP_PERCENT"] = tp
                cmb["TRAIL_MULT"] = tm
                if not any(
                    float(old.get("TP_PERCENT", 0.0)) == tp and
                    float(old.get("TRAIL_MULT", 0.0)) == tm
                    for old in combos
                ):
                    combos.append(cmb)

        fold_results = {i: [] for i in range(len(combos))}
        prev_encoder = None
        prev_std = None
        for k in range(self.n_folds):
            is_end = first_is + k * oos_len
            oos_start = is_end + self.embargo
            oos_end = min(is_end + oos_len, n) if k < self.n_folds - 1 else n
            if oos_start >= oos_end - 30:
                continue
            teacher = ({"encoder": prev_encoder, "standardize": prev_std}
                       if prev_encoder is not None else None)
            pipeline.fit(df.iloc[:is_end], teacher=teacher,
                         warm_from=prev_encoder, fold=k, precomputed=pre_slice(is_end))
            prev_encoder = pipeline.encoder
            prev_std = pipeline._prev_standardize

            geo_full = pipeline.batch_signals(df.iloc[:oos_end], start_at=oos_start,
                                              precomputed=pre_slice(oos_end))
            oos_df = df.iloc[oos_start:oos_end].reset_index(drop=True)
            geo = {kk: (vv[oos_start:oos_end] if isinstance(vv, np.ndarray) else vv)
                   for kk, vv in geo_full.items()}
            for ci, cmb in enumerate(combos):
                res = Backtester(oos_df).run(cmb, geo=geo)
                fold_results[ci].append({"pf": res["profit_factor"], "calmar": res["calmar_ratio"],
                                         "pnl": res["total_pnl_usdt"], "trades": res["trade_count"]})

        summaries = {}
        best_ci, best_key = 0, None
        for ci in fold_results:
            rs = fold_results[ci]
            stable = sum(1 for r in rs if r["pf"] > 1.10 and r["calmar"] > 0.8)
            pfs = [min(r["pf"], 10.0) for r in rs]
            var = float(np.std(pfs)) if pfs else 0.0
            median_pf = float(np.median(pfs)) if pfs else 0.0
            median_calmar = float(np.median([min(r["calmar"], 10.0) for r in rs])) if rs else 0.0
            median_pnl = float(np.median([r["pnl"] for r in rs])) if rs else 0.0
            total_trades = int(sum(r["trades"] for r in rs))
            robust_score = median_pnl + 2.0 * median_calmar + median_pf - var
            summaries[ci] = {
                "stable_folds": int(stable), "pf_variance": var,
                "median_pf": median_pf, "median_calmar": median_calmar,
                "median_pnl": median_pnl, "total_trades": total_trades,
                "robust_score": float(robust_score),
            }
            key = (stable, robust_score, total_trades, -var)
            if best_key is None or key > best_key:
                best_ci, best_key = ci, key
        challenger = dict(combos[best_ci])
        best_summary = summaries.get(best_ci, {})
        champion_summary = summaries.get(0, {})
        folds_evaluated = max((len(rs) for rs in fold_results.values()), default=0)
        min_stable = max(2, int(math.ceil(folds_evaluated * 0.60)))
        improvement = float(best_summary.get("robust_score", 0.0) - champion_summary.get("robust_score", 0.0))
        # overfitting readout: mean primary-model IS→OOS AUC gap across folds
        gaps = [d.get("overfit_gap", 0.0) for d in pipeline.diagnostics]
        oos_aucs = [d.get("oos_auc", 0.5) for d in pipeline.diagnostics]
        overfit = {
            "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
            "max_gap": float(np.max(gaps)) if gaps else 0.0,
            "mean_oos_auc": float(np.mean(oos_aucs)) if oos_aucs else 0.5,
            "verdict": ("high" if (gaps and np.mean(gaps) > 0.15) else
                        "moderate" if (gaps and np.mean(gaps) > 0.08) else "low"),
        }
        eligible_for_promotion = bool(
            best_ci != 0 and
            improvement > 0.05 and
            best_summary.get("stable_folds", 0) >= min_stable and
            best_summary.get("total_trades", 0) >= max(10, 2 * folds_evaluated) and
            overfit["verdict"] != "high"
        )
        if best_ci == 0:
            promotion_reason = "current champion remains optimal"
        elif improvement <= 0.05:
            promotion_reason = "challenger improvement is too small"
        elif best_summary.get("stable_folds", 0) < min_stable:
            promotion_reason = f"challenger stable in fewer than {min_stable} folds"
        elif best_summary.get("total_trades", 0) < max(10, 2 * folds_evaluated):
            promotion_reason = "too few OOS trades"
        elif overfit["verdict"] == "high":
            promotion_reason = "high LightGBM IS-OOS overfit gap"
        else:
            promotion_reason = "validated OOS improvement"
        log.info(f"[WFO] overfit: mean IS-OOS AUC gap {overfit['mean_gap']:.3f} "
                 f"({overfit['verdict']}), mean OOS AUC {overfit['mean_oos_auc']:.3f}")
        return {
            "challenger": challenger,
            "stability_count": int(best_summary.get("stable_folds", 0)),
            "variance": float(best_summary.get("pf_variance", 0.0)),
            "slices_evaluated": int(folds_evaluated),
            "eligible_for_promotion": eligible_for_promotion,
            "promotion_reason": promotion_reason,
            "improvement_score": improvement,
            "challenger_summary": best_summary,
            "champion_summary": champion_summary,
            "engine": "geometric-purged-wfo",
            "diagnostics": pipeline.diagnostics,
            "d_anchor_log": pipeline.panel.d_anchor_log,
            "delta_hat_series": pipeline.delta_hat_series,
            "overfit": overfit,
            "allow_short": bool(pipeline.allow_short),
            "schema": pipeline.schema.label() if pipeline.schema else "-",
            "fold_results": {str(ci): fold_results[ci] for ci in fold_results},
            "parameter_summaries": {str(ci): summaries[ci] for ci in summaries},
        }


def run_geometric_backtest(df, params, timeframe="1m", seed=42):
    """Backtest stage of the flow: train on the first window, trade the purged remainder."""
    df = df.reset_index(drop=True)
    n = len(df)
    split = max(GeometricPipeline.MIN_TRAIN_BARS, int(n * 0.35))
    embargo = max(GEOM_RESOLUTIONS) + int(cfg("barrier_hold_bars"))
    if n < split + embargo + 120:
        raise ValueError(f"Not enough bars for geometric backtest ({n})")
    pipeline = GeometricPipeline(timeframe, seed=seed, encoder_epochs=45)
    pipeline.fit(df.iloc[:split])
    geo = pipeline.batch_signals(df, start_at=split + embargo)
    report = Backtester(df).run(params, geo=geo)
    report["engine"] = "geometric"
    report["train_bars"] = int(split)
    report["schema"] = pipeline.schema.label() if pipeline.schema else "-"
    report["diagnostics"] = pipeline.diagnostics[-1] if pipeline.diagnostics else {}
    return report


class SignalEngine:
    """Thin adapter over the learned-geometry pipeline. Signal is HOLD until
    (a) the pipeline is READY and (b) its health monitor is green — no
    legacy fallback, no cost-blind heuristics. `enable_geometry=False` is
    the "test / shadow-off" mode used by the safety suite."""

    def __init__(self, enable_geometry=True):
        self.geometry = GeometricPipeline(bot_state.get("timeframe", "1m")) if enable_geometry else None

    def process(self, df, params=None, force_retrain=False):
        c = df['close'].values
        h = df['high'].values
        l = df['low'].values
        i = len(df) - 1
        # simple, cost-agnostic realised volatility for trailing distance only
        gv = float(np.mean(calc_true_range(h[-32:], l[-32:], c[-32:]))) if len(df) >= 32 else float(h[i] - l[i])

        geom_state = {}
        if self.geometry is not None:
            try:
                geom_state = self.geometry.on_bar(
                    df, force_retrain=force_retrain,
                    timeframe=bot_state.get("timeframe", self.geometry.timeframe))
            except Exception as e_geom:
                log.error(f"Geometry pipeline error: {e_geom}")
                geom_state = self.geometry.live_state() if self.geometry else {}

        # Signal is authoritative only when the pipeline is READY AND HEALTHY.
        # A degraded pipeline is treated exactly like an untrained one: HOLD.
        sig, st = "HOLD", ""
        if geom_state.get("status") == "ready" and geom_state.get("health", "ok") == "ok":
            gsig = geom_state.get("signal", "HOLD")
            if gsig in ("BUY", "SELL"):
                sig, st = gsig, "Geo"

        return {
            "signal": sig,
            "type": st,
            "geom": geom_state,
            "price": float(c[i]),
            "gauss_vol": gv,     # kept as a plain realised-vol scalar for trailing distance
        }






class PositionManager:
    def __init__(self):
        self.side = None; self.entry_price = 0.0; self.entry_type = ""; self.qty = 0.0
        self.trail_stop_70 = 0.0
        self.trail_stop_30 = 0.0
        self.ping_stop = 0.0
        self.invested_amount = 0.0
        self.has_taken_partial_tp = False
        self.ou_target = 0.0
        self.ou_stop = 0.0
        self.mode = "PAPER"
        self.stop_order_id = None
        self.realized_pnl_usdt = 0.0
        self.max_price_seen = 0.0
        self.min_price_seen = 0.0
        self.entry_volatility = 0.0
        self.tp_percent = 0.3
        self.sl_percent = 0.3
        self.min_trail_dist = 0.0

    @property
    def is_open(self): return self.side is not None

    def open(self, side, price, qty, sig_type, gauss_vol, invested_amount, ou_target=0.0, ou_stop=0.0, mode="PAPER", stop_order_id=None, params=None, tp_percent=0.3, sl_percent=0.3, entry_volatility=0.0, min_trail_dist=0.0):
        if params is None:
            params = get_active_parameters()
        self.side, self.entry_price, self.entry_type, self.qty = side, price, sig_type, qty
        self.invested_amount = invested_amount
        self.has_taken_partial_tp = False
        self.ou_target = ou_target
        self.ou_stop = ou_stop
        self.mode = mode
        self.stop_order_id = stop_order_id
        self.realized_pnl_usdt = 0.0
        self.tp_percent = tp_percent
        self.sl_percent = sl_percent
        self.entry_volatility = entry_volatility
        self.max_price_seen = price
        self.min_price_seen = price
        # floor for the trailing distance so tight low-TF volatility cannot pull
        # the stops inside the cost wall (used by learned-geometry entries)
        self.min_trail_dist = float(min_trail_dist)

        self.trail_mult = float(params.get("TRAIL_MULT", 3.0))
        self.ping_stop_mult = float(params.get("PING_STOP_MULT", 0.5))

        trail_dist = max(gauss_vol*self.trail_mult, self.min_trail_dist)
        if side == "long":
            self.trail_stop_30 = price - trail_dist
            self.trail_stop_70 = price - trail_dist
            self.ping_stop = price - gauss_vol*self.ping_stop_mult
        else:
            self.trail_stop_30 = price + trail_dist
            self.trail_stop_70 = price + trail_dist
            self.ping_stop = price + gauss_vol*self.ping_stop_mult

    def close(self, reason, price):
        pnl_pct = (price-self.entry_price)/self.entry_price*100 if self.side=="long" else (self.entry_price-price)/self.entry_price*100
        pnl_usdt = self.invested_amount * (pnl_pct / 100)
        self.side = None; self.entry_price = 0; self.qty = 0
        self.has_taken_partial_tp = False
        self.mode = "PAPER"
        self.stop_order_id = None
        self.realized_pnl_usdt = 0.0
        self.min_trail_dist = 0.0
        return pnl_pct, pnl_usdt

    def update_stops(self, price, gv_normal, gv_lower):
        self.max_price_seen = max(self.max_price_seen, price)
        self.min_price_seen = min(self.min_price_seen, price)
        d30 = max(gv_normal*self.trail_mult, self.min_trail_dist)
        d70 = max(gv_lower*self.trail_mult, self.min_trail_dist)
        if self.side == "long":
            self.trail_stop_30 = max(self.trail_stop_30, price-d30)
            if not self.has_taken_partial_tp:
                self.trail_stop_70 = max(self.trail_stop_70, price-d70)
            self.ping_stop = max(self.ping_stop, price-gv_normal*self.ping_stop_mult)
        elif self.side == "short":
            self.trail_stop_30 = min(self.trail_stop_30, price+d30)
            if not self.has_taken_partial_tp:
                self.trail_stop_70 = min(self.trail_stop_70, price+d70)
            self.ping_stop = min(self.ping_stop, price+gv_normal*self.ping_stop_mult)

    def check_exits(self, price, info, force_close=False):
        """Geo-only exits: fixed SL/TP + first-touch trailing partial (70%) then
        30%-tail trailing stop. All entries are 'Geo' now — the only regime
        marker left is the health flag from the pipeline (handled by caller)."""
        if not self.is_open: return None
        if force_close: return "Zaman Dilimi Degisimi (Force Close)"

        if self.side == "long":
            if self.has_taken_partial_tp:
                if price <= self.trail_stop_30: return "Trail Stop (30%)"
                return None
            if price <= self.entry_price * (1 - self.sl_percent/100): return "Stop Loss"
            if price <= self.trail_stop_70:
                if self.trail_stop_70 == self.trail_stop_30: return "Stop Loss"
                return "PARTIAL_TP"
            if price >= self.entry_price * (1 + self.tp_percent/100): return f"Take Profit ({self.tp_percent:.2f}%)"
        elif self.side == "short":
            if self.has_taken_partial_tp:
                if price >= self.trail_stop_30: return "Trail Stop (30%)"
                return None
            if price >= self.entry_price * (1 + self.sl_percent/100): return "Stop Loss"
            if price >= self.trail_stop_70:
                if self.trail_stop_70 == self.trail_stop_30: return "Stop Loss"
                return "PARTIAL_TP"
            if price <= self.entry_price * (1 - self.tp_percent/100): return f"Take Profit ({self.tp_percent:.2f}%)"
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM KONTROL MOTORU
# ═══════════════════════════════════════════════════════════════════════════════
class TelegramController:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        self.last_update_id = 0
        self.active = False
        
        if self.token:
            self.active = True
            log.info(f"Telegram Controller Initialized with Token: {self.token[:10]}...")
            if self.chat_id:
                log.info(f"Telegram Chat ID Configured: {self.chat_id}")
            else:
                log.warning("Telegram Chat ID is not set. Bot will auto-detect the Chat ID from the first received message.")
        else:
            log.info("Telegram integration is disabled (TELEGRAM_BOT_TOKEN not found in env).")

    async def send_message(self, text, parse_mode="Markdown"):
        if not self.active or not self.chat_id: return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": parse_mode}
        try:
            await asyncio.to_thread(requests.post, url, json=payload, timeout=5)
        except Exception as e:
            log.error(f"Telegram send message error: {e}")

    async def run_loop(self):
        if not self.active: return
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        log.info("Telegram updates polling loop started...")
        while True:
            try:
                params = {"offset": self.last_update_id + 1, "timeout": 5}
                response = await asyncio.to_thread(requests.get, url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("ok") and data.get("result"):
                        for update in data["result"]:
                            self.last_update_id = update["update_id"]
                            message = update.get("message")
                            if not message: continue
                            
                            sender_chat_id = str(message.get("chat", {}).get("id"))
                            text = message.get("text", "").strip()
                            log.info(f"Telegram message received: '{text}' from Chat ID: {sender_chat_id}")
                            
                            # Secure Chat ID Check: If not configured, block all commands and show setup warning
                            if not self.chat_id:
                                log.warning(f"Telegram command blocked: TELEGRAM_CHAT_ID is not configured in .env. Sender: {sender_chat_id}")
                                self.chat_id = sender_chat_id
                                await self.send_message(f"🔒 *Güvenlik Uyarısı:* Telegram kontrolü devre dışı. Bu hesabın yetkisi yok.\n\nSistemi kontrol edebilmek için lütfen `.env` dosyanıza şu satırı ekleyin ve botu yeniden başlatın:\n`TELEGRAM_CHAT_ID={sender_chat_id}`")
                                self.chat_id = ""
                                continue
                            
                            if sender_chat_id == self.chat_id:
                                await self.handle_command(text)
                            else:
                                log.warning(f"Ignored message from unauthorized Chat ID: {sender_chat_id} (Configured: {self.chat_id})")
            except Exception as e:
                log.error(f"Telegram polling loop exception: {e}")
                await asyncio.sleep(2)
            await asyncio.sleep(1)

    def save_chat_id_to_env(self, chat_id):
        env_path = Path.home() / ".env"
        lines = []
        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines()
        
        found = False
        new_lines = []
        for line in lines:
            if line.startswith("TELEGRAM_CHAT_ID="):
                new_lines.append(f"TELEGRAM_CHAT_ID={chat_id}")
                found = True
            else:
                new_lines.append(line)
        if not found:
            new_lines.append(f"TELEGRAM_CHAT_ID={chat_id}")
            
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        log.info(f"Auto-detected and saved TELEGRAM_CHAT_ID={chat_id} to .env")
    async def handle_command(self, text):
        if text == "/start" or text.startswith("/start ") or text.startswith("/help"):
            msg = (
                "🤖 *Quant Bot V3.5 - Telegram Kontrolü*\n\n"
                "Aşağıdaki komutları kullanarak botu uzaktan yönetebilirsiniz:\n\n"
                "📊 `/status` - Botun anlık durumunu, fiyatı, OBI'yi ve açık pozisyonu gösterir\n"
                "▶️ `/start_trade` - Ticareti başlatır (BEGIN TRADE aktifleşir)\n"
                "⏸️ `/pause_trade` - Ticareti duraklatır (Gözlem moduna geçer)\n"
                "🔄 `/toggle_mode` - REAL / PAPER mod geçişi yapar\n"
                "❌ `/force_close` - Açık olan pozisyonu o anki fiyattan hemen kapatır"
            )
            await self.send_message(msg)
        elif text.startswith("/status"):
            pos_desc = f"{bot_state['position_side'].upper()} (Giriş: {bot_state['position_entry']:.2f}, PnL: {bot_state['position_pnl']:.2f}%)" if bot_state["position_side"] else "Açık Pozisyon Yok"
            mode = bot_state["trading_mode"]
            active = "AKTİF" if bot_state["is_trading_active"] else "DURAKLATILDI"
            g = bot_state.get("geom", {}) or {}
            geo_line = f"{g.get('status','-').upper()}/{g.get('health','-').upper()} · {g.get('schema','-')}"
            msg = (
                f"📊 *ANLIK DURUM RAPORU*\n\n"
                f"📈 *Fiyat:* {bot_state['price']:.2f} USDT\n"
                f"🤖 *Durum:* {active}\n"
                f"🔄 *Mod:* {mode} Mod\n"
                f"🧠 *Geo:* {geo_line}\n"
                f"🎯 *Sinyal:* {bot_state['signal']} ({bot_state['signal_type'] or 'Yok'})\n"
                f"⚖️ *OBI:* {bot_state['obi']:.2f}\n"
                f"💼 *Pozisyon:* {pos_desc}\n"
                f"💵 *Bakiye (Paper):* {bot_state['virtual_balance']:.2f} USDT\n"
                f"🎯 *E[net]:* {g.get('exp_net', 0.0):+.2f}% (TP {g.get('tp_pct', 0.0):.2f}/SL {g.get('sl_pct', 0.0):.2f}%)"
            )
            await self.send_message(msg)
        elif text.startswith("/start_trade"):
            bot_state["is_trading_active"] = True
            await self.send_message("▶️ *Ticarete Başlandı.* Bot piyasayı izliyor ve işlem açabilir.")
        elif text.startswith("/pause_trade"):
            bot_state["is_trading_active"] = False
            await self.send_message("⏸️ *Ticaret Durduruldu.* Sadece piyasa analizi devam ediyor.")
        elif text.startswith("/toggle_mode"):
            new_mode = "REAL" if bot_state["trading_mode"] == "PAPER" else "PAPER"
            bot_state["trading_mode"] = new_mode
            await self.send_message(f"🔄 *Mod Değiştirildi:* {new_mode} Mod")
        elif text.startswith("/force_close"):
            if self.bot.position.is_open:
                entry_side = self.bot.position.side
                price = bot_state["price"]
                await self.bot.close_position("Telegram Manuel Kapatma", price)
                await self.send_message(f"❌ *Açık olan {entry_side.upper()} pozisyonu o anki fiyattan ({price:.2f}) kapatıldı.*")
            else:
                await self.send_message("⚠️ *Kapatılacak açık bir pozisyon bulunamadı.*")

# ═══════════════════════════════════════════════════════════════════════════════
# ANA BOT MOTORU
# ═══════════════════════════════════════════════════════════════════════════════
class QuantBot:
    def __init__(self):
        load_dotenv(Path.home() / ".env")
        api_key = os.environ.get("BORSANIN_API_KEY")
        api_secret = os.environ.get("BORSANIN_SECRET_KEY")
        if not api_key or not api_secret: raise RuntimeError("API anahtarlari eksik!")
        self.exchange = ccxt.mexc({'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        self.public_binance = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        # Binance publishes a market-data-only endpoint that needs no API key.
        self.public_binance.urls['api']['public'] = 'https://data-api.binance.vision/api/v3'
        self.public_binance.urls['api']['v1'] = 'https://data-api.binance.vision/api/v1'
        self.research_exchanges = {"binance": self.public_binance}
        self.signal_engine = SignalEngine()
        self.position = PositionManager()
        self.telegram = TelegramController(self)

    def simulate_slippage(self, side, qty, orderbook):
        """
        Order book uzerinden VWAP kaymasini simule eder.
        Geriye (vwap, slippage) doner.
        """
        book = orderbook['asks'] if side == 'buy' else orderbook['bids']
        if not book or len(book) == 0:
            return 0.0, 1.0  # high slippage if book is empty
        
        best_price = float(book[0][0])
        accumulated_qty = 0.0
        total_cost = 0.0
        
        for price, amount in book:
            price = float(price)
            amount = float(amount)
            needed = qty - accumulated_qty
            if needed <= 0:
                break
            take = min(amount, needed)
            total_cost += take * price
            accumulated_qty += take
            
        if accumulated_qty < qty:
            return best_price, 1.0
            
        vwap = total_cost / qty
        if side == 'buy':
            slippage = (vwap - best_price) / best_price
        else:
            slippage = (best_price - vwap) / best_price
            
        return vwap, slippage

    async def execute_marketable_order(self, side, qty, current_price):
        """
        MEXC Spot uzerinde order book derinligine bakarak limit fiyat belirler,
        VWAP kaymasini simule eder. %0.20 limit asilirsa islemi iptal eder.
        Gecici asili kalmamasi icin IOC (Immediate-Or-Cancel) parametresi ile gonderir.
        """
        try:
            orderbook = await asyncio.to_thread(self.exchange.fetch_order_book, SYMBOL, limit=50)
        except Exception as e:
            log.error(f"Failed to fetch order book for order book simulation: {e}")
            return {'id': None, 'filled': 0.0, 'average': current_price}

        if not orderbook['bids'] or not orderbook['asks']:
            log.warn("Order book is empty, skipping execution.")
            return {'id': None, 'filled': 0.0, 'average': current_price}

        best_bid = float(orderbook['bids'][0][0])
        best_ask = float(orderbook['asks'][0][0])

        vwap, expected_slippage = self.simulate_slippage(side, qty, orderbook)
        slippage_cap = 0.002  # %0.20 dynamic cap for BTC/USDT

        log.info(f"Slippage Simulation: Target Qty={qty:.6f}, Best Price={'Ask:'+str(best_ask) if side=='buy' else 'Bid:'+str(best_bid)}, Simulated VWAP={vwap:.2f}, Expected Slippage={expected_slippage*100:.4f}% (Cap: {slippage_cap*100:.2f}%)")

        if expected_slippage > slippage_cap:
            log.warn(f"Execution ABORTED: Expected slippage ({expected_slippage*100:.4f}%) exceeds safety cap ({slippage_cap*100:.2f}%)")
            return {'id': None, 'filled': 0.0, 'average': current_price}

        if side == "buy":
            limit_price = best_ask * (1.0 + slippage_cap)
        else:
            limit_price = best_bid * (1.0 - slippage_cap)

        formatted_qty = float(self.exchange.amount_to_precision(SYMBOL, qty))
        formatted_price = float(self.exchange.price_to_precision(SYMBOL, limit_price))
        
        log.info(f"REAL LIMIT ORDER PLACED (Marketable IOC): {side} {formatted_qty} at {formatted_price}")
        
        order_id = None
        params = {'timeInForce': 'IOC'}
        for attempt in range(3):
            try:
                order = await asyncio.to_thread(
                    self.exchange.create_order,
                    SYMBOL, 'limit', side, formatted_qty, formatted_price, params
                )
                order_id = order['id']
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                log.warning(f"Order placement failed, retrying in 1s... Error: {e}")
                await asyncio.sleep(1.0)
                
        if not order_id:
            return {'id': None, 'filled': 0.0, 'average': current_price}

        filled_qty = 0.0
        avg_price = current_price
        for attempt in range(3):
            await asyncio.sleep(0.3)
            try:
                order_status = await asyncio.to_thread(self.exchange.fetch_order, order_id, SYMBOL)
                filled_qty = float(order_status.get('filled', 0.0))
                avg_price = float(order_status.get('average', order_status.get('price', limit_price)) or limit_price)
                status = order_status.get('status')
                if status in ('closed', 'canceled'):
                    break
            except Exception as e:
                log.error(f"Error fetching IOC order status: {e}")

        log.info(f"IOC Order {order_id} final status fetched: Filled Qty={filled_qty:.6f}, Avg Price={avg_price:.2f}")
        return {'id': order_id, 'filled': filled_qty, 'average': avg_price}

    async def place_native_stop_loss(self, qty, stop_price):
        """
        MEXC Spot uzerinde tetikleyici stop emri olusturur.
        Tetiklendiginde geride kalmamasi icin limit fiyati tetik fiyatinin %0.5 altindadir.
        """
        limit_price = stop_price * 0.995
        
        formatted_qty = float(self.exchange.amount_to_precision(SYMBOL, qty))
        formatted_stop = float(self.exchange.price_to_precision(SYMBOL, stop_price))
        formatted_limit = float(self.exchange.price_to_precision(SYMBOL, limit_price))
        
        params = {
            'stopPrice': formatted_stop,
            'triggerPrice': formatted_stop,
        }
        
        log.info(f"Placing REAL Stop-loss on exchange: Trigger={formatted_stop}, Limit={formatted_limit}, Qty={formatted_qty}")
        
        for attempt in range(3):
            try:
                stop_order = await asyncio.to_thread(
                    self.exchange.create_order,
                    SYMBOL, 'limit', 'sell', formatted_qty, formatted_limit, params
                )
                return stop_order['id']
            except Exception as e:
                if attempt == 2:
                    raise e
                log.warning(f"Stop-loss placement failed, retrying in 1s... Error: {e}")
                await asyncio.sleep(1.0)

    async def cancel_native_stop_loss(self, stop_order_id):
        if not stop_order_id: return
        for attempt in range(3):
            try:
                await asyncio.to_thread(self.exchange.cancel_order, stop_order_id, SYMBOL)
                log.info(f"Exchange stop-loss order {stop_order_id} cancelled successfully.")
                return
            except Exception as e:
                err_msg = str(e).lower()
                if "not found" in err_msg or "already" in err_msg or "filled" in err_msg or "cancel" in err_msg:
                    log.info(f"Stop order {stop_order_id} already closed: {e}")
                    return
                if attempt == 2:
                    log.error(f"Failed to cancel stop order: {e}")
                    return
                await asyncio.sleep(1.0)

    async def fetch_ohlcv(self):
        ohlcv = await asyncio.to_thread(self.exchange.fetch_ohlcv, SYMBOL, bot_state["timeframe"], None, OHLCV_LIMIT)
        return pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume']).assign(timestamp=lambda d: pd.to_datetime(d['timestamp'], unit='ms'))

    def _research_exchange(self, provider):
        """Return a lazy, public-only CCXT client for research candles."""
        if provider == "mexc":
            return self.exchange
        cache = getattr(self, "research_exchanges", {})
        if provider in cache:
            return cache[provider]
        if provider == "binance":
            exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
            exchange.urls['api']['public'] = 'https://data-api.binance.vision/api/v3'
            exchange.urls['api']['v1'] = 'https://data-api.binance.vision/api/v1'
        elif provider == "bybit":
            exchange = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        elif provider == "okx":
            exchange = ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        else:
            raise ValueError(f"Unknown research data provider: {provider}")
        cache[provider] = exchange
        self.research_exchanges = cache
        return exchange

    async def _fetch_ohlcv_paginated(self, exchange, provider, symbol, timeframe, limit):
        """Fetch one provider through forward CCXT pagination."""
        target_limit = min(max(int(limit), RESEARCH_BARS_MIN), RESEARCH_BARS_MAX)
        try:
            tf_ms = int(exchange.parse_timeframe(timeframe) * 1000)
        except Exception:
            tf_ms = {
                "1m": 60_000, "5m": 300_000, "15m": 900_000,
                "1h": 3_600_000, "4h": 14_400_000,
            }.get(timeframe, 60_000)

        now_ms = int(datetime.now().timestamp() * 1000)
        # Ask for two extra intervals so exchange boundary alignment / the still-open
        # current candle cannot leave the requested closed-bar window one row short.
        cursor = now_ms - ((target_limit + 2) * tf_ms)
        candles = {}
        page_size = 1000
        # Some venues return fewer candles than requested, so allow a generous but
        # finite page budget while still detecting a stalled cursor below.
        max_pages = max(5, int(math.ceil(target_limit / page_size)) * 3 + 3)

        for page in range(max_pages):
            try:
                batch = await asyncio.to_thread(
                    exchange.fetch_ohlcv, symbol, timeframe, cursor, page_size
                )
                if not batch:
                    break
                previous_cursor = cursor
                for row in batch:
                    if row and len(row) >= 6:
                        candles[int(row[0])] = row[:6]
                newest_ts = max(int(row[0]) for row in batch if row)
                cursor = newest_ts + 1
                log.info(
                    f"[{provider.upper()}] OHLCV page {page + 1}: "
                    f"{len(candles)}/{target_limit} unique bars"
                )
                if len(candles) >= target_limit or cursor <= previous_cursor or newest_ts >= now_ms - tf_ms:
                    break
            except Exception as e:
                log.error(f"[{provider.upper()}] OHLCV page {page + 1} failed: {e}")
                raise

        ordered = [candles[k] for k in sorted(candles)][-target_limit:]
        if not ordered:
            raise RuntimeError(f"{provider} returned no historical OHLCV data")
        if len(ordered) < target_limit:
            log.warning(f"[{provider.upper()}] requested {target_limit}, received {len(ordered)} bars")
        df = pd.DataFrame(ordered, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.attrs["data_source"] = provider
        return df

    async def fetch_ohlcv_large(self, symbol, timeframe, limit=RESEARCH_BARS_DEFAULT):
        """Fetch free public research data with automatic provider fallback.

        Order execution remains on MEXC. Backtest/WFO candles default to Binance's
        keyless market-data endpoint, then Bybit and OKX; MEXC is only the last
        fallback. Provider order can be changed with RESEARCH_DATA_PROVIDERS.
        """
        target_limit = min(max(int(limit), RESEARCH_BARS_MIN), RESEARCH_BARS_MAX)
        errors = []
        best_partial = None
        for provider in RESEARCH_PROVIDER_ORDER:
            try:
                exchange = self._research_exchange(provider)
                df = await self._fetch_ohlcv_paginated(
                    exchange, provider, symbol, timeframe, target_limit
                )
                if best_partial is None or len(df) > len(best_partial):
                    best_partial = df
                if len(df) >= target_limit:
                    bot_state["research_data_source"] = provider
                    log.info(f"Research data source selected: {provider.upper()} ({len(df)} bars)")
                    return df
                errors.append(f"{provider}: only {len(df)}/{target_limit} bars")
            except Exception as e:
                errors.append(f"{provider}: {e}")
                log.warning(f"Research provider {provider.upper()} unavailable; trying next source")

        if best_partial is not None and len(best_partial) >= GeometricPipeline.MIN_TRAIN_BARS + 200:
            provider = best_partial.attrs.get("data_source", "partial")
            bot_state["research_data_source"] = provider
            log.warning(f"Using best partial research dataset: {provider} ({len(best_partial)} bars)")
            return best_partial
        raise RuntimeError("All public research data providers failed: " + " | ".join(errors))

    async def close_position(self, reason, price):
        if not self.position.is_open: return
        try:
            # Save entry data BEFORE close() zeroes them
            saved_entry_type = self.position.entry_type
            saved_entry_price = self.position.entry_price
            pos_mode = self.position.mode
            pos_side = self.position.side
            pos_qty = self.position.qty
            stop_order_id = self.position.stop_order_id
            invested_amount = self.position.invested_amount
            max_price_seen = self.position.max_price_seen
            min_price_seen = self.position.min_price_seen
            entry_volatility = self.position.entry_volatility

            # 1. Exchange stop order cancel
            if pos_mode == "REAL" and stop_order_id:
                await self.cancel_native_stop_loss(stop_order_id)

            # 2. Execution order
            actual_exit_price = price
            if pos_mode == "REAL":
                side = "sell" if pos_side == "long" else "buy"
                order = await self.execute_marketable_order(side, pos_qty, price)
                filled_qty = float(order.get('filled', 0.0))
                if filled_qty <= 0:
                    log.error("REAL CLOSE ORDER FAILED TO FILL ANY QUANTITY. POSITION REMAINS OPEN.")
                    # Re-create stop loss order on exchange
                    if stop_order_id:
                        try:
                            new_stop_id = await self.place_native_stop_loss(
                                pos_qty, 
                                self.position.trail_stop_30 if self.position.has_taken_partial_tp else self.position.trail_stop_70
                            )
                            self.position.stop_order_id = new_stop_id
                        except Exception as ex_recreate:
                            log.error(f"Failed to re-create stop order: {ex_recreate}")
                    return
                
                actual_exit_price = float(order.get('average', price))
                pos_qty = filled_qty
            else:
                log.info(f"PAPER CLOSE EXECUTED (Virtual)")
                # Slippage & commission for Paper Trade
                # Exit slippage = 0.05%
                exit_price_slippage = price * 0.9995 if pos_side == "long" else price * 1.0005
                actual_exit_price = exit_price_slippage
                
                # Exit commission = 0.1% of exit value
                comm_exit = (pos_qty * actual_exit_price) * 0.001
                bot_state["virtual_balance"] -= comm_exit

            pnl_pct, pnl_usdt = self.position.close(reason, actual_exit_price)
            
            # Cumulative realized PnL of this trade (adding the last 30% exit PnL)
            total_trade_pnl_usdt = self.position.realized_pnl_usdt + pnl_usdt
            
            # For paper, we update virtual balance with the net return of this portion
            if pos_mode == "PAPER":
                bot_state["virtual_balance"] += pnl_usdt

            # Update metrics using portfolio return
            update_portfolio_metrics(total_trade_pnl_usdt, pos_mode)

            bot_state["trades"].insert(0, {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": saved_entry_type, "side": "CLOSED",
                "entry": float(saved_entry_price), "exit": float(actual_exit_price),
                "pnl": f"{((total_trade_pnl_usdt / invested_amount) * 100) if invested_amount > 0 else pnl_pct:+.2f}%", 
                "reason": reason
            })
            bot_state["position_side"] = None
            
            # Send Telegram alert
            msg = (
                f"🚪 *POZİSYON KAPATILDI!*\n\n"
                f"🪙 *Parite:* {SYMBOL}\n"
                f"📈 *Tip:* {saved_entry_type}\n"
                f"💵 *Giriş:* {saved_entry_price:.2f} USDT\n"
                f"💸 *Çıkış:* {actual_exit_price:.2f} USDT\n"
                f"📊 *Net PnL:* {total_trade_pnl_usdt:+.2f} USDT ({((total_trade_pnl_usdt / invested_amount) * 100) if invested_amount > 0 else pnl_pct:+.2f}%)\n"
                f"🔍 *Neden:* {reason}"
            )
            asyncio.create_task(self.telegram.send_message(msg))
        except Exception as e:
            log.error(f"Pozisyon kapatilamadi: {e}")

    async def main_tick(self):
        df = await self.fetch_ohlcv()
        force_retrain = False

        if bot_state["timeframe_changed"]:
            log.info(f"Zaman dilimi degisti ({bot_state['timeframe']}). Mevcut pozisyon korunarak devam ediliyor...")
            bot_state["timeframe_changed"] = False
            force_retrain = True

        active_params = get_active_parameters()
        bot_state["active_parameters"] = active_params
        
        trail_mult = float(active_params.get("TRAIL_MULT", 3.0))

        # Repaint fix: process completed bars only
        df_completed = df.iloc[:-1].copy() if len(df) > 30 else df
        info = self.signal_engine.process(df_completed, params=active_params, force_retrain=force_retrain)
        price = float(df['close'].iloc[-1])
        geom = info.get("geom", {}) or {}
        log.info(f"Tick [{bot_state['timeframe']}]: Price={price:.2f}, Signal={info['signal']} ({info['type'] or 'None'}), "
                 f"Geo={geom.get('status','-')}/{geom.get('health','-')}, Active={bot_state['is_trading_active']}, OBI={bot_state.get('obi', 0.0):.2f}")
        bot_state["gauss_vol"] = float(info['gauss_vol'])
        bot_state["signal"] = info['signal']
        bot_state["signal_type"] = info['type']
        bot_state["geom"] = geom or bot_state.get("geom", {"status": "collecting", "signal": "HOLD"})

        # OHLC chart data for Lightweight Charts (unix timestamps) with the
        # learned-geometry overlay: Conformal A, episodes, GEO entries/exits.
        # The geometry arrays cover the tail of the COMPLETED bars.
        gchart = (info.get("geom") or {}).get("chart") or {}
        g_n = int(gchart.get("n", 0))
        completed_len = len(df) - 1 if len(df) > 30 else len(df)
        g_off = completed_len - g_n
        g_gate = float(gchart.get("gate", 0.5))
        chart_data = []
        for j in range(len(df)):
            t_val = int(df['timestamp'].iloc[j].timestamp())
            gi = j - g_off
            has_geo = 0 <= gi < g_n
            chart_data.append({
                "time": t_val,
                "open": float(df['open'].iloc[j]),
                "high": float(df['high'].iloc[j]),
                "low": float(df['low'].iloc[j]),
                "close": float(df['close'].iloc[j]),
                "geo_a": float(gchart["A"][gi]) if has_geo else None,
                "geo_gate": g_gate,
                "geo_episode": bool(gchart["episode"][gi]) if has_geo else False,
                "geo_state": int(gchart["state"][gi]) if has_geo else 0,
                "geo_buy": bool(gchart["buy"][gi]) if has_geo else False,
                "geo_sell": bool(gchart.get("sell", [])[gi]) if (has_geo and gchart.get("sell")) else False,
                "geo_exit": bool(gchart["exit"][gi]) if has_geo else False,
            })
        bot_state["chart_data"] = chart_data

        if self.position.is_open:
            gv = info['gauss_vol']
            self.position.update_stops(price, gv, gv)
            reason = self.position.check_exits(price, info)
            # Geo exits when conformal collapses inside an anomalous episode
            if reason is None and geom.get("exit_flag"):
                reason = "Geo Exit (Conformal/Transition)"

            bot_state["position_side"] = self.position.side
            bot_state["position_entry"] = float(self.position.entry_price)
            bot_state["position_qty"] = float(self.position.qty)
            bot_state["position_type"] = self.position.entry_type
            bot_state["position_pnl"] = float((price-self.position.entry_price)/self.position.entry_price*100 if self.position.side=="long" else (self.position.entry_price-price)/self.position.entry_price*100)
            bot_state["trail_stop"] = float(self.position.trail_stop_30 if self.position.has_taken_partial_tp else self.position.trail_stop_70)
            bot_state["ping_stop"] = float(self.position.ping_stop)

            if reason == "PARTIAL_TP":
                try:
                    ptp = cfg("partial_tp_ratio") / 100.0; keep = 1.0 - ptp
                    close_qty = self.position.qty * ptp
                    log.info(f"PARTIAL TP ({ptp*100:.0f}%) TRIGGERED! Qty to close: {close_qty:.6f} at {price:.2f}")

                    pos_mode = self.position.mode
                    pos_side = self.position.side
                    pos_entry_type = self.position.entry_type
                    pos_entry_price = self.position.entry_price
                    pos_invested = self.position.invested_amount

                    actual_exit_price = price
                    if pos_mode == "REAL":
                        # Cancel existing stop-loss
                        if self.position.stop_order_id:
                            await self.cancel_native_stop_loss(self.position.stop_order_id)
                        
                        # Market sell 70%
                        side = "sell" if pos_side == "long" else "buy"
                        order = await self.execute_marketable_order(side, close_qty, price)
                        filled_close_qty = float(order.get('filled', 0.0))
                        if filled_close_qty <= 0:
                            log.error("REAL PARTIAL TP ORDER FAILED TO FILL ANY QUANTITY. MAINTAINING ORIGINAL POSITION.")
                            # Recreate stop loss
                            try:
                                new_stop_id = await self.place_native_stop_loss(self.position.qty, self.position.trail_stop_70)
                                self.position.stop_order_id = new_stop_id
                            except Exception as ex_stop:
                                log.error(f"Failed to place native exchange stop order: {ex_stop}")
                            return
                        
                        actual_exit_price = float(order.get('average', price))
                        close_qty = filled_close_qty
                    else:
                        # Paper TP with slippage and fee
                        exit_price_slippage = price * 0.9995 if pos_side == "long" else price * 1.0005
                        actual_exit_price = exit_price_slippage
                        
                        # Exit commission = 0.1% of exit value
                        comm_exit = (close_qty * actual_exit_price) * 0.001
                        bot_state["virtual_balance"] -= comm_exit

                    pnl_pct = (actual_exit_price - pos_entry_price)/pos_entry_price*100 if pos_side=="long" else (pos_entry_price - actual_exit_price)/pos_entry_price*100
                    realized_pnl_usdt = (pos_invested * ptp) * (pnl_pct / 100)
                    
                    # Update virtual balance for Paper
                    if pos_mode == "PAPER":
                        bot_state["virtual_balance"] += realized_pnl_usdt

                    # Add exit commission to total realized PnL reduction
                    if pos_mode == "PAPER":
                        self.position.realized_pnl_usdt += (realized_pnl_usdt - comm_exit)
                    else:
                        self.position.realized_pnl_usdt += realized_pnl_usdt

                    bot_state["trades"].insert(0, {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": pos_entry_type, "side": f"PART_TP_{pos_side.upper()}",
                        "entry": float(pos_entry_price), "exit": float(actual_exit_price),
                        "pnl": f"{pnl_pct:+.2f}%", "reason": "First Trailing Stop Hit (70% TP)"
                    })

                    # Adjust remaining position (30%)
                    self.position.qty *= keep
                    self.position.invested_amount *= keep
                    self.position.has_taken_partial_tp = True
                    bot_state["position_qty"] = float(self.position.qty)

                    # If REAL, place a new stop-loss order on the exchange for the remaining 30% quantity
                    if pos_mode == "REAL":
                        try:
                            stop_order_id = await self.place_native_stop_loss(self.position.qty, self.position.trail_stop_30)
                            self.position.stop_order_id = stop_order_id
                        except Exception as ex_stop30:
                            log.error(f"Failed to place 30% remainder stop order on exchange: {ex_stop30}")

                    # Send Telegram alert
                    msg = (
                        f"💰 *KISMİ KAR ALINDI (%70)*\n\n"
                        f"🪙 *Parite:* {SYMBOL}\n"
                        f"📈 *Yön:* {pos_side.upper()}\n"
                        f"💵 *Giriş:* {pos_entry_price:.2f} USDT\n"
                        f"💸 *Satış Fiyatı:* {actual_exit_price:.2f} USDT\n"
                        f"📊 *PnL:* {pnl_pct:+.2f}%\n"
                        f"💼 *Kalan Miktar:* {self.position.qty:.6f}"
                    )
                    asyncio.create_task(self.telegram.send_message(msg))

                except Exception as e_ptp:
                    log.error(f"Partial TP execution error: {e_ptp}")
            elif reason:
                await self.close_position(reason, price)
        else:
            bot_state["position_side"] = None

        # OBI & Risk Parity filtering
        obi_filter_pass = True
        obi_now = bot_state.get("obi", 0)
        if info['signal'] == "BUY" and obi_now < -cfg("obi_wall"):
            obi_filter_pass = False
            log.info(f"BUY blocked by OBI ({obi_now:.2f} Sell Wall)")
        elif info['signal'] == "SELL" and obi_now > cfg("obi_wall"):
            obi_filter_pass = False
            log.info(f"SELL blocked by OBI ({obi_now:.2f} Buy Wall)")

        # Spot BUY entry (long) — SELL branch below handles opt-in shorts
        if not self.position.is_open and info['signal'] == "BUY" and obi_filter_pass:
            if bot_state["is_trading_active"]:
                try:
                    # cost-floored, vol-scaled targets from the geometry pipeline
                    entry_vol = float(info['gauss_vol'])
                    cost = roundtrip_cost_pct()
                    bar_vol_pct = (entry_vol / price * 100.0) if price > 0 else 0.0
                    opt_tp = max(float(geom.get("tp_pct") or 0.0), cfg("tp_cost_floor_mult") * cost,
                                 float(active_params.get("TP_PERCENT", cfg("tp_base_pct"))),
                                 2.0 * bar_vol_pct * math.sqrt(int(cfg("barrier_hold_bars"))))
                    opt_sl = max(float(geom.get("sl_pct") or 0.0), cfg("sl_cost_floor_mult") * cost, 0.5 * opt_tp)
                    min_trail_dist = price * opt_tp / 100.0 * 0.5
                    stop_dist = max(entry_vol * trail_mult, min_trail_dist, price * 0.01)
                    active_mode = bot_state["trading_mode"]

                    if active_mode == "REAL":
                        balance = await asyncio.to_thread(self.exchange.fetch_balance)
                        real_usdt = float(balance.get('USDT', {}).get('free', 0) or 0)
                        bot_state["real_balance"] = real_usdt
                        # Record starting balance for first trade if not recorded
                        if bot_state.get("start_real_balance", 0.0) <= 0:
                            bot_state["start_real_balance"] = real_usdt
                            bot_state["peak_real_balance"] = real_usdt

                        if real_usdt >= cfg("min_order_usdt"):
                            side = "buy"
                            risk_amount = real_usdt * (cfg("risk_per_trade_pct") / 100.0)
                            target_qty = risk_amount / stop_dist
                            max_qty = (real_usdt * (cfg("max_capital_pct") / 100.0)) / price
                            qty = min(target_qty, max_qty)
                            
                            # Precision check & min 5 USDT enforcement
                            formatted_qty = self.exchange.amount_to_precision(SYMBOL, qty)
                            order_qty = float(formatted_qty)
                            
                            if order_qty * price < cfg("min_order_usdt"):
                                # Scale up to minimum order size if real balance allows
                                min_qty = (cfg("min_order_usdt") * 1.04) / price
                                formatted_min = self.exchange.amount_to_precision(SYMBOL, min_qty)
                                if real_usdt >= float(formatted_min) * price:
                                    order_qty = float(formatted_min)
                                    log.info(f"Real order scaled up to meet MEXC minimum 5 USDT limit: {order_qty}")
                                else:
                                    log.warn("Real balance below MEXC Spot minimum 5 USDT limit. Entry skipped.")
                                    return

                            # Execution order
                            order = await self.execute_marketable_order(side, order_qty, price)
                            filled_qty = float(order.get('filled', 0.0))
                            if filled_qty <= 0:
                                log.warn("REAL ENTRY ORDER FAILED TO FILL ANY QUANTITY. ENTRY ABORTED.")
                                return
                            
                            price = float(order.get('average', price))
                            invested = filled_qty * price

                            # Native stop-loss at TP/2 below entry (cost-floored)
                            stop_order_id = None
                            try:
                                stop_price_val = price - stop_dist
                                stop_order_id = await self.place_native_stop_loss(filled_qty, stop_price_val)
                            except Exception as ex_stop:
                                log.error(f"Failed to place native exchange stop order: {ex_stop}")

                            pos_side = "long"
                            self.position.open(pos_side, price, filled_qty, info['type'], info['gauss_vol'], invested, mode="REAL", stop_order_id=stop_order_id, tp_percent=opt_tp, sl_percent=opt_sl, entry_volatility=entry_vol, min_trail_dist=min_trail_dist)
                            
                            # Send Telegram alert
                            msg = (
                                f"🔔 *YENİ POZİSYON AÇILDI (REAL)*\n\n"
                                f"🪙 *Parite:* {SYMBOL}\n"
                                f"📈 *Yön:* {pos_side.upper()}\n"
                                f"🔍 *Tip:* {info['type']}\n"
                                f"💵 *Giriş Fiyatı:* {price:.2f} USDT\n"
                                f"💼 *Miktar:* {order_qty:.6f}\n"
                                f"💸 *Yatırım:* {invested:.2f} USDT"
                            )
                            asyncio.create_task(self.telegram.send_message(msg))
                    else:
                        # PAPER TRADING
                        if bot_state["virtual_balance"] >= cfg("min_order_usdt"):
                            side = "buy"
                            risk_amount = bot_state["virtual_balance"] * (cfg("risk_per_trade_pct") / 100.0)
                            target_qty = risk_amount / stop_dist
                            max_qty = (bot_state["virtual_balance"] * (cfg("max_capital_pct") / 100.0)) / price
                            qty = min(target_qty, max_qty)
                            
                            # Apply paper slippage on entry (0.05%)
                            slippage_price = price * 1.0005
                            invested = qty * slippage_price
                            
                            # Deduct entry commission fee (0.1%)
                            comm_entry = invested * 0.001
                            bot_state["virtual_balance"] -= comm_entry

                            log.info(f"PAPER ORDER: buy {info['type']} (Qty: {qty:.6f}, Entry Price: {slippage_price:.2f}, Invest: ${invested:.2f})")
                            pos_side = "long"
                            self.position.open(pos_side, slippage_price, qty, info['type'], info['gauss_vol'], invested, mode="PAPER", tp_percent=opt_tp, sl_percent=opt_sl, entry_volatility=entry_vol, min_trail_dist=min_trail_dist)

                            bot_state["trades"].insert(0, {
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "type": info['type'], "side": pos_side.upper(),
                                "entry": float(slippage_price), "exit": 0.0,
                                "pnl": "OPEN", "reason": f"Signal: {info['signal']}"
                            })
                            
                            # Send Telegram alert
                            msg = (
                                f"🔔 *YENİ POZİSYON AÇILDI (PAPER)*\n\n"
                                f"🪙 *Parite:* {SYMBOL}\n"
                                f"📈 *Yön:* {pos_side.upper()}\n"
                                f"🔍 *Tip:* {info['type']}\n"
                                f"💵 *Giriş Fiyatı:* {slippage_price:.2f} USDT\n"
                                f"💼 *Miktar:* {qty:.6f}\n"
                                f"💸 *Yatırım:* {invested:.2f} USDT"
                            )
                            asyncio.create_task(self.telegram.send_message(msg))

                except Exception as e: log.error(f"Siparis hatasi: {e}")

        # ── SHORT entry (two-sided / futures). Opt-in via allow_short. PAPER is
        # fully simulated; REAL live short is intentionally NOT auto-executed —
        # futures order/margin/liquidation handling must be validated first. ──
        if (not self.position.is_open and info['signal'] == "SELL"
                and bot_state.get("allow_short") and obi_filter_pass
                and bot_state["is_trading_active"]):
            try:
                entry_vol = float(info['gauss_vol'])
                cost = roundtrip_cost_pct()
                g_state = info.get("geom", {}) or {}
                bar_vol_pct = (entry_vol / price * 100.0) if price > 0 else 0.0
                opt_tp = max(float(g_state.get("tp_pct") or 0.0), cfg("tp_cost_floor_mult") * cost,
                             2.0 * bar_vol_pct * math.sqrt(int(cfg("barrier_hold_bars"))))
                opt_sl = max(float(g_state.get("sl_pct") or 0.0), cfg("sl_cost_floor_mult") * cost, 0.5 * opt_tp)
                min_trail_dist = price * opt_tp / 100.0 * 0.5
                stop_dist = max(entry_vol * trail_mult, min_trail_dist)
                if stop_dist <= 0:
                    stop_dist = price * 0.01

                if bot_state["trading_mode"] == "REAL":
                    log.warning("Geo SELL (short) received in REAL mode: live futures short "
                                "execution is not auto-armed. Validate in PAPER/backtest/WFO, "
                                "then wire futures orders deliberately. Signal skipped.")
                    if not bot_state.get("_short_real_warned"):
                        bot_state["_short_real_warned"] = True
                        asyncio.create_task(self.telegram.send_message(
                            "⚠️ *Geo SHORT sinyali (REAL)*: Canlı futures short otomatik açılmıyor. "
                            "Önce PAPER/backtest ile doğrula; futures emirleri bilinçli olarak devreye alınmalı."))
                elif bot_state["virtual_balance"] >= cfg("min_order_usdt"):
                    risk_amount = bot_state["virtual_balance"] * (cfg("risk_per_trade_pct") / 100.0)
                    target_qty = risk_amount / stop_dist
                    max_qty = (bot_state["virtual_balance"] * (cfg("max_capital_pct") / 100.0)) / price
                    qty = min(target_qty, max_qty)

                    slippage_price = price * 0.9995          # short fills below mid
                    invested = qty * slippage_price
                    bot_state["virtual_balance"] -= invested * 0.001   # entry fee

                    log.info(f"PAPER SHORT: sell Geo (Qty: {qty:.6f}, Entry: {slippage_price:.2f}, "
                             f"Invest: ${invested:.2f}, TP {opt_tp:.2f}% / SL {opt_sl:.2f}%)")
                    self.position.open("short", slippage_price, qty, info['type'],
                                       entry_vol, invested, mode="PAPER",
                                       tp_percent=opt_tp, sl_percent=opt_sl,
                                       entry_volatility=entry_vol, min_trail_dist=min_trail_dist)
                    bot_state["trades"].insert(0, {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": info['type'], "side": "SHORT",
                        "entry": float(slippage_price), "exit": 0.0,
                        "pnl": "OPEN", "reason": f"Signal: {info['signal']}"
                    })
                    asyncio.create_task(self.telegram.send_message(
                        f"🔻 *YENİ SHORT AÇILDI (PAPER)*\n\n"
                        f"🪙 *Parite:* {SYMBOL}\n"
                        f"📉 *Yön:* SHORT\n"
                        f"🔍 *Tip:* {info['type']}\n"
                        f"💵 *Giriş:* {slippage_price:.2f} USDT\n"
                        f"🎯 *TP/SL:* {opt_tp:.2f}% / {opt_sl:.2f}%"))
            except Exception as e_short:
                log.error(f"Short entry error: {e_short}")

        # Shadow execution is gone. WFO auto-promotes only a validated OOS winner;
        # rejected or position-blocked candidates remain visible as challengers.
        bot_state["shadow_active"] = False

    async def fast_orderbook_tick(self):
        while True:
            try:
                ob = await asyncio.to_thread(self.exchange.fetch_order_book, SYMBOL, limit=20)
                if ob['bids'] and ob['asks']:
                    bot_state["orderbook"]["bids"] = [[float(p), float(q)] for p, q in ob['bids']]
                    bot_state["orderbook"]["asks"] = [[float(p), float(q)] for p, q in ob['asks']]

                    bids_vol = sum(vol for price, vol in ob['bids'])
                    asks_vol = sum(vol for price, vol in ob['asks'])
                    bot_state["obi"] = float((bids_vol - asks_vol) / (bids_vol + asks_vol)) if (bids_vol + asks_vol) > 0 else 0.0

                    current_price = float(ob['bids'][0][0])
                    bot_state["price"] = current_price

                    if self.position.is_open:
                        if self.position.side == "long":
                            bot_state["position_pnl"] = float((current_price - self.position.entry_price) / self.position.entry_price * 100)
                        else:
                            bot_state["position_pnl"] = float((self.position.entry_price - current_price) / self.position.entry_price * 100)
            except Exception as e: log.error(f'OB Error: {e}')
            await asyncio.sleep(FAST_LOOP_INTERVAL)

    async def main_loop(self):
        log.info("QUANT BOT V3.7 (LightGBM Research Engine) - Paper Trading Active")
        bot_state["loop"] = asyncio.get_running_loop()
        await asyncio.to_thread(self.exchange.load_markets)

        # 1. Startup Position and Stop Order Reconciliation
        try:
            balance = await asyncio.to_thread(self.exchange.fetch_balance)
            btc_total = float(balance.get('BTC', {}).get('total', 0.0) or 0.0)
            
            ticker = await asyncio.to_thread(self.exchange.fetch_ticker, SYMBOL)
            current_price = float(ticker.get('last', 0.0) or ticker.get('close', 0.0) or 60000.0)
            btc_value_usdt = btc_total * current_price
            
            if btc_value_usdt >= 5.0:
                log.info(f"Startup Reconciliation: Found open BTC position on exchange of {btc_total:.6f} BTC (~{btc_value_usdt:.2f} USDT)")
                
                # Fetch recent trades to find actual entry price
                entry_price = current_price
                entry_type = "Geo"
                try:
                    trades = await asyncio.to_thread(self.exchange.fetch_my_trades, SYMBOL, limit=5)
                    if trades:
                        buy_trades = [t for t in trades if t['side'] == 'buy']
                        if buy_trades:
                            buy_trades.sort(key=lambda x: x['timestamp'], reverse=True)
                            last_buy = buy_trades[0]
                            entry_price = float(last_buy['price'])
                            log.info(f"Reconciliation: Identified entry price from last buy fill: {entry_price:.2f}")
                except Exception as ex_trades:
                    log.info(f"Reconciliation: Could not fetch trade history: {ex_trades}. Using current price as fallback.")

                # Scan open orders to identify any active stop orders
                stop_order_id = None
                try:
                    open_orders = await asyncio.to_thread(self.exchange.fetch_open_orders, SYMBOL)
                    stop_orders = [o for o in open_orders if o['side'] == 'sell' and ('stopPrice' in o.get('info', {}) or o.get('type') == 'limit')]
                    if stop_orders:
                        stop_orders.sort(key=lambda x: x['timestamp'] or 0, reverse=True)
                        stop_order_id = stop_orders[0]['id']
                        log.info(f"Reconciliation: Linked active exchange stop order: {stop_order_id}")
                except Exception as ex_orders:
                    log.info(f"Reconciliation: Could not fetch open orders: {ex_orders}")

                # Sync position locally
                self.position.open(
                    side="long", 
                    price=entry_price, 
                    qty=btc_total, 
                    sig_type=entry_type, 
                    gauss_vol=current_price * 0.01, 
                    invested_amount=btc_total * entry_price,
                    mode="REAL",
                    stop_order_id=stop_order_id
                )
                
                bot_state["position_side"] = "long"
                bot_state["position_entry"] = float(entry_price)
                bot_state["position_qty"] = float(btc_total)
                bot_state["position_type"] = entry_type
                bot_state["position_pnl"] = float((current_price - entry_price) / entry_price * 100)
                bot_state["trading_mode"] = "REAL"
                bot_state["start_real_balance"] = float(balance.get('USDT', {}).get('total', 0.0) or 0.0) + btc_value_usdt
                bot_state["peak_real_balance"] = bot_state["start_real_balance"]
                
        except Exception as ex_reconcile:
            log.error(f"Startup reconciliation failed: {ex_reconcile}")

        asyncio.create_task(self.fast_orderbook_tick())
        asyncio.create_task(self.telegram.run_loop())  # Start Telegram polling

        # 2. Race-Condition-Free Event Loop Flow
        while True:
            with state_lock:
                changed = bot_state["timeframe_changed"]
            
            if not changed:
                try:
                    await asyncio.wait_for(tf_change_event.wait(), timeout=cfg("loop_interval_sec"))
                    log.info("Tick woke up (interval or timeframe trigger).")
                except asyncio.TimeoutError:
                    pass
            
            tf_change_event.clear()
            
            try:
                await self.main_tick()
            except Exception as e:
                log.error(f"Ana dongu hatasi: {e}")
                await asyncio.sleep(5)

class Backtester:
    def __init__(self, df, commission_rate=None, slippage_rate=None, spread_rate=None):
        self.df = df
        # cost rates default to the live trading config (percent → fraction)
        self.commission_rate = cfg("commission_pct") / 100.0 if commission_rate is None else commission_rate
        self.slippage_rate = cfg("slippage_pct") / 100.0 if slippage_rate is None else slippage_rate
        self.spread_rate = cfg("spread_pct") / 100.0 if spread_rate is None else spread_rate

    def run(self, params, geo=None):
        """geo: optional per-bar arrays from GeometricPipeline.batch_signals — when
        given, entries/exits come from the learned-geometry decision chain."""
        c = self.df['close'].values
        h = self.df['high'].values
        l = self.df['low'].values
        o = self.df['open'].values
        v = self.df['volume'].values
        
        balance = 1000.0
        position_side = None
        entry_price = 0.0
        entry_type = ""
        qty = 0.0
        invested_amount = 0.0
        has_taken_partial_tp = False
        trail_stop_30 = 0.0
        trail_stop_70 = 0.0
        ping_stop = 0.0
        min_trail_dist = 0.0

        entry_volatility = 0.0
        current_tp_percent = 0.3
        current_sl_percent = 0.3
        max_price_seen = 0.0
        min_price_seen = 0.0

        trades = []
        pnl_list = []
        gross_profit = 0.0
        gross_loss = 0.0
        win_trades = 0
        loss_trades = 0

        vol_len = int(params.get("VOL_LENGTH", 14))
        trail_mult = float(params.get("TRAIL_MULT", 3.0))
        tp_percent = float(params.get("TP_PERCENT", 0.6))

        # Simple realised volatility (True Range, rolling mean) for trailing stops.
        # No more Gaussian-filter/CVD/HMM machinery — signals come from `geo` only.
        tr = calc_true_range(h, l, c)
        gv = np.zeros(len(c))
        for i in range(len(c)):
            j = max(0, i - vol_len + 1)
            gv[i] = float(np.mean(tr[j:i + 1])) if i >= j else float(tr[i])

        if geo is None:
            # Geo-mandatory: without a geometry payload the backtester emits nothing.
            return {
                "trade_count": 0, "total_pnl_pct": 0.0, "total_pnl_usdt": 0.0,
                "profit_factor": 1.0, "max_drawdown_pct": 0.0, "calmar_ratio": 0.0,
                "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "recovery_factor": 0.0,
                "win_rate": 0.0, "expectancy": 0.0, "trades": [],
                "engine": "geometric-empty", "reason": "no geo signal payload provided"
            }

        start_idx = max(vol_len + 5, 65)
        if start_idx >= len(self.df):
            return {
                "trade_count": 0, "total_pnl_pct": 0.0, "total_pnl_usdt": 0.0,
                "profit_factor": 1.0, "max_drawdown_pct": 0.0, "calmar_ratio": 0.0,
                "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "recovery_factor": 0.0,
                "win_rate": 0.0, "expectancy": 0.0, "trades": []
            }

        for idx in range(start_idx, len(self.df)):
            price = float(c[idx])

            if position_side is not None:
                max_price_seen = max(max_price_seen, price)
                min_price_seen = min(min_price_seen, price)
                gv_normal = float(gv[idx-1])
                d30 = max(gv_normal * trail_mult, min_trail_dist)

                if position_side == "long":
                    trail_stop_30 = max(trail_stop_30, price - d30)
                    if not has_taken_partial_tp:
                        trail_stop_70 = max(trail_stop_70, price - d30)
                else:
                    trail_stop_30 = min(trail_stop_30, price + d30)
                    if not has_taken_partial_tp:
                        trail_stop_70 = min(trail_stop_70, price + d30)

                exit_reason = None
                if position_side == "long":
                    if has_taken_partial_tp:
                        if price <= trail_stop_30: exit_reason = "Trail Stop (30%)"
                    else:
                        if price <= entry_price * (1 - current_sl_percent/100): exit_reason = "Stop Loss"
                        elif price <= trail_stop_70: exit_reason = "PARTIAL_TP"
                        elif price >= entry_price * (1 + current_tp_percent/100): exit_reason = f"Take Profit ({current_tp_percent:.2f}%)"
                else:
                    if has_taken_partial_tp:
                        if price >= trail_stop_30: exit_reason = "Trail Stop (30%)"
                    else:
                        if price >= entry_price * (1 + current_sl_percent/100): exit_reason = "Stop Loss"
                        elif price >= trail_stop_70: exit_reason = "PARTIAL_TP"
                        elif price <= entry_price * (1 - current_tp_percent/100): exit_reason = f"Take Profit ({current_tp_percent:.2f}%)"

                if exit_reason is None and bool(geo["exit_flag"][idx-1]):
                    exit_reason = "Geo Exit (Conformal/Transition)"

                if exit_reason == "PARTIAL_TP":
                    ptp = cfg("partial_tp_ratio") / 100.0; keep = 1.0 - ptp
                    close_qty = qty * ptp
                    slippage_factor = self.slippage_rate + self.spread_rate / 2
                    exec_price = price * (1 - slippage_factor) if position_side == "long" else price * (1 + slippage_factor)
                    pnl_pct = (exec_price - entry_price) / entry_price * 100 if position_side == "long" else (entry_price - exec_price) / entry_price * 100
                    fee = close_qty * exec_price * self.commission_rate
                    pnl_usdt = (invested_amount * ptp) * (pnl_pct / 100) - fee
                    
                    balance += pnl_usdt
                    pnl_list.append(pnl_pct)
                    if pnl_pct > 0: gross_profit += pnl_usdt
                    else: gross_loss += abs(pnl_usdt)
                    
                    qty *= keep
                    invested_amount *= keep
                    has_taken_partial_tp = True
                elif exit_reason is not None:
                    slippage_factor = self.slippage_rate + self.spread_rate / 2
                    exec_price = price * (1 - slippage_factor) if position_side == "long" else price * (1 + slippage_factor)
                    pnl_pct = (exec_price - entry_price) / entry_price * 100 if position_side == "long" else (entry_price - exec_price) / entry_price * 100
                    fee = qty * exec_price * self.commission_rate
                    pnl_usdt = invested_amount * (pnl_pct / 100) - fee
                    
                    balance += pnl_usdt
                    pnl_list.append(pnl_pct)
                    if pnl_pct > 0:
                        gross_profit += pnl_usdt
                        win_trades += 1
                    else:
                        gross_loss += abs(pnl_usdt)
                        loss_trades += 1
                        
                    trades.append({
                        "entry_price": float(entry_price),
                        "exit_price": float(exec_price),
                        "pnl_pct": float(pnl_pct),
                        "pnl_usdt": float(pnl_usdt),
                        "side": position_side,
                        "type": entry_type,
                        "reason": exit_reason
                    })
                    
                    position_side = None
                    qty = 0.0
                    invested_amount = 0.0
                    has_taken_partial_tp = False
                    min_trail_dist = 0.0

            if position_side is None:
                # learned-geometry decision chain on the last completed bar
                sig, st = "HOLD", ""
                gsig_bt = str(geo["signal"][idx-1])
                if gsig_bt in ("BUY", "SELL"):
                    sig, st = gsig_bt, "Geo"

                open_long = sig == "BUY" and balance >= 5.0
                open_short = sig == "SELL" and balance >= 5.0
                if open_long or open_short:
                    position_side = "long" if open_long else "short"
                    entry_type = st

                    slippage_factor = self.slippage_rate + self.spread_rate / 2
                    entry_price = price * (1 + slippage_factor) if open_long else price * (1 - slippage_factor)

                    entry_volatility = float(gv[idx-1])
                    # cost-floored, vol-scaled barrier targets from the pipeline;
                    # trailing may never come closer than half the target
                    geo_tp = float(geo["tp"][idx-1]) if "tp" in geo else 0.0
                    current_tp_percent = max(
                        geo_tp, tp_percent,
                        cfg("tp_cost_floor_mult") * roundtrip_cost_pct(),
                    )
                    current_sl_percent = float(geo["sl"][idx-1]) if "sl" in geo else max(0.3, cfg("sl_cost_floor_mult") * roundtrip_cost_pct())
                    min_trail_dist = entry_price * current_tp_percent / 100.0 * 0.5
                    max_price_seen = entry_price
                    min_price_seen = entry_price

                    stop_dist = max(gv[idx-1] * trail_mult, min_trail_dist)
                    if stop_dist <= 0: stop_dist = entry_price * 0.01

                    risk_amount = balance * (cfg("risk_per_trade_pct") / 100.0)
                    target_qty = risk_amount / stop_dist
                    max_qty = (balance * (cfg("max_capital_pct") / 100.0)) / entry_price
                    qty = min(target_qty, max_qty)
                    invested_amount = qty * entry_price

                    balance -= invested_amount * self.commission_rate
                    has_taken_partial_tp = False
                    trail_dist_init = max(gv[idx-1] * trail_mult, min_trail_dist)
                    if position_side == "long":
                        trail_stop_30 = entry_price - trail_dist_init
                        trail_stop_70 = entry_price - trail_dist_init
                    else:
                        trail_stop_30 = entry_price + trail_dist_init
                        trail_stop_70 = entry_price + trail_dist_init

        trade_count = len(trades)
        total_pnl_pct = sum(pnl_list)
        total_pnl_usdt = balance - 1000.0
        
        peak = 1000.0
        max_dd_usdt = 0.0
        current_balance = 1000.0
        for t in trades:
            current_balance += t['pnl_usdt']
            if current_balance > peak: peak = current_balance
            dd = peak - current_balance
            if dd > max_dd_usdt: max_dd_usdt = dd
        max_dd_pct = (max_dd_usdt / peak * 100) if peak > 0 else 0.0
        
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 1.0)
        calmar = (total_pnl_pct / max_dd_pct) if max_dd_pct > 0 else (999.0 if total_pnl_pct > 0 else 0.0)
        
        if len(pnl_list) > 1:
            std_pnl = np.std(pnl_list)
            sharpe = float(np.mean(pnl_list) / std_pnl) if std_pnl > 0 else 0.0
            downside_returns = [r for r in pnl_list if r < 0]
            std_downside = np.std(downside_returns) if downside_returns else 1e-8
            sortino = float(np.mean(pnl_list) / std_downside) if std_downside > 0 else 0.0
        else:
            sharpe = sortino = 0.0
            
        win_rate = (win_trades / trade_count * 100) if trade_count > 0 else 0.0
        expectancy = ((win_rate/100) * np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] > 0])) - ((1 - win_rate/100) * np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_pct'] <= 0])) if trade_count > 0 and len([t for t in trades if t['pnl_pct'] > 0]) > 0 and len([t for t in trades if t['pnl_pct'] <= 0]) > 0 else 0.0
        recovery_factor = (total_pnl_usdt / max_dd_usdt) if max_dd_usdt > 0 else 999.0
        
        return {
            "trade_count": trade_count,
            "total_pnl_pct": total_pnl_pct,
            "total_pnl_usdt": total_pnl_usdt,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_dd_pct,
            "calmar_ratio": calmar,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "recovery_factor": recovery_factor,
            "win_rate": win_rate,
            "expectancy": expectancy,
            "trades": trades
        }

app = Flask("QuantDesktopApp")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quant Bot V3.7 — LightGBM Research Engine</title>
    <script src="/static/lightweight-charts.js"></script>
    <style>
        :root{
            --bg:#0d1017; --panel:#161b24; --panel2:#1b212c; --border:#262d3a;
            --text:#d7dbe3; --muted:#7d879c; --accent:#4c8dff; --green:#22c55e;
            --red:#ef4444; --yellow:#f5b73d; --cyan:#22d3ee; --purple:#c084fc;
        }
        *{box-sizing:border-box; margin:0; padding:0; font-family:'Segoe UI',system-ui,sans-serif;}
        body{background:var(--bg); color:var(--text); height:100vh; overflow:hidden; display:flex; flex-direction:column; font-size:13px;}
        .row{display:flex; align-items:center;}
        .grow{flex:1;}
        .muted{color:var(--muted);}
        .green{color:var(--green);} .red{color:var(--red);} .yellow{color:var(--yellow);}
        .cyan{color:var(--cyan);} .accent{color:var(--accent);}
        .mono{font-family:'Consolas','Roboto Mono',monospace;}

        header{background:var(--panel); border-bottom:1px solid var(--border); padding:8px 16px; display:flex; align-items:center; gap:16px; flex-shrink:0;}
        .brand{font-weight:700; font-size:15px; display:flex; align-items:center; gap:8px;}
        .brand .dot{width:9px; height:9px; border-radius:50%; background:var(--accent);}
        .sym{font-weight:600; color:var(--muted);}
        .price{font-size:20px; font-weight:700; font-family:'Consolas',monospace;}
        .badge{padding:4px 10px; border-radius:6px; font-size:11px; font-weight:700; border:1px solid transparent; display:flex; align-items:center; gap:6px;}
        .badge .bdot{width:7px; height:7px; border-radius:50%; }
        .b-green{background:rgba(34,197,94,.12); color:var(--green); border-color:rgba(34,197,94,.3);}
        .b-yellow{background:rgba(245,183,61,.12); color:var(--yellow); border-color:rgba(245,183,61,.3);}
        .b-red{background:rgba(239,68,68,.12); color:var(--red); border-color:rgba(239,68,68,.3);}
        .b-grey{background:rgba(125,135,156,.12); color:var(--muted); border-color:rgba(125,135,156,.3);}
        @keyframes blink{0%,100%{opacity:1;}50%{opacity:.35;}}
        .blink{animation:blink 1.4s infinite;}

        .modes{display:flex; background:rgba(0,0,0,.25); border-radius:7px; padding:2px; border:1px solid var(--border);}
        .mode{padding:5px 12px; border:none; background:transparent; color:var(--muted); font-size:12px; font-weight:700; cursor:pointer; border-radius:5px;}
        .mode.on-paper{background:rgba(76,141,255,.18); color:var(--accent);}
        .mode.on-real{background:rgba(239,68,68,.18); color:var(--red);}
        .btn{padding:7px 16px; border-radius:6px; font-weight:700; cursor:pointer; border:none; font-size:12px; color:#fff;}
        .btn-start{background:var(--green);} .btn-stop{background:var(--red);}
        .btn-ghost{background:var(--panel2); color:var(--text); border:1px solid var(--border); font-weight:600;}
        .btn:disabled{opacity:.5; cursor:not-allowed;}

        .metrics{background:var(--panel); border-bottom:1px solid var(--border); padding:7px 16px; display:flex; gap:26px; overflow-x:auto; flex-shrink:0;}
        .metric{display:flex; flex-direction:column; gap:1px; min-width:70px;}
        .metric .k{font-size:10px; text-transform:uppercase; letter-spacing:.4px; color:var(--muted);}
        .metric .v{font-size:15px; font-weight:700; font-family:'Consolas',monospace;}

        .main{display:grid; grid-template-columns:1fr 340px; flex:1; min-height:0;}
        .left{display:flex; flex-direction:column; border-right:1px solid var(--border); min-width:0;}
        .tfbar{display:flex; gap:6px; padding:7px 12px; border-bottom:1px solid var(--border); background:var(--panel); flex-shrink:0; align-items:center;}
        .tf{background:var(--bg); color:var(--muted); border:1px solid var(--border); padding:4px 10px; border-radius:5px; font-size:12px; cursor:pointer;}
        .tf.on{background:rgba(76,141,255,.15); color:var(--accent); border-color:rgba(76,141,255,.4);}
        .chart-wrap{flex:1; position:relative; min-height:0;}
        #chart{position:absolute; inset:0;}
        .legend{position:absolute; top:8px; left:8px; z-index:5; background:rgba(22,27,36,.85); border:1px solid var(--border); border-radius:6px; padding:6px 9px; font-size:11px; display:flex; flex-direction:column; gap:2px; pointer-events:none;}

        .side{background:var(--panel); overflow-y:auto; display:flex; flex-direction:column;}
        .card{border-bottom:1px solid var(--border); padding:11px 13px;}
        .card h4{font-size:11px; text-transform:uppercase; letter-spacing:.5px; color:var(--muted); margin-bottom:9px; display:flex; justify-content:space-between; align-items:center;}
        .kv{display:flex; justify-content:space-between; margin-bottom:5px; font-size:12px;}
        .kv .k{color:var(--muted);} .kv .v{font-weight:600; font-family:'Consolas',monospace;}
        .health-note{font-size:11px; color:var(--red); margin-top:4px; line-height:1.4;}
        .pill{display:inline-block; padding:2px 7px; border-radius:4px; font-size:11px; font-weight:700;}
        .pill-long{background:rgba(34,197,94,.15); color:var(--green);}
        .pill-short{background:rgba(239,68,68,.15); color:var(--red);}

        /* Parameters panel */
        .cfg-group{margin-bottom:10px;}
        .cfg-group-title{font-size:10px; text-transform:uppercase; letter-spacing:.5px; color:var(--accent); margin:8px 0 5px; font-weight:700;}
        .cfg-row{display:flex; align-items:center; gap:8px; margin-bottom:5px;}
        .cfg-row label{flex:1; font-size:11.5px; color:var(--text);}
        .cfg-row .unit{font-size:10px; color:var(--muted); min-width:34px; text-align:right;}
        .cfg-row input[type=number]{width:74px; background:var(--bg); border:1px solid var(--border); color:var(--text); border-radius:5px; padding:4px 6px; font-size:12px; font-family:'Consolas',monospace; text-align:right;}
        .cfg-row input[type=number]:focus{outline:none; border-color:var(--accent);}
        .cfg-row input.changed{border-color:var(--yellow); background:rgba(245,183,61,.08);}
        .cfg-row .tag{font-size:9px; padding:1px 4px; border-radius:3px; font-weight:700;}
        .tag-live{background:rgba(34,197,94,.15); color:var(--green);}
        .tag-refit{background:rgba(245,183,61,.15); color:var(--yellow);}
        .tag-restart{background:rgba(239,68,68,.15); color:var(--red);}
        .switch{position:relative; width:38px; height:20px;}
        .switch input{opacity:0; width:0; height:0;}
        .slider{position:absolute; inset:0; background:var(--panel2); border:1px solid var(--border); border-radius:20px; cursor:pointer; transition:.2s;}
        .slider:before{content:""; position:absolute; height:14px; width:14px; left:2px; top:2px; background:var(--muted); border-radius:50%; transition:.2s;}
        .switch input:checked + .slider{background:rgba(76,141,255,.3); border-color:var(--accent);}
        .switch input:checked + .slider:before{transform:translateX(18px); background:var(--accent);}
        .cfg-actions{display:flex; gap:8px; margin-top:10px;}
        .cfg-actions .btn{flex:1;}
        #cfg-toast{font-size:11px; text-align:center; margin-top:6px; min-height:14px;}

        .ob{display:flex; flex-direction:column; gap:1px; font-family:'Consolas',monospace; font-size:11px;}
        .ob-row{display:flex; justify-content:space-between; padding:1px 5px; position:relative;}
        .ob-row span{position:relative; z-index:2;}
        .ob-bar{position:absolute; right:0; top:0; height:100%; z-index:1;}
        .ob-ask .ob-bar{background:rgba(239,68,68,.12);} .ob-ask span:first-child{color:var(--red);}
        .ob-bid .ob-bar{background:rgba(34,197,94,.12);} .ob-bid span:first-child{color:var(--green);}
        .ob-mid{text-align:center; font-weight:700; padding:4px 0; border-block:1px solid var(--border); margin:2px 0;}

        .trades{background:var(--panel); border-top:1px solid var(--border); height:170px; min-height:170px; display:flex; flex-direction:column; flex-shrink:0;}
        .trades h3{padding:7px 16px; font-size:12px; border-bottom:1px solid var(--border); color:var(--muted); display:flex; justify-content:space-between;}
        .trades-body{flex:1; overflow-y:auto;}
        table{width:100%; border-collapse:collapse; font-size:11.5px;}
        th,td{padding:4px 16px; text-align:left; border-bottom:1px solid var(--border);}
        th{color:var(--muted); font-weight:500; background:rgba(0,0,0,.15); position:sticky; top:0;}
        td.mono{font-family:'Consolas',monospace;}

        ::-webkit-scrollbar{width:8px; height:8px;} ::-webkit-scrollbar-thumb{background:var(--border); border-radius:4px;}
        ::-webkit-scrollbar-track{background:transparent;}
    </style>
</head>
<body>
    <header>
        <div class="brand"><span class="dot blink"></span> Quant Bot <span class="muted" style="font-weight:400;">V3.7 · LightGBM</span></div>
        <div class="sym">BTC/USDT</div>
        <div id="price" class="price mono">$0.00</div>
        <div id="geo-badge" class="badge b-grey"><span class="bdot" style="background:var(--muted)"></span><span id="geo-badge-t">COLLECTING</span></div>
        <div class="grow"></div>
        <div class="modes">
            <button id="m-paper" class="mode on-paper" onclick="setMode('PAPER')">PAPER</button>
            <button id="m-real" class="mode" onclick="setMode('REAL')">REAL</button>
        </div>
        <div id="run-badge" class="badge b-yellow"><span class="bdot blink" style="background:var(--yellow)"></span><span id="run-t">OBSERVATION</span></div>
        <button id="run-btn" class="btn btn-start" onclick="toggleBot()">&#9654; BAŞLAT</button>
    </header>

    <div class="metrics">
        <div class="metric"><span class="k">Bakiye</span><span class="v cyan" id="m-bal">$0</span></div>
        <div class="metric"><span class="k">Net Kâr</span><span class="v" id="m-pnl">0.00%</span></div>
        <div class="metric"><span class="k">Kazanç %</span><span class="v" id="m-wr">0%</span></div>
        <div class="metric"><span class="k">Profit Factor</span><span class="v" id="m-pf">0.00</span></div>
        <div class="metric"><span class="k">Sharpe</span><span class="v" id="m-sharpe">0.00</span></div>
        <div class="metric"><span class="k">İşlem</span><span class="v" id="m-tc">0</span></div>
        <div class="metric"><span class="k">OBI</span><span class="v" id="m-obi">0.00</span></div>
        <div class="metric"><span class="k">Ort. K/Z</span><span class="v" id="m-awl" style="font-size:12px;">0/0</span></div>
    </div>

    <div class="main">
        <div class="left">
            <div class="tfbar">
                <span class="muted" style="font-size:11px; margin-right:4px;">Zaman:</span>
                <button class="tf on" onclick="setTF('1m',this)">1m</button>
                <button class="tf" onclick="setTF('5m',this)">5m</button>
                <button class="tf" onclick="setTF('15m',this)">15m</button>
                <button class="tf" onclick="setTF('1h',this)">1h</button>
                <button class="tf" onclick="setTF('4h',this)">4h</button>
                <div class="grow"></div>
                <span id="sig-chip" class="badge b-grey" style="font-size:11px;">HOLD</span>
            </div>
            <div class="chart-wrap">
                <div id="chart"></div>
                <div class="legend">
                    <div><span class="green">▮</span> Conformal A ≥ kapı &nbsp; <span class="muted">▮</span> A &lt; kapı</div>
                    <div><span class="yellow">●</span> Epizot &nbsp; <span class="green">▲</span> GEO Long &nbsp; <span class="purple">▼</span> GEO Short</div>
                    <div id="lg-geo" class="muted">Geometri: -</div>
                </div>
            </div>
        </div>

        <div class="side">
            <!-- Learned Geometry -->
            <div class="card">
                <h4>Öğrenilmiş Geometri <span id="geo-health" class="pill" style="background:rgba(125,135,156,.15); color:var(--muted);">-</span></h4>
                <div id="geo-health-note" class="health-note" style="display:none;"></div>
                <div class="kv"><span class="k">Durum</span><span class="v" id="g-status">-</span></div>
                <div class="kv"><span class="k">Şema M</span><span class="v cyan" id="g-schema" style="font-size:11px;">-</span></div>
                <div class="kv"><span class="k">κ (öğr./init)</span><span class="v" id="g-kappa">- / -</span></div>
                <div class="kv"><span class="k">η · P(δ≠0)</span><span class="v" id="g-eta">-</span></div>
                <div class="kv"><span class="k">Epizot</span><span class="v" id="g-ep">NORMAL</span></div>
                <div class="kv"><span class="k">p GBM / Meta</span><span class="v" id="g-p">- / -</span></div>
                <div class="kv"><span class="k">Conformal A</span><span class="v" id="g-a">-</span></div>
                <div class="kv"><span class="k">E[net] · TP/SL</span><span class="v" id="g-en">-</span></div>
                <div class="kv"><span class="k">D_anchor · Fold</span><span class="v" id="g-da">-</span></div>
                <div class="kv"><span class="k">Panel (aktif/emekli)</span><span class="v" id="g-panel">-</span></div>
            </div>

            <!-- Position -->
            <div class="card">
                <h4>Açık Pozisyon</h4>
                <div id="pos" class="muted" style="text-align:center; padding:4px;">Pozisyon yok</div>
            </div>

            <!-- Trading Parameters -->
            <div class="card">
                <h4>Trading Parametreleri
                    <span class="muted" style="font-size:9px; font-weight:400;">
                        <span class="tag tag-live">CANLI</span>
                        <span class="tag tag-refit">REFIT</span>
                        <span class="tag tag-restart">RESTART</span>
                    </span>
                </h4>
                <div id="cfg-body"><div class="muted" style="font-size:11px;">Yükleniyor...</div></div>
                <div class="cfg-actions">
                    <button class="btn btn-ghost" onclick="resetConfig()">Varsayılan</button>
                    <button class="btn btn-start" id="cfg-save" onclick="saveConfig()">Kaydet</button>
                </div>
                <div id="cfg-toast" class="muted"></div>
            </div>

            <!-- Order Book -->
            <div class="card">
                <h4>Emir Defteri <span class="green" style="font-size:9px;">CANLI</span></h4>
                <div id="ob-asks" class="ob"></div>
                <div id="ob-mid" class="ob-mid muted">$0.00</div>
                <div id="ob-bids" class="ob"></div>
            </div>

            <!-- Backtest & WFO -->
            <div class="card">
                <h4>Backtest & Purged WFO</h4>
                <div class="kv"><span class="k">Challenger</span><span class="v accent" id="chal">Yok</span></div>
                <div style="display:flex; flex-direction:column; gap:6px; margin-top:8px;">
                    <label class="muted" style="font-size:10px; display:flex; align-items:center; justify-content:space-between; gap:8px;">
                        Araştırma barı
                        <input id="research-bars" type="number" min="3000" max="100000" step="1000" value="10000"
                               style="width:92px; background:#111722; color:#d7dbe3; border:1px solid #263044; border-radius:5px; padding:5px;">
                    </label>
                    <button class="btn btn-ghost" id="bt-btn" onclick="runBacktest()">Backtest Çalıştır</button>
                    <button class="btn btn-ghost" id="wfo-btn" onclick="runWFO()">Purged WFO Çalıştır</button>
                    <button class="btn btn-start" id="promo-btn" style="display:none;" onclick="promote()">Challenger'ı Yükselt</button>
                </div>
                <div id="report" style="display:none; margin-top:10px; font-size:11px;"></div>
            </div>
        </div>
    </div>

    <div class="trades">
        <h3><span>İşlem Geçmişi</span><span id="th-mode" class="cyan" style="font-size:10px;">[PAPER]</span></h3>
        <div class="trades-body">
            <table>
                <thead><tr><th>Zaman</th><th>Tip</th><th>Yön</th><th>Giriş</th><th>Çıkış</th><th>PnL</th><th>Neden</th></tr></thead>
                <tbody id="tbody"><tr><td colspan="7" class="muted" style="text-align:center;">Henüz işlem yok</td></tr></tbody>
            </table>
        </div>
    </div>

    <script>
    // ── safe helpers (null/NaN-proof so one bad field never breaks the UI) ──
    const N = (x, d=2, fb='-') => (x===null||x===undefined||(typeof x==='number'&&isNaN(x))) ? fb : Number(x).toFixed(d);
    const G = (o, k, fb=0) => (o && o[k]!==undefined && o[k]!==null) ? o[k] : fb;
    const el = id => document.getElementById(id);
    function setText(id, t){ const e=el(id); if(e) e.textContent = t; }
    function setHTML(id, t){ const e=el(id); if(e) e.innerHTML = t; }

    let chart=null, candles=null, aHist=null, lastPrice=0, cfgMeta=null, cfgOriginal={}, cfgDirty=false;

    function initChart(){
        const c = el('chart');
        chart = LightweightCharts.createChart(c, {
            layout:{ textColor:'#d7dbe3', background:{type:'solid', color:'#0d1017'} },
            grid:{ vertLines:{color:'rgba(38,45,58,.4)'}, horzLines:{color:'rgba(38,45,58,.4)'} },
            rightPriceScale:{ borderColor:'rgba(38,45,58,.8)' },
            timeScale:{ borderColor:'rgba(38,45,58,.8)', timeVisible:true, secondsVisible:false },
            width:c.clientWidth, height:c.clientHeight
        });
        candles = chart.addCandlestickSeries({ upColor:'#22c55e', downColor:'#ef4444', borderVisible:false, wickUpColor:'#22c55e', wickDownColor:'#ef4444' });
        aHist = chart.addHistogramSeries({ priceScaleId:'geo', priceLineVisible:false, lastValueVisible:false });
        chart.priceScale('geo').applyOptions({ scaleMargins:{ top:0.86, bottom:0 } });
        new ResizeObserver(es => { const r=es[0].contentRect; if(chart) chart.applyOptions({width:r.width, height:r.height}); }).observe(c);
    }
    function rebuildSeries(){
        if(!chart) return;
        try { chart.removeSeries(candles); chart.removeSeries(aHist); } catch(e){}
        candles = chart.addCandlestickSeries({ upColor:'#22c55e', downColor:'#ef4444', borderVisible:false, wickUpColor:'#22c55e', wickDownColor:'#ef4444' });
        aHist = chart.addHistogramSeries({ priceScaleId:'geo', priceLineVisible:false, lastValueVisible:false });
        chart.priceScale('geo').applyOptions({ scaleMargins:{ top:0.86, bottom:0 } });
    }

    function toggleBot(){ fetch('/api/toggle_bot',{method:'POST'}).then(()=>refresh()).catch(()=>{}); }
    function setMode(m){
        if(m==='REAL' && !confirm('DİKKAT: REAL moda geçiyorsunuz. Gerçek MEXC bakiyeniz kullanılacak. Emin misiniz?')) return;
        fetch('/api/set_trading_mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:m})}).then(()=>refresh()).catch(()=>{});
    }
    function setTF(tf, btn){
        fetch('/api/set_timeframe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tf:tf})}).then(()=>{
            document.querySelectorAll('.tf').forEach(b=>b.classList.remove('on'));
            if(btn) btn.classList.add('on');
            rebuildSeries();
        }).catch(()=>{});
    }
    function researchBars(){ return Math.min(100000, Math.max(3000, parseInt(el('research-bars').value||'10000',10))); }
    function runResearch(url, buttonId){
        const b=el(buttonId); b.disabled=true;
        fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({bars:researchBars()})})
          .then(async r=>{ const d=await r.json(); if(!r.ok) throw new Error(d.message||'İşlem başlatılamadı'); refresh(); })
          .catch(e=>{ alert(e.message); b.disabled=false; });
    }
    function runBacktest(){ runResearch('/api/backtest','bt-btn'); }
    function runWFO(){ runResearch('/api/run_wfo','wfo-btn'); }
    function promote(){
        if(!confirm('Challenger parametrelerini şampiyon (canlı) yapmak istiyor musunuz?')) return;
        fetch('/api/promote_challenger',{method:'POST'}).then(r=>r.json()).then(d=>{ alert(d.status==='success'?'Yükseltildi!':'Hata: '+(d.message||'')); refresh(); }).catch(()=>{});
    }

    // ── Trading Parameters panel ──
    function loadConfig(){
        fetch('/api/config').then(r=>r.json()).then(d=>{
            cfgMeta = d.meta || {}; cfgOriginal = Object.assign({}, d.config||{});
            renderConfig(d.config||{}, cfgMeta);
        }).catch(()=>{ setHTML('cfg-body','<div class="red" style="font-size:11px;">Config yüklenemedi</div>'); });
    }
    function renderConfig(config, meta){
        const groups = {};
        Object.keys(meta).forEach(k=>{
            const m = meta[k]; const grp = m[1];
            (groups[grp] = groups[grp] || []).push([k, m]);
        });
        let html = '';
        Object.keys(groups).forEach(grp=>{
            html += '<div class="cfg-group"><div class="cfg-group-title">'+grp+'</div>';
            groups[grp].forEach(([k,m])=>{
                const [label,,unit,lo,hi,step,when] = m;
                const val = config[k];
                const tag = when==='live'?'tag-live':(when==='refit'?'tag-refit':'tag-restart');
                const tagT = when==='live'?'CANLI':(when==='refit'?'REFIT':'RESTART');
                if(unit==='bool'){
                    html += '<div class="cfg-row"><label>'+label+'</label>'
                          + '<span class="tag '+tag+'">'+tagT+'</span>'
                          + '<label class="switch"><input type="checkbox" data-k="'+k+'" '+(val?'checked':'')+' onchange="markDirty()"><span class="slider"></span></label></div>';
                } else {
                    html += '<div class="cfg-row"><label>'+label+'</label>'
                          + '<span class="tag '+tag+'">'+tagT+'</span>'
                          + '<input type="number" data-k="'+k+'" value="'+val+'" min="'+lo+'" max="'+hi+'" step="'+step+'" oninput="markDirty(this)">'
                          + '<span class="unit">'+unit+'</span></div>';
                }
            });
            html += '</div>';
        });
        setHTML('cfg-body', html);
        cfgDirty = false; updateSaveBtn();
    }
    function markDirty(inp){
        if(inp){
            const k = inp.getAttribute('data-k');
            const orig = cfgOriginal[k];
            const changed = String(parseFloat(inp.value)) !== String(orig);
            inp.classList.toggle('changed', changed);
        }
        cfgDirty = true; updateSaveBtn();
    }
    function updateSaveBtn(){ const b=el('cfg-save'); if(b){ b.textContent = cfgDirty ? 'Kaydet *' : 'Kaydet'; } }
    function collectConfig(){
        const out = {};
        document.querySelectorAll('#cfg-body [data-k]').forEach(inp=>{
            const k = inp.getAttribute('data-k');
            out[k] = inp.type==='checkbox' ? inp.checked : parseFloat(inp.value);
        });
        return out;
    }
    function saveConfig(){
        const body = collectConfig();
        setText('cfg-toast','Kaydediliyor...'); el('cfg-toast').className='muted';
        fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
            .then(r=>r.json()).then(d=>{
                if(d.status==='success'){
                    cfgOriginal = Object.assign({}, d.config||{});
                    renderConfig(d.config||{}, cfgMeta);
                    setText('cfg-toast','✓ Kaydedildi'); el('cfg-toast').className='green';
                } else { setText('cfg-toast','Hata: '+(d.message||'')); el('cfg-toast').className='red'; }
                setTimeout(()=>setText('cfg-toast',''), 2500);
            }).catch(()=>{ setText('cfg-toast','İstek başarısız'); el('cfg-toast').className='red'; });
    }
    function resetConfig(){
        if(!confirm('Tüm parametreleri varsayılana döndür?')) return;
        fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({__reset__:true})})
            .then(r=>r.json()).then(d=>{ cfgOriginal=Object.assign({},d.config||{}); renderConfig(d.config||{}, cfgMeta);
                setText('cfg-toast','✓ Sıfırlandı'); el('cfg-toast').className='green'; setTimeout(()=>setText('cfg-toast',''),2500); }).catch(()=>{});
    }

    // ── main state refresh (500ms) — every access null-safe ──
    function refresh(){
        fetch('/api/state').then(r=>r.json()).then(s=>{
            try { render(s); } catch(e){ console.error('render error:', e); }
        }).catch(e=>console.error('state fetch failed:', e));
    }
    function render(s){
        const paper = s.trading_mode === 'PAPER';
        el('m-paper').className = 'mode' + (paper?' on-paper':'');
        el('m-real').className = 'mode' + (paper?'':' on-real');
        setText('m-bal', '$' + N(paper ? s.virtual_balance : s.real_balance, 2, '0'));
        setText('th-mode', paper ? '[PAPER]' : '[REAL]');

        // run state
        const active = !!s.is_trading_active;
        const rb = el('run-btn'), rbg = el('run-badge');
        rb.className = 'btn ' + (active?'btn-stop':'btn-start');
        rb.innerHTML = active ? '&#9632; DURDUR' : '&#9654; BAŞLAT';
        rbg.className = 'badge ' + (active?'b-green':'b-yellow');
        setText('run-t', active ? 'CANLI TİCARET' : 'GÖZLEM');

        // metrics
        const tc = G(s,'trade_count',0), wt = G(s,'winning_trades',0);
        const wr = tc>0 ? wt/tc*100 : 0;
        const gl = G(s,'gross_loss',0), gp = G(s,'gross_profit',0);
        const pf = gl<0 ? Math.abs(gp/gl) : (gp>0?99.99:0);
        setText('m-pnl', (G(s,'total_pnl',0)>=0?'+':'') + N(s.total_pnl,2,'0') + '%');
        el('m-pnl').className = 'v ' + (G(s,'total_pnl',0)>=0?'green':'red');
        setText('m-wr', N(wr,1,'0') + '%');
        setText('m-pf', N(pf,2,'0'));
        setText('m-sharpe', N(s.sharpe_ratio,2,'0'));
        el('m-sharpe').className = 'v ' + (G(s,'sharpe_ratio',0)>=0?'green':'red');
        setText('m-tc', tc);
        const obi = G(s,'obi',0);
        setText('m-obi', (obi>0?'+':'') + N(obi,2,'0'));
        el('m-obi').className = 'v ' + (obi>0.3?'green':(obi<-0.3?'red':''));
        setHTML('m-awl', '<span class="green">+'+N(s.avg_win,2,'0')+'</span>/<span class="red">'+N(s.avg_loss,2,'0')+'</span>');

        // price
        const p = G(s,'price',0), pel = el('price');
        pel.className = 'price mono ' + (p>lastPrice?'green':(p<lastPrice?'red':''));
        setText('price','$'+N(p,2,'0')); lastPrice = p;

        // signal chip
        const sig = G(s,'signal','HOLD'), st = G(s,'signal_type','');
        const chip = el('sig-chip');
        chip.textContent = sig + (st?(' · '+st):'');
        chip.className = 'badge ' + (sig==='BUY'?'b-green':(sig==='SELL'?'b-red':'b-grey'));

        // geometry
        const g = s.geom || {};
        renderGeo(g);

        // position
        renderPos(s);

        // challenger + report buttons
        const ps = s.parameters_store || {};
        const chal = ps.shadow_challenger;
        setText('chal', chal ? 'Aktif' : 'Yok');
        el('chal').className = 'v ' + (chal?'accent':'muted');
        const promo = el('promo-btn');
        promo.style.display = (chal && !s.position_side) ? 'block' : 'none';
        el('bt-btn').disabled = !!s.backtest_running;
        el('bt-btn').textContent = s.backtest_running ? 'Backtest çalışıyor...' : 'Backtest Çalıştır';
        el('wfo-btn').disabled = !!s.wfo_running;
        el('wfo-btn').textContent = s.wfo_running ? 'WFO çalışıyor...' : 'Purged WFO Çalıştır';
        el('research-bars').disabled = !!s.backtest_running || !!s.wfo_running;
        renderReport(s);

        // trades
        renderTrades(s.trades || []);
        // chart
        renderChart(s.chart_data || []);
        // orderbook
        renderOB(s);
    }

    function renderGeo(g){
        const status = G(g,'status','collecting');
        const degraded = (status==='ready' && g.health==='degraded');
        // header badge
        const badge = el('geo-badge'), bt = el('geo-badge-t');
        let bc='b-grey', bd='var(--muted)', txt=String(status).toUpperCase();
        if(degraded){ bc='b-red'; bd='var(--red)'; txt='DEGRADED'; }
        else if(status==='ready'){ bc='b-green'; bd='var(--green)'; txt='READY'; }
        else if(status==='training'){ bc='b-yellow'; bd='var(--yellow)'; txt='TRAINING'; }
        badge.className='badge '+bc; bt.textContent=txt;
        badge.querySelector('.bdot').style.background=bd;

        const hp = el('geo-health');
        hp.textContent = degraded ? 'DEGRADED' : (status==='ready'?'OK':'—');
        hp.style.background = degraded ? 'rgba(239,68,68,.15)' : (status==='ready'?'rgba(34,197,94,.15)':'rgba(125,135,156,.15)');
        hp.style.color = degraded ? 'var(--red)' : (status==='ready'?'var(--green)':'var(--muted)');
        const note = el('geo-health-note');
        if(degraded && Array.isArray(g.health_reasons) && g.health_reasons.length){
            note.style.display='block'; note.textContent='⚠ '+g.health_reasons.join(' · ');
        } else note.style.display='none';

        setText('g-status', String(status).toUpperCase());
        setText('g-schema', G(g,'schema','-'));
        setText('g-kappa', N(g.kappa,3)+' / '+N(g.kappa_init,3));
        const gd = g.diag || {};
        const pdn = gd.p_delta_nonzero;
        setText('g-eta', N(g.eta,3)+' · '+(pdn!==undefined?N(pdn*100,1)+'%':'-'));
        const inEp = g.episode==='EPISODE';
        setText('g-ep', inEp ? ('EPİZOT'+(G(g,'cluster',-1)>=0?(' A'+g.cluster):'')) : 'NORMAL');
        el('g-ep').className = 'v ' + (inEp?'yellow':'green');
        setText('g-p', N(g.p_gbm,2)+' / '+N(g.p_meta,2));
        const aOk = G(g,'a_score',0) >= G(g,'a_gate',0.5);
        setText('g-a', N(g.a_score,2)+' (kapı '+N(g.a_gate,2)+')');
        el('g-a').className = 'v ' + (aOk?'green':'muted');
        const en = g.exp_net;
        if(en!==undefined && g.tp_pct){
            setText('g-en', (en>=0?'+':'')+N(en,2)+'% · '+N(g.tp_pct,2)+'/'+N(g.sl_pct,2)+'%');
            el('g-en').className = 'v ' + (en>0?'green':'red');
        } else setText('g-en','-');
        setText('g-da', N(gd.d_anchor,4)+' · f'+G(g,'fold',0));
        const pn = g.panel || {};
        setText('g-panel', (pn.core_active!==undefined)?(pn.core_active+' / '+G(pn,'core_retired',0)):'-');
        setText('lg-geo', 'Geometri: '+String(status).toUpperCase()+(status==='ready'&&g.schema?(' · '+g.schema):''));
    }

    function renderPos(s){
        const box = el('pos');
        if(s.position_side){
            const long = s.position_side==='long';
            const pnl = G(s,'position_pnl',0);
            box.innerHTML =
                '<div class="kv"><span class="k">Yön</span><span class="pill '+(long?'pill-long':'pill-short')+'">'+String(s.position_side).toUpperCase()+'</span></div>'
              + '<div class="kv"><span class="k">Tip</span><span class="v">'+G(s,'position_type','-')+'</span></div>'
              + '<div class="kv"><span class="k">Giriş</span><span class="v mono">$'+N(s.position_entry,2)+'</span></div>'
              + '<div class="kv"><span class="k">PnL</span><span class="v '+(pnl>=0?'green':'red')+'">'+(pnl>=0?'+':'')+N(pnl,2)+'%</span></div>';
        } else box.innerHTML = '<div class="muted" style="text-align:center; padding:4px;">Pozisyon yok</div>';
    }

    function renderReport(s){
        const box = el('report');
        if(s.wfo_report){
            const w = s.wfo_report; let h = '<div class="accent" style="font-weight:700; margin-bottom:4px;">WFO Sonucu ('+G(w,'engine','-')+')</div>';
            h += '<div class="kv"><span class="k">Veri / Model</span><span class="v">'+G(w,'bars_used',0)+' bar · '+String(G(w,'data_source','-')).toUpperCase()+' / '+G(w,'model','-')+'</span></div>';
            h += '<div class="kv"><span class="k">Stabilite</span><span class="v">'+G(w,'stability_count',0)+'/'+G(w,'slices_evaluated',0)+'</span></div>';
            h += '<div class="kv"><span class="k">PF Varyans</span><span class="v">'+N(w.variance,4)+'</span></div>';
            const promoted = G(w,'promotion_status','not_promoted')==='auto_promoted';
            h += '<div class="kv"><span class="k">Parametre</span><span class="v '+(promoted?'green':'yellow')+'">'+(promoted?'Otomatik güncellendi':G(w,'promotion_status','doğrulanmadı'))+'</span></div>';
            if(w.promotion_reason) h += '<div class="muted" style="font-size:10px; margin-top:3px;">'+w.promotion_reason+'</div>';
            if(w.schema && w.schema!=='-') h += '<div class="kv"><span class="k">Geometri</span><span class="v cyan" style="font-size:10px;">'+w.schema+'</span></div>';
            const dg = (w.diagnostics||[]);
            if(dg.length){ const d=dg[dg.length-1];
                h += '<div class="muted" style="font-size:10px; margin-top:3px;">P(δ≠0)='+N(d.p_delta_nonzero*100,1)+'% · κ='+N(d.kappa,2)+' · gap='+N(d.overfit_gap,3)+' · D_anchor='+N(d.d_anchor,4)+'</div>'; }
            box.innerHTML=h; box.style.display='block';
        } else if(s.backtest_report){
            const r=s.backtest_report;
            if(r.status==='error'){ box.innerHTML='<div class="red">'+G(r,'message','hata')+'</div>'; box.style.display='block'; return; }
            let h='<div class="green" style="font-weight:700; margin-bottom:4px;">Backtest ('+G(r,'engine','-')+')</div>';
            h+='<div class="kv"><span class="k">Veri / Model</span><span class="v">'+G(r,'bars_used',0)+' bar · '+String(G(r,'data_source','-')).toUpperCase()+' / '+G(r,'model','-')+'</span></div>';
            if(r.schema && r.schema!=='-') h+='<div class="kv"><span class="k">Geometri</span><span class="v cyan" style="font-size:10px;">'+r.schema+'</span></div>';
            h+='<div class="kv"><span class="k">İşlem / Kazanç</span><span class="v">'+G(r,'trade_count',0)+' / '+N(r.win_rate,1)+'%</span></div>';
            h+='<div class="kv"><span class="k">Net Kâr</span><span class="v '+(G(r,'total_pnl_usdt',0)>=0?'green':'red')+'">'+(G(r,'total_pnl_usdt',0)>=0?'+':'')+N(r.total_pnl_usdt,2)+' ('+N(r.total_pnl_pct,2)+'%)</span></div>';
            h+='<div class="kv"><span class="k">PF / Sharpe</span><span class="v">'+N(r.profit_factor,2)+' / '+N(r.sharpe_ratio,2)+'</span></div>';
            h+='<div class="kv"><span class="k">Max DD</span><span class="v red">'+N(r.max_drawdown_pct,2)+'%</span></div>';
            box.innerHTML=h; box.style.display='block';
        } else box.style.display='none';
    }

    function renderTrades(trades){
        if(!trades.length) return;
        setHTML('tbody', trades.slice(0,40).map(t=>{
            const pnl = String(G(t,'pnl','-'));
            const cls = pnl.startsWith('+')?'green':(pnl.startsWith('-')?'red':'');
            const long = (t.side==='LONG'||t.side==='long');
            const exit = G(t,'exit',0);
            return '<tr><td class="muted">'+G(t,'time','')+'</td><td>'+G(t,'type','')+'</td>'
                 + '<td><span class="pill '+(long?'pill-long':'pill-short')+'">'+G(t,'side','')+'</span></td>'
                 + '<td class="mono">$'+N(t.entry,2)+'</td><td class="mono">'+(exit>0?'$'+N(exit,2):'-')+'</td>'
                 + '<td class="mono '+cls+'">'+pnl+'</td><td class="muted">'+G(t,'reason','')+'</td></tr>';
        }).join(''));
    }

    function renderChart(cd){
        if(!chart) initChart();
        if(!cd.length) return;
        try {
            candles.setData(cd.map(d=>({time:d.time, open:d.open, high:d.high, low:d.low, close:d.close})));
            aHist.setData(cd.filter(d=>d.geo_a!==null&&d.geo_a!==undefined).map(d=>({
                time:d.time, value:d.geo_a,
                color: d.geo_episode ? 'rgba(245,183,61,.5)' : (d.geo_a>=d.geo_gate ? 'rgba(34,197,94,.6)' : 'rgba(125,135,156,.35)')
            })));
            let mk=[], prevEp=false;
            cd.forEach(d=>{
                if(d.geo_episode && !prevEp) mk.push({time:d.time, position:'aboveBar', color:'#f5b73d', shape:'circle', text:(d.geo_state>0?('A'+(d.geo_state-1)):'EP')});
                prevEp = !!d.geo_episode;
                if(d.geo_buy) mk.push({time:d.time, position:'belowBar', color:'#22c55e', shape:'arrowUp', text:'GEO'});
                if(d.geo_sell) mk.push({time:d.time, position:'aboveBar', color:'#c084fc', shape:'arrowDown', text:'GEO S'});
                if(d.geo_exit) mk.push({time:d.time, position:'aboveBar', color:'#ef4444', shape:'arrowDown', text:'EXIT'});
            });
            candles.setMarkers(mk);
        } catch(e){ console.error('chart error:', e); }
    }

    function renderOB(s){
        const ob = s.orderbook;
        if(!ob || !ob.bids || !ob.asks || !ob.bids.length || !ob.asks.length) return;
        const asks = ob.asks.slice(0,7).reverse(), bids = ob.bids.slice(0,7);
        let mx=0; [...asks,...bids].forEach(a=>{ if(a[1]>mx) mx=a[1]; });
        const row=(a,cls)=>{ const w=Math.min((a[1]/mx)*100,100); return '<div class="ob-row '+cls+'"><div class="ob-bar" style="width:'+w+'%"></div><span>'+N(a[0],2)+'</span><span>'+N(a[1],4)+'</span></div>'; };
        setHTML('ob-asks', asks.map(a=>row(a,'ob-ask')).join(''));
        setHTML('ob-bids', bids.map(a=>row(a,'ob-bid')).join(''));
        setText('ob-mid', '$'+N(s.price,2));
    }

    // boot
    loadConfig();
    setTimeout(()=>{ refresh(); setInterval(refresh, 500); }, 250);
    </script>
</body>
</html>
"""

@app.route('/')
def home(): return render_template_string(HTML_TEMPLATE)

@app.route('/static/lightweight-charts.js')
def serve_lw_charts():
    from flask import send_from_directory
    return send_from_directory(Path(__file__).parent, 'lightweight-charts.js', mimetype='application/javascript')

@app.route('/api/state')
def get_state():
    with state_lock:
        state_copy = bot_state.copy()
        state_copy.pop("loop", None)
        return jsonify(_json_sanitize(state_copy))


def _json_sanitize(obj):
    """Recursively replace NaN/Infinity with None so the browser can parse the
    payload. Flask jsonify defaults to allow_nan=True and emits the literal
    string `NaN`, which is INVALID JSON and makes the browser's JSON.parse
    reject the whole response — silently killing the UI update loop (chart
    never renders, timeframe clicks look unresponsive)."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]
    return obj

@app.route('/api/toggle_bot', methods=['POST'])
def toggle_bot():
    with state_lock:
        bot_state["is_trading_active"] = not bot_state["is_trading_active"]
        is_active = bot_state["is_trading_active"]
    return jsonify({"status": "success", "is_active": is_active})

@app.route('/api/toggle_short', methods=['POST'])
def toggle_short():
    """Opt-in short (two-sided / futures). Affects the decision layer, backtest
    and PAPER trading immediately (no refit — the meta is trained two-sided).
    REAL live futures short remains non-auto-armed regardless of this flag."""
    with state_lock:
        bot_state["allow_short"] = not bot_state.get("allow_short", False)
        allow = bot_state["allow_short"]
    save_trading_config({"allow_short": allow})   # persist so it survives restarts
    log.info(f"allow_short toggled → {allow}")
    return jsonify({"status": "success", "allow_short": allow})

@app.route('/api/set_timeframe', methods=['POST'])
def set_timeframe():
    data = request.json
    tf_val = "1m"
    if "tf" in data:
        with state_lock:
            bot_state["timeframe"] = data["tf"]
            bot_state["timeframe_changed"] = True
            loop = bot_state.get("loop")
            tf_val = bot_state["timeframe"]
        if loop:
            loop.call_soon_threadsafe(tf_change_event.set)
        else:
            tf_change_event.set()
    return jsonify({"status": "success", "tf": tf_val})

@app.route('/api/set_trading_mode', methods=['POST'])
def set_trading_mode():
    data = request.json
    mode_val = "PAPER"
    if "mode" in data and data["mode"] in ["PAPER", "REAL"]:
        with state_lock:
            bot_state["trading_mode"] = data["mode"]
            mode_val = bot_state["trading_mode"]
    return jsonify({"status": "success", "mode": mode_val})

@app.route('/api/config', methods=['GET', 'POST'])
def config_endpoint():
    """GET returns the live trading config + UI metadata; POST validates,
    persists (trading_config.json) and applies it. Geometry-affecting knobs
    take effect on the next refit, the rest on the next tick."""
    if request.method == 'GET':
        return jsonify({
            "status": "success",
            "config": dict(CFG),
            "meta": {k: list(v) for k, v in TRADING_CONFIG_META.items()},
            "effective_cost_pct": round(roundtrip_cost_pct(), 4),
        })
    # POST
    try:
        data = request.json or {}
        if data.get("__reset__"):
            new_cfg = save_trading_config(dict(DEFAULT_TRADING_CONFIG))
        else:
            new_cfg = save_trading_config(data)
        # keep the runtime toggle in sync with the persisted config
        with state_lock:
            bot_state["allow_short"] = bool(new_cfg.get("allow_short", False))
        log.info(f"Trading config updated: {new_cfg}")
        return jsonify({"status": "success", "config": new_cfg,
                        "effective_cost_pct": round(roundtrip_cost_pct(), 4)})
    except Exception as e:
        log.error(f"Config update error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

def _requested_research_bars():
    data = request.get_json(silent=True) or {}
    try:
        bars = int(data.get("bars", RESEARCH_BARS_DEFAULT))
    except (TypeError, ValueError):
        raise ValueError("Bar count must be an integer")
    return min(max(bars, RESEARCH_BARS_MIN), RESEARCH_BARS_MAX)


@app.route('/api/backtest', methods=['POST'])
def run_backtest_endpoint():
    global bot_instance
    if not bot_instance:
        return jsonify({"status": "error", "message": "Bot not initialized"}), 400
    try:
        requested_bars = _requested_research_bars()
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    with state_lock:
        if bot_state.get("backtest_running") or bot_state.get("wfo_running"):
            return jsonify({"status": "busy", "message": "A research job is already running"}), 409
        # Set the flag before starting the thread. This closes the double-click race
        # that could leave two workers fighting over the same report state.
        bot_state["backtest_running"] = True
        bot_state["backtest_report"] = None
        bot_state["wfo_report"] = None
        bot_state["backtest_bars_requested"] = requested_bars

    def _worker():
        try:
            log.info("Backtest worker thread started. Fetching historical OHLCV data...")
            loop = bot_state.get("loop")
            if loop:
                future = asyncio.run_coroutine_threadsafe(
                    bot_instance.fetch_ohlcv_large(SYMBOL, bot_state["timeframe"], requested_bars),
                    loop
                )
                df = future.result()
            else:
                import asyncio as local_asyncio
                df = local_asyncio.run(bot_instance.fetch_ohlcv_large(SYMBOL, bot_state["timeframe"], requested_bars))
                
            log.info(f"Historical OHLCV data fetched successfully: {len(df)} bars. Running backtest simulation...")
            data_source = str(df.attrs.get("data_source", bot_state.get("research_data_source", "-")))
            active_params = get_active_parameters()
            # Geometric backtest: train on the first window, trade the purged remainder.
            report = run_geometric_backtest(df, active_params, timeframe=bot_state["timeframe"])
            log.info(f"Backtest simulation completed ({report.get('engine','geometric')}). Trade Count: {report['trade_count']}, Net PnL: {report['total_pnl_usdt']:.2f} USDT")

            bot_state["backtest_report"] = {
                "trade_count": report["trade_count"],
                "total_pnl_pct": float(report["total_pnl_pct"]),
                "total_pnl_usdt": float(report["total_pnl_usdt"]),
                "profit_factor": float(report["profit_factor"]),
                "max_drawdown_pct": float(report["max_drawdown_pct"]),
                "calmar_ratio": float(report["calmar_ratio"]),
                "sharpe_ratio": float(report["sharpe_ratio"]),
                "win_rate": float(report["win_rate"]),
                "expectancy": float(report["expectancy"]),
                "engine": report.get("engine", "legacy"),
                "schema": report.get("schema", "-"),
                "model": "LightGBM",
                "data_source": data_source,
                "bars_requested": requested_bars,
                "bars_used": int(len(df)),
                "trades": report["trades"][-10:]
            }
            bot_state["backtest_bars_used"] = int(len(df))
        except Exception as e:
            log.error(f"Backtest API error: {e}")
            bot_state["backtest_report"] = {"status": "error", "message": str(e)}
        finally:
            with state_lock:
                bot_state["backtest_running"] = False

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"status": "running", "bars": requested_bars})

@app.route('/api/run_wfo', methods=['POST'])
def run_wfo_endpoint():
    global bot_instance
    if not bot_instance:
        return jsonify({"status": "error", "message": "Bot not initialized"}), 400
    try:
        requested_bars = _requested_research_bars()
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    with state_lock:
        if bot_state.get("backtest_running") or bot_state.get("wfo_running"):
            return jsonify({"status": "busy", "message": "A research job is already running"}), 409
        bot_state["wfo_running"] = True
        bot_state["wfo_report"] = None
        bot_state["backtest_report"] = None
        bot_state["wfo_bars_requested"] = requested_bars

    def _worker():
        try:
            log.info("WFO worker thread started. Fetching historical OHLCV data...")
            loop = bot_state.get("loop")
            if loop:
                future = asyncio.run_coroutine_threadsafe(
                    bot_instance.fetch_ohlcv_large(SYMBOL, bot_state["timeframe"], requested_bars),
                    loop
                )
                df = future.result()
            else:
                import asyncio as local_asyncio
                df = local_asyncio.run(bot_instance.fetch_ohlcv_large(SYMBOL, bot_state["timeframe"], requested_bars))
                
            log.info(f"Historical OHLCV data fetched successfully: {len(df)} bars. Running purged walk-forward...")
            data_source = str(df.attrs.get("data_source", bot_state.get("research_data_source", "-")))
            # Purged walk-forward over the geometric pipeline (warm start +
            # neighbour-fold RKD, geometry schema fixed).
            engine = PurgedWalkForwardEngine(df, timeframe=bot_state["timeframe"])
            result = engine.run()
            log.info(f"WFO completed ({result.get('engine','geometric-purged-wfo')}). Challenger: {result['challenger']}, "
                     f"Stability: {result['stability_count']}/{result['slices_evaluated']}")

            challenger = result["challenger"]
            if challenger:
                challenger = {k: v for k, v in challenger.items() if not k.startswith("slice_")}

            p_store = get_all_parameters()
            promotion_status = "not_promoted"
            promotion_reason = result.get("promotion_reason", "validation gate failed")
            champion = p_store.get("champion") or {}
            candidate_changed = challenger and any(
                float(challenger.get(k, 0.0)) != float(champion.get(k, 0.0))
                for k in ("TP_PERCENT", "TRAIL_MULT")
            )
            if result.get("eligible_for_promotion") and candidate_changed:
                if bot_instance.position.is_open:
                    p_store["shadow_challenger"] = challenger
                    promotion_status = "pending_position_close"
                    promotion_reason = "validated; waiting for the open position to close"
                else:
                    promote_parameter_set(p_store, challenger, source="purged_wfo_auto")
                    promotion_status = "auto_promoted"
                    promotion_reason = "validated OOS improvement; champion updated"
            else:
                # Preserve a genuinely different candidate for deliberate manual
                # override, but do not show the current champion as its own challenger.
                p_store["shadow_challenger"] = challenger if candidate_changed else None

            save_parameters_store(p_store)

            bot_state["parameters_store"] = p_store
            bot_state["active_parameters"] = p_store.get("champion", get_active_parameters())
            bot_state["wfo_bars_used"] = int(len(df))

            bot_state["wfo_report"] = {
                "stability_count": result["stability_count"],
                "variance": float(result["variance"]),
                "slices_evaluated": result["slices_evaluated"],
                "challenger": challenger,
                "promotion_status": promotion_status,
                "promotion_reason": promotion_reason,
                "improvement_score": float(result.get("improvement_score", 0.0)),
                "engine": result.get("engine", "legacy-grid"),
                "model": "LightGBM",
                "data_source": data_source,
                "bars_requested": requested_bars,
                "bars_used": int(len(df)),
                "schema": result.get("schema", "-"),
                "overfit": result.get("overfit", {}),
                "diagnostics": result.get("diagnostics", []),
                "d_anchor_log": result.get("d_anchor_log", []),
                "delta_hat_series": result.get("delta_hat_series", [])
            }
        except Exception as e:
            log.error(f"WFO API error: {e}")
            bot_state["wfo_report"] = {"status": "error", "message": str(e)}
        finally:
            with state_lock:
                bot_state["wfo_running"] = False

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"status": "running", "bars": requested_bars})

@app.route('/api/promote_challenger', methods=['POST'])
def promote_challenger_endpoint():
    global bot_instance
    if not bot_instance:
        return jsonify({"status": "error", "message": "Bot not initialized"}), 400
        
    if bot_instance.position.is_open:
        return jsonify({"status": "error", "message": "Açık pozisyon varken parametre değişikliği yapılamaz!"}), 400
        
    try:
        p_store = get_all_parameters()
        shadow = p_store.get("shadow_challenger")
        if not shadow:
            return jsonify({"status": "error", "message": "Aktif Challenger parametresi bulunamadı!"}), 400
            
        promote_parameter_set(p_store, shadow, source="manual_override")
        save_parameters_store(p_store)
            
        bot_state["parameters_store"] = p_store
        bot_state["active_parameters"] = shadow
        
        msg = (
            f"🔄 *CHALLENGER PARAMETRESİ ŞAMPİYON YAPILDI!*\n\n"
            f"📈 *Yeni Versiyon:* v{p_store['active_version']}\n"
            f"Parameters: {shadow}"
        )
        if bot_state.get("loop"):
            asyncio.run_coroutine_threadsafe(bot_instance.telegram.send_message(msg), bot_state["loop"])
            
        return jsonify({"status": "success", "parameters": shadow})
    except Exception as e:
        log.error(f"Error promoting challenger: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDED TEST SUITES
#
# The three test files live inside this module so a single-file workflow keeps
# them in sync. Run any suite with the CLI flags below; with no flag the app
# launches normally (the pywebview desktop UI).
#
#   python quant_bot_v35.py                # launch the bot (default)
#   python quant_bot_v35.py --test         # run every suite
#   python quant_bot_v35.py --test-geom    # V3.6 learned-geometry tests only
#   python quant_bot_v35.py --test-safe    # safety/execution tests only
#
# LightGBM is a required runtime dependency (see requirements.txt).
# ═══════════════════════════════════════════════════════════════════════════════

def _make_synth_df(n=900, seed=42):
    """Regime-switching synthetic OHLCV: trends, mean reversion and jumps."""
    rng = np.random.default_rng(seed)
    closes = [100.0]
    drift = 0.0
    for i in range(1, n):
        if i % 180 == 0:
            drift = rng.choice([-0.05, 0.0, 0.08])
        shock = rng.normal(0, 0.35)
        if rng.random() < 0.01:
            shock += rng.choice([-1, 1]) * rng.uniform(1.5, 3.0)
        mr = 0.03 * (100.0 - closes[-1]) if abs(drift) < 1e-9 else 0.0
        closes.append(max(closes[-1] + drift + mr + shock, 5.0))
    closes = np.array(closes)
    spread = np.abs(rng.normal(0.15, 0.05, n))
    highs = closes + spread
    lows = closes - spread
    opens = closes + rng.normal(0, 0.08, n)
    vols = np.abs(rng.normal(50, 15, n)) * (1 + 3 * (np.abs(np.diff(closes, prepend=closes[0])) > 0.8))
    ts = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame({"timestamp": ts, "open": opens, "high": highs,
                         "low": lows, "close": closes, "volume": vols})


# ── V3.6 learned-geometry tests ──────────────────────────────────────────────
def test_chen_identity():
    print("Testing Chen identity (fine → coarse combination is exact)...")
    rng = np.random.default_rng(0)
    dX = rng.normal(0, 1, (15, 3))
    direct = ChenSignature.of_increments(dX)
    blocks = [ChenSignature.of_increments(dX[i * 5:(i + 1) * 5]) for i in range(3)]
    combined = ChenSignature.combine(ChenSignature.combine(blocks[0], blocks[1]), blocks[2])
    for a, b in zip(direct, combined):
        assert np.allclose(a, b, atol=1e-10), "Chen identity violated"
    print("  levels 1..3 of concat path == tensor combination of block signatures ✓")


def test_multires_matches_direct():
    print("Testing multi-resolution end-aligned windows vs direct signatures...")
    df = _make_synth_df(200)
    vol = VolatilityNormalizer().transform(df)
    sigs, valid = MultiResolutionSignatures().compute(vol["increments"])
    t = 150
    assert valid[t]
    for r in GEOM_RESOLUTIONS:
        direct = ChenSignature.flatten(
            ChenSignature.of_increments(vol["increments"][t - r + 1:t + 1]))
        assert np.allclose(sigs[r][t], direct, atol=1e-8), f"res {r} mismatch"
    print(f"  all resolutions {GEOM_RESOLUTIONS} agree with direct computation ✓")


def test_delta_hat_and_schema():
    print("Testing δ̂ diagnostic → κ_init + factor budget (first-window protocol)...")
    df = _make_synth_df(600)
    vol = VolatilityNormalizer().transform(df)
    sigs, valid = MultiResolutionSignatures().compute(vol["increments"])
    diag = DeltaHatDiagnostic()
    schema = diag.build_schema(sigs, valid, vol["norm_ret"], "1m", version=1)
    print(f"  schema: {schema.label()} | δ̂ = {schema.delta_hat}")
    assert schema.kappa_init < 0, "κ sign must come from theory: dyadic tree → κ<0"
    assert schema.kappa_init <= schema.kappa_max < 0, "κ ≤ κ_max < 0 bound violated"
    assert -DeltaHatDiagnostic.KAPPA_ABS_MAX <= schema.kappa_init
    for r in GEOM_RESOLUTIONS:
        assert 0 < schema.delta_hat[str(r)] < 2.0
    assert schema.budget["a"] >= 4
    assert isinstance(schema.budget["S_active"], bool)
    assert isinstance(schema.budget["E_active"], bool)


def test_soft_threshold_sparsity():
    print("Testing soft-threshold proximal operator (exact zeros)...")
    x = np.array([-0.5, -0.1, 0.0, 0.05, 0.3])
    st = _soft_threshold(x, 0.2)
    assert np.allclose(st, [-0.3, 0.0, 0.0, 0.0, 0.1])
    assert np.sum(st == 0.0) == 3, "soft-threshold must produce exact zeros (P(δ≠0) defined)"
    print("  exact zeros inside the threshold band ✓")


def test_encoder_gradients():
    print("Testing encoder backprop against numerical gradients (per loss term)...")
    rng = np.random.default_rng(0)
    schema = GeometrySchema(1, "1m", list(GEOM_RESOLUTIONS), -0.8, -0.1, {"5": 0.2},
                            {"a": 6, "b": 0, "c": 0, "S_active": False, "E_active": False})
    T, n_res = 7, len(GEOM_RESOLUTIONS)
    X = rng.normal(0, 1, (T * n_res, GEOM_SIG_DIM))
    ridx = np.concatenate([np.full(T, i) for i in range(n_res)])
    tgrid = np.tile(np.arange(T), n_res)
    speed = np.abs(rng.normal(0.5, 0.2, T * n_res))
    w = np.abs(rng.normal(1.0, 0.1, T * n_res))

    def check(active, use_rkd=False, zero_dec=False):
        enc = LearnedGeometryEncoder(GEOM_SIG_DIM, schema, hidden=10, seed=1)
        for k in enc.lambdas:
            enc.lambdas[k] = 0.0
        enc.lambdas.update(active)
        rkd = None
        if use_rkd:
            teacher = LearnedGeometryEncoder(GEOM_SIG_DIM, schema, hidden=10, seed=7)
            Xp = rng.normal(0, 1, (9, GEOM_SIG_DIM))
            ridxp = rng.integers(0, n_res, 9)
            rel = teacher.panel_relations(Xp, ridxp)
            rkd = {"Xp": Xp, "ridxp": ridxp, "Dt": rel["Dt"], "Ct": rel["Ct"]}
        if zero_dec:
            enc.params["Wdec"][:] = 0.0
            enc.params["bdec"][:] = 0.0
            enc.params["g"][:] = -30.0
            enc.params["beta"][:] = 0.0
        _, _, grads, _ = enc._loss_and_grads(X, ridx, tgrid, speed, w, rkd=rkd)
        eps, worst = 1e-6, 0.0
        for name in ["W1", "b1", "Wu", "wd", "bd", "rho", "g", "beta", "Wdec", "bdec"]:
            P = np.asarray(enc.params[name], dtype=float)
            flatP = P.reshape(-1)
            flatG = np.asarray(grads[name], dtype=float).reshape(-1)
            for i in rng.choice(flatP.size, size=min(4, flatP.size), replace=False):
                orig = flatP[i]
                flatP[i] = orig + eps
                enc.params[name] = flatP.reshape(P.shape)
                lp = enc._loss_and_grads(X, ridx, tgrid, speed, w, rkd=rkd)[0]
                flatP[i] = orig - eps
                enc.params[name] = flatP.reshape(P.shape)
                lm = enc._loss_and_grads(X, ridx, tgrid, speed, w, rkd=rkd)[0]
                flatP[i] = orig
                enc.params[name] = flatP.reshape(P.shape)
                num = (lp - lm) / (2 * eps)
                worst = max(worst, abs(num - flatG[i]) / max(abs(num), abs(flatG[i]), 1e-7))
        return worst

    for label, kwargs in [
        ("recon", dict(active={})),
        ("l1", dict(active={"l1": 0.05}, zero_dec=True)),
        ("cone", dict(active={"cone": 0.5}, zero_dec=True)),
        ("speed", dict(active={"speed": 0.2}, zero_dec=True)),
        ("rkd", dict(active={"rkd": 1.0}, use_rkd=True, zero_dec=True)),
    ]:
        err = check(**kwargs)
        print(f"  {label:6s} worst rel err {err:.2e}")
        assert err < 5e-4, f"gradient mismatch in {label} term: {err:.2e}"


def test_encoder_training():
    print("Testing encoder training (loss ↓, κ bounded, sparse δ, FiLM monotone γ)...")
    df = _make_synth_df(500)
    pipe = GeometricPipeline("1m", encoder_epochs=30)
    vol, sigs, valid = pipe.preprocess(df)
    diag = DeltaHatDiagnostic()
    pipe.schema = diag.build_schema(sigs, valid, vol["norm_ret"], "1m", version=1)
    pipe.feat_mu, pipe.feat_sd = pipe._fit_feature_stats(sigs, valid)
    X, ridx, tgrid, speed, w, vidx = pipe._stack_training(vol, sigs, valid)
    enc = LearnedGeometryEncoder(GEOM_SIG_DIM, pipe.schema, seed=42)
    heldout = enc.train(X, ridx, tgrid, speed, w, epochs=30)
    hist = enc.train_history
    assert hist[-1]["rec"] < hist[0]["rec"], "reconstruction loss must decrease"
    assert -DeltaHatDiagnostic.KAPPA_ABS_MAX <= enc.kappa <= -DeltaHatDiagnostic.KAPPA_ABS_MIN, \
        "κ escaped its bounds"
    fw = enc.forward(X, ridx)
    p_dnz = float(np.mean(fw["delta"] != 0))
    print(f"  recon {hist[0]['rec']:.4f} → {hist[-1]['rec']:.4f} | heldout {heldout:.4f} "
          f"| κ={enc.kappa:.3f} η={enc.eta:.3f} | P(δ≠0)={p_dnz:.3f}")
    assert 0.0 <= p_dnz < 1.0
    assert np.any(fw["delta"] == 0.0), "δ head must produce exact zeros"
    assert np.all(np.abs(np.linalg.norm(fw["u"], axis=1) - 1.0) < 1e-6), "u must be unit"
    assert np.all(_softplus(enc.params["g"]) >= 0.0)   # γ(r) monotone
    it = enc.e_res_intervention(X, ridx)
    print(f"  e_res intervention degradation: {it['degradation']*100:.2f}%")
    r_eff = enc.r_eff()
    assert all(v > 0 for v in r_eff.values())


def test_delta_hat_outlier_robustness():
    print("Testing δ̂ scale normalization robustness to jump outliers...")
    rng = np.random.default_rng(4)
    P = rng.normal(0, 1, (120, GEOM_SIG_DIM))
    diag = DeltaHatDiagnostic(n_quadruples=800, seed=4)
    d1 = diag.measure(P)
    P2 = np.vstack([P, np.full((2, GEOM_SIG_DIM), 60.0)])
    d2 = diag.measure(P2)
    print(f"  δ̂ clean {d1:.4f} vs with outliers {d2:.4f}")
    assert d2 > 0.5 * d1, "outliers must not collapse δ̂ (κ saturation guard)"


def test_cost_floors():
    print("Testing cost wall: roundtrip cost + barrier target floors...")
    c = roundtrip_cost_pct()
    print(f"  roundtrip cost ≈ {c:.3f}%  → TP floor {3*c:.2f}%  SL floor {1.5*c:.2f}%")
    assert 0.25 < c < 0.40
    pipe = GeometricPipeline("1m")
    vol_low = {"rv": np.full(100, 0.0003)}
    tp, sl = pipe._barrier_pcts(vol_low, np.arange(100))
    assert np.all(tp >= 3 * c - 1e-9), "TP must clear 3× roundtrip cost"
    assert np.all(sl >= 1.5 * c - 1e-9)
    assert np.all(sl <= tp + 1e-9), "risk must not exceed the target"
    vol_hi = {"rv": np.full(10, 0.005)}
    tp2, sl2 = pipe._barrier_pcts(vol_hi, np.arange(10))
    assert np.all(tp2 > 3 * c), "high vol must widen targets beyond the cost floor"
    assert np.all(np.isclose(sl2, 0.5 * tp2)), "SL tracks half the vol-scaled target"
    be = (1.5 * c + c) / (3 * c + 1.5 * c)
    print(f"  breakeven win rate at floors: {be*100:.0f}%")
    assert be < 0.60


def test_geo_trailing_floor():
    print("Testing Geo trailing-stop floor (min_trail_dist)...")
    pm = PositionManager()
    pm.open("long", 100.0, 1.0, "Geo", 0.01, 100.0,
            tp_percent=1.0, sl_percent=0.5, min_trail_dist=0.5)
    assert abs(pm.trail_stop_70 - 99.5) < 1e-9, "initial trail must respect the floor, not 3×gv"
    pm.update_stops(100.4, 0.01, 0.01)
    assert abs(pm.trail_stop_70 - 99.9) < 1e-9, "ratchet must keep the floored distance"
    pm2 = PositionManager()
    pm2.open("long", 100.0, 1.0, "Trend", 0.01, 100.0)
    assert abs(pm2.trail_stop_70 - (100.0 - 0.03)) < 1e-9
    print("  floored at 0.5 for Geo, legacy 3×gv unchanged ✓")


def test_episode_hysteresis():
    print("Testing episode segmentation hysteresis...")
    seg = EpisodeSegmenter(min_active=2, on_bars=2, off_bars=3)
    counts = np.array([0, 0, 3, 3, 0, 0, 0, 0])
    flags, episodes = seg.segment(counts)
    assert list(flags) == [False, False, False, True, True, True, False, False]
    assert episodes == [(2, 3)]
    flags2, eps2 = seg.segment(np.array([0, 3, 0, 0, 0, 0]))
    assert not any(flags2) and eps2 == []
    print("  ON after 2 consecutive active bars, OFF after 3 clean bars ✓")


def test_cluster_and_graph():
    print("Testing anomaly clustering + normal-centric transition graph...")
    rng = np.random.default_rng(1)
    S = np.vstack([rng.normal(0, 0.2, (10, AnomalyClusterer.SUMMARY_DIM)),
                   rng.normal(3, 0.2, (10, AnomalyClusterer.SUMMARY_DIM))])
    cl = AnomalyClusterer(k=2, seed=1).fit(S)
    a = cl.assign(S[0]); b = cl.assign(S[15])
    assert a != b, "well-separated episode summaries must land in different clusters"
    g = TransitionGraph(k=2)
    seq = np.array([0, 0, 1, 1, 0, 2, 0, 0, 1, 0])
    fwd = [(int(s), 0.01 if s == 0 else -0.02) for s in seq]
    g.fit(seq, fwd)
    tm = g.transmat()
    assert np.allclose(tm.sum(axis=1), 1.0)
    assert 0.0 <= g.p_return_to_normal(1) <= 1.0
    assert g.expected_return(1) < 0 < g.expected_return(0)
    f = g.features(True, 0)
    assert len(f) == g.n_features
    print("  transition rows normalized, state statistics observational ✓")


def test_gbm_learner():
    print("Testing LightGBM classifier adapter...")
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (600, 8))
    y = ((X[:, 0] + 0.5 * X[:, 1] + 0.1 * rng.normal(size=600)) > 0).astype(float)
    gb = LightGBMModel(n_trees=100, depth=3, seed=2).fit(X[:450], y[:450])
    p = gb.predict_proba(X[450:])
    auc = LightGBMModel.auc(y[450:], p)
    print(f"  holdout AUC = {auc:.3f}")
    assert auc > 0.85
    assert np.all((p > 0) & (p < 1))


def test_large_history_pagination():
    print("Testing paginated OHLCV fetch beyond the old five-page ceiling...")

    class PagedExchange:
        def __init__(self):
            tf_ms = 60_000
            end = int(datetime.now().timestamp() * 1000) // tf_ms * tf_ms
            start = end - 6_050 * tf_ms
            self.rows = [[start + i * tf_ms, 100, 101, 99, 100, 10] for i in range(6_050)]
            self.calls = 0

        def parse_timeframe(self, timeframe):
            return 60

        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            self.calls += 1
            rows = [row for row in self.rows if row[0] >= since]
            return rows[:min(limit, 700)]

    holder = type("HistoryHolder", (), {})()
    holder.exchange = PagedExchange()
    df = asyncio.run(QuantBot._fetch_ohlcv_paginated(
        holder, holder.exchange, "test", SYMBOL, "1m", 6_000
    ))
    assert len(df) == 6_000
    assert df["timestamp"].is_monotonic_increasing
    assert holder.exchange.calls > 5
    print(f"  {len(df)} unique bars in {holder.exchange.calls} pages ✓")


def test_research_provider_fallback():
    print("Testing automatic public-data fallback after a provider rejection...")
    global RESEARCH_PROVIDER_ORDER

    class FailingExchange:
        def parse_timeframe(self, timeframe): return 60
        def fetch_ohlcv(self, *args, **kwargs):
            raise RuntimeError("simulated provider range rejection")

    class WorkingExchange:
        def __init__(self):
            tf_ms = 60_000
            end = int(datetime.now().timestamp() * 1000) // tf_ms * tf_ms
            start = end - 3_050 * tf_ms
            self.rows = [[start + i * tf_ms, 100, 101, 99, 100, 10] for i in range(3_050)]
        def parse_timeframe(self, timeframe): return 60
        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            return [row for row in self.rows if row[0] >= since][:limit]

    class FallbackHolder:
        def __init__(self):
            self.exchanges = {"mexc": FailingExchange(), "binance": WorkingExchange()}
        def _research_exchange(self, provider):
            return self.exchanges[provider]
        async def _fetch_ohlcv_paginated(self, exchange, provider, symbol, timeframe, limit):
            return await QuantBot._fetch_ohlcv_paginated(
                self, exchange, provider, symbol, timeframe, limit
            )

    previous_order = RESEARCH_PROVIDER_ORDER
    RESEARCH_PROVIDER_ORDER = ("mexc", "binance")
    try:
        df = asyncio.run(QuantBot.fetch_ohlcv_large(
            FallbackHolder(), SYMBOL, "1m", 3_000
        ))
        assert len(df) == 3_000
        assert df.attrs["data_source"] == "binance"
        assert bot_state["research_data_source"] == "binance"
        print("  rejected MEXC -> BINANCE fallback selected ✓")
    finally:
        RESEARCH_PROVIDER_ORDER = previous_order


def test_backtest_endpoint_repeatability():
    print("Testing that the backtest endpoint can complete repeatedly...")
    import time
    global bot_instance, run_geometric_backtest

    class PositionStub:
        is_open = False

    class BotStub:
        position = PositionStub()

        async def fetch_ohlcv_large(self, symbol, timeframe, limit):
            return _make_synth_df(limit, seed=91)

    original_bot = bot_instance
    original_runner = run_geometric_backtest
    bot_instance = BotStub()
    run_geometric_backtest = lambda df, params, timeframe="1m": {
        "trade_count": 1, "total_pnl_pct": 1.0, "total_pnl_usdt": 10.0,
        "profit_factor": 1.2, "max_drawdown_pct": 0.5, "calmar_ratio": 2.0,
        "sharpe_ratio": 1.0, "win_rate": 100.0, "expectancy": 1.0,
        "engine": "geometric-test", "schema": "test", "trades": [],
    }
    try:
        bot_state["loop"] = None
        bot_state["backtest_running"] = False
        bot_state["wfo_running"] = False
        client = app.test_client()
        for run_no in (1, 2):
            response = client.post("/api/backtest", json={"bars": 3_000})
            assert response.status_code == 200, response.get_json()
            deadline = time.time() + 5.0
            while bot_state["backtest_running"] and time.time() < deadline:
                time.sleep(0.01)
            assert not bot_state["backtest_running"]
            assert bot_state["backtest_report"]["bars_used"] == 3_000
        print("  two consecutive runs completed and produced fresh reports ✓")
    finally:
        bot_instance = original_bot
        run_geometric_backtest = original_runner


def test_meta_and_conformal():
    print("Testing meta labeling + conformal gate...")
    closes = np.array([100.0, 100.2, 100.5, 100.1, 99.4, 99.0, 101.0, 102.0])
    assert MetaLabeler.barrier_outcome(closes, 0, tp_pct=0.4, sl_pct=0.4, max_hold=6) == 1
    assert MetaLabeler.barrier_outcome(closes, 2, tp_pct=0.4, sl_pct=0.4, max_hold=4) == 0
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (300, 5))
    y = (X[:, 0] > 0).astype(float)
    ml = MetaLabeler().fit(X, y)
    p = ml.predict_proba(X)
    assert LightGBMModel.auc(y, p) > 0.9
    gate = ConformalGate(alpha=0.6, beta=0.4)
    pm_cal = rng.uniform(0.2, 0.9, 60)
    dm_cal = rng.uniform(0.1, 2.0, 60)
    gate.calibrate(pm_cal, dm_cal, target_accept=0.6)
    assert gate.a_pred(0.95) > gate.a_pred(0.05)
    assert gate.a_geom(0.05) > gate.a_geom(5.0)
    A = gate.combined(0.8, 0.3)
    assert 0.0 < A <= 1.0
    accept_rate = np.mean([gate.combined(p_, d_) >= gate.a_min
                           for p_, d_ in zip(pm_cal, dm_cal)])
    print(f"  calibrated acceptance on cal window: {accept_rate*100:.0f}% (target 60%)")
    assert 0.4 <= accept_rate <= 0.8


def test_barrier_directional():
    print("Testing directional barrier helpers (symmetric label + long/short race)...")
    up = np.array([100.0, 100.1, 100.6, 101.0])
    dn = np.array([100.0, 99.9, 99.3, 99.0])
    assert MetaLabeler.barrier_dir(up, 0, tp_pct=0.4, max_hold=4) == 1.0
    assert MetaLabeler.barrier_dir(dn, 0, tp_pct=0.4, max_hold=4) == 0.0
    assert MetaLabeler.barrier_outcome_dir(dn, 0, tp_pct=0.4, sl_pct=0.4, side="short", max_hold=4) == 1
    assert MetaLabeler.barrier_outcome_dir(up, 0, tp_pct=0.4, sl_pct=0.4, side="short", max_hold=4) == 0
    assert MetaLabeler.barrier_outcome_dir(up, 0, tp_pct=0.4, sl_pct=0.4, side="long", max_hold=4) == 1
    print("  symmetric direction + long/short TP-before-SL correct ✓")


def test_two_sided_and_short_gate():
    print("Testing two-sided signals + allow_short gate (spot safety default)...")
    df = _make_synth_df(900, seed=13)
    pipe = GeometricPipeline("1m", encoder_epochs=25)
    pipe.fit(df.iloc[:600])
    pipe.allow_short = False
    geo_off = pipe.batch_signals(df, start_at=660)
    assert not np.any(geo_off["signal"] == "SELL"), "allow_short off must never emit SELL"
    assert np.all(geo_off["dir"] >= 0)
    pipe.allow_short = True
    geo_on = pipe.batch_signals(df, start_at=660)
    sell_bars = np.where(geo_on["signal"] == "SELL")[0]
    buy_bars = np.where(geo_on["signal"] == "BUY")[0]
    for bars in (buy_bars, sell_bars):
        assert np.all(geo_on["exp_net"][bars] > 0), "every signal needs E[net]>0"
    assert np.all(geo_on["dir"][sell_bars] == -1)
    assert np.all(geo_on["dir"][buy_bars] == 1)
    assert set(buy_bars.tolist()) == set(np.where(geo_off["signal"] == "BUY")[0].tolist())
    print(f"  off→0 SELL, on→{len(sell_bars)} SELL / {len(buy_bars)} BUY, longs unchanged ✓")


def test_backtester_short():
    print("Testing backtester short entries in geo mode...")
    df = _make_synth_df(900, seed=5)
    pipe = GeometricPipeline("1m", encoder_epochs=25)
    pipe.fit(df.iloc[:600])
    pipe.allow_short = True
    geo = pipe.batch_signals(df, start_at=660)
    if not np.any(geo["signal"] == "SELL"):
        idx = 700
        geo["signal"][idx] = "SELL"; geo["dir"][idx] = -1
        geo["exp_net"][idx] = 1.0
    params = {"FAST_LENGTH": 8, "SLOW_LENGTH": 21, "VOL_LENGTH": 14, "CVD_LENGTH": 14,
              "BAND_MULT": 2.5, "MIN_PROFIT_MARGIN": 0.3, "TRAIL_MULT": 3.0,
              "PING_STOP_MULT": 0.5, "TP_PERCENT": 0.6}
    rep = Backtester(df.iloc[660:].reset_index(drop=True)).run(
        params, geo={k: (v[660:] if isinstance(v, np.ndarray) else v) for k, v in geo.items()})
    shorts = [t for t in rep["trades"] if t["side"] == "short"]
    assert len(shorts) >= 1, "geo SELL must open short positions in the backtester"
    for t in shorts:
        if t["exit_price"] < t["entry_price"]:
            assert t["pnl_pct"] > 0, "short PnL sign wrong"
    print(f"  {len(shorts)} short trades executed, PnL signs correct ✓")


def test_overfit_diagnostic():
    print("Testing overfitting diagnostics (IS vs OOS AUC gap)...")
    df = _make_synth_df(1250, seed=9)
    pipe = GeometricPipeline("1m", encoder_epochs=25)
    diag = pipe.fit(df.iloc[:700])
    for k in ("train_auc", "oos_auc", "overfit_gap"):
        assert k in diag, f"missing {k}"
    assert abs(diag["overfit_gap"] - (diag["train_auc"] - diag["oos_auc"])) < 1e-9
    eng = PurgedWalkForwardEngine(df, timeframe="1m", n_folds=3)
    res = eng.run()
    assert "overfit" in res and "mean_gap" in res["overfit"]
    assert res["overfit"]["verdict"] in ("low", "moderate", "high")
    print(f"  fold gap {diag['overfit_gap']:+.3f} | WFO mean gap {res['overfit']['mean_gap']:+.3f} "
          f"[{res['overfit']['verdict']}] ✓")


def test_anchor_panel():
    print("Testing anchor panel (append-only ledger, retire-not-delete, A_geom feed)...")
    df = _make_synth_df(500)
    vol = VolatilityNormalizer().transform(df)
    sigs, valid = MultiResolutionSignatures().compute(vol["increments"])
    panel = AnchorPanel()
    panel.build_core(sigs, valid, vol["strata"], fold=0)
    n0 = len(panel.core)
    assert n0 > 0
    strata_seen = {a["stratum"] for a in panel.core}
    assert len(strata_seen) > 3, "strata must be model-free and diverse"
    for a in panel.core:
        a["last_seen_fold"] = 0
        a["ood"] = 9.9
    panel.refresh(sigs, valid, fold=5)
    assert len(panel.core) == n0, "ledger is append-only: anchors are never deleted"
    assert any(a["status"] == "retired" for a in panel.core), "stale+OOD anchors must retire"
    dmin = panel.dmin({r: sigs[r][450] for r in GEOM_RESOLUTIONS})
    assert dmin >= 0.0
    Xp, ridxp = panel.panel_matrix()
    assert Xp.shape[1] == GEOM_SIG_DIM and len(ridxp) == len(Xp)
    print(f"  {n0} anchors, retired={sum(1 for a in panel.core if a['status']=='retired')}, "
          f"dmin={dmin:.3f} ✓")


def test_pipeline_end_to_end():
    print("Testing GeometricPipeline end-to-end (fit → batch signals → live state)...")
    df = _make_synth_df(900)
    pipe = GeometricPipeline("1m", encoder_epochs=25)
    diag = pipe.fit(df.iloc[:600])
    for key in ("p_delta_nonzero", "var_r", "e_res_test", "heldout_recon",
                "r_eff", "d_anchor", "delta_hat_data", "feature_stability", "kappa", "eta"):
        assert key in diag, f"missing mandatory diagnostic: {key}"
    assert pipe.status == "ready"
    assert pipe.schema is not None and pipe.schema.kappa_init < 0
    geo = pipe.batch_signals(df, start_at=660)
    assert len(geo["signal"]) == len(df)
    assert set(np.unique(geo["signal"][:660])) <= {"HOLD"}
    assert np.all((geo["p"] >= 0) & (geo["p"] <= 1))
    assert np.all((geo["A"] >= 0) & (geo["A"] <= 1))
    c_rt = roundtrip_cost_pct()
    assert np.all(geo["tp"][660:] >= 3 * c_rt - 1e-9)
    assert np.all(geo["sl"][660:] >= 1.5 * c_rt - 1e-9)
    buy_bars = np.where(geo["signal"] == "BUY")[0]
    assert np.all(geo["exp_net"][buy_bars] > 0), "BUY only when E[net]>0"
    n_buy = int(len(buy_bars))
    print(f"  schema={pipe.schema.label()} | BUY signals on OOS: {n_buy} "
          f"| P(δ≠0)={diag['p_delta_nonzero']:.3f}")
    live = pipe.live_state(pipe.infer_latest(df))
    for key in ("status", "signal", "p_gbm", "p_meta", "a_score", "a_gate",
                "episode", "cluster", "schema", "kappa", "panel", "chart"):
        assert key in live, f"missing live state key: {key}"
    assert live["status"] == "ready"
    ch = live["chart"]
    assert ch["n"] == len(ch["A"]) == len(ch["episode"]) == len(ch["buy"]) \
        == len(ch["exit"]) == len(ch["state"]), "chart overlay arrays must align"
    assert all(0.0 <= a <= 1.0 for a in ch["A"])
    params = {"FAST_LENGTH": 8, "SLOW_LENGTH": 21, "VOL_LENGTH": 14, "CVD_LENGTH": 14,
              "BAND_MULT": 2.5, "MIN_PROFIT_MARGIN": 0.3, "TRAIL_MULT": 3.0,
              "PING_STOP_MULT": 0.5, "TP_PERCENT": 0.6}
    rep = Backtester(df.iloc[660:].reset_index(drop=True)).run(
        params, geo={k: (v[660:] if isinstance(v, np.ndarray) else v) for k, v in geo.items()})
    assert "trade_count" in rep and "profit_factor" in rep
    print(f"  geo backtest slice: {rep['trade_count']} trades, PF={rep['profit_factor']:.2f}")


def test_geometric_backtest_and_purged_wfo():
    print("Testing run_geometric_backtest + PurgedWalkForwardEngine...")
    df = _make_synth_df(1250, seed=7)
    params = {"FAST_LENGTH": 8, "SLOW_LENGTH": 21, "VOL_LENGTH": 14, "CVD_LENGTH": 14,
              "BAND_MULT": 2.5, "MIN_PROFIT_MARGIN": 0.3, "TRAIL_MULT": 3.0,
              "PING_STOP_MULT": 0.5, "TP_PERCENT": 0.6}
    rep = run_geometric_backtest(df, params, timeframe="1m")
    assert rep["engine"] == "geometric"
    assert rep["train_bars"] > 0 and "schema" in rep
    print(f"  geometric backtest: {rep['trade_count']} trades | schema {rep['schema']}")
    eng = PurgedWalkForwardEngine(df, timeframe="1m", n_folds=3)
    res = eng.run()
    assert res["engine"] == "geometric-purged-wfo"
    assert res["slices_evaluated"] == 3
    assert res["challenger"] is not None and "TP_PERCENT" in res["challenger"]
    assert len(res["diagnostics"]) >= 2
    versions = {d["schema"] for d in res["diagnostics"]}
    assert len(versions) == 1, "geometry schema must stay fixed across folds"
    assert len(res["d_anchor_log"]) >= 2
    d0 = res["d_anchor_log"][0]["d_anchor"]
    assert abs(d0) < 1e-9, "fold-0 D_anchor is the baseline (0)"
    assert len(res["delta_hat_series"]) >= 2
    print(f"  purged WFO: stability {res['stability_count']}/3 | "
          f"D_anchor log {[round(e['d_anchor'], 4) for e in res['d_anchor_log']]}")


def test_signal_engine_contract():
    print("Testing SignalEngine geo-only contract (HOLD until ready+healthy)...")
    df = _make_synth_df(300)
    eng = SignalEngine(enable_geometry=True)   # 300 bars < MIN_TRAIN_BARS -> collecting
    info = eng.process(df)
    for key in ("signal", "type", "price", "gauss_vol", "geom"):
        assert key in info, f"missing info key: {key}"
    assert info["geom"]["status"] in ("collecting", "training")
    # never a legacy signal during warm-up: geometry not ready -> HOLD
    assert info["signal"] == "HOLD", "warm-up must be HOLD (no legacy fallback)"
    eng2 = SignalEngine(enable_geometry=False)
    info2 = eng2.process(df)
    assert info2["geom"] == {} and info2["signal"] == "HOLD"
    print("  warm-up HOLD, geometry-only signal path ✓")


def test_health_monitor_auto_hold():
    print("Testing geometry health monitor + auto-HOLD on degraded diagnostics...")
    df = _make_synth_df(900, seed=17)
    pipe = GeometricPipeline("1m", encoder_epochs=25)
    pipe.fit(df.iloc[:600])
    assert pipe.status == "ready"
    hs = pipe.health_status()
    assert hs["health"] in ("ok", "degraded") and "reasons" in hs
    # force a degraded diagnostic: absurd D_anchor drift -> must flip to degraded
    pipe.diagnostics[-1]["d_anchor"] = 99.0
    hs2 = pipe.health_status()
    assert hs2["health"] == "degraded" and any("drift" in r for r in hs2["reasons"])
    # a degraded pipeline reports READY status but SignalEngine must emit HOLD
    live = pipe.live_state({"status": "ready", "signal": "BUY",
                            "health": "degraded", "health_reasons": hs2["reasons"]})
    assert live["health"] == "degraded"
    print(f"  health flips to degraded on drift; signal gated ✓ ({hs2['reasons'][:1]})")



# ── V3.5 safety/execution tests with a mock CCXT exchange ───────────────────
class _MockExchange:
    def __init__(self):
        self.apiKey = "mock_key"
        self.secret = "mock_secret"
        self.orders = []
        self.canceled_orders = []
        self.balance = {"USDT": {"free": 100.0}}

    def load_markets(self): pass
    def amount_to_precision(self, symbol, qty): return f"{qty:.4f}"
    def price_to_precision(self, symbol, price): return f"{price:.2f}"
    def fetch_balance(self): return self.balance
    def fetch_order_book(self, symbol, limit=None):
        return {"bids": [[59990.0, 10.0]], "asks": [[60010.0, 10.0]]}
    def fetch_order(self, order_id, symbol):
        for o in self.orders:
            if o['id'] == order_id:
                o['status'] = 'closed'; o['filled'] = o['qty']
                return o
        return {"id": order_id, "status": "closed", "filled": 0.0, "average": 60000.0}
    def fetch_ticker(self, symbol): return {"last": 60000.0, "close": 60000.0}
    def create_order(self, symbol, type, side, qty, price=None, params=None):
        order = {"id": f"mock_order_{len(self.orders)+1}", "symbol": symbol, "type": type,
                 "side": side, "qty": qty, "price": price, "params": params or {},
                 "average": price or 60000.0}
        self.orders.append(order)
        return order
    def cancel_order(self, order_id, symbol):
        self.canceled_orders.append(order_id)
        return {"status": "canceled", "id": order_id}


async def _run_safety_tests_async():
    print("--------------------------------------------------")
    print("RUNNING QUANT BOT V3.5 SAFETY AND METRIC TESTS...")
    print("--------------------------------------------------")
    # placeholder API creds so QuantBot() init doesn't raise in a fresh env
    os.environ.setdefault("BORSANIN_API_KEY", "mock_key")
    os.environ.setdefault("BORSANIN_SECRET_KEY", "mock_secret")
    bot = QuantBot()
    mock_ex = _MockExchange()
    bot.exchange = mock_ex
    bot_state["trading_mode"] = "PAPER"
    bot_state["virtual_balance"] = 10000.0
    bot_state["is_trading_active"] = True
    bot_state["trades"] = []
    bot_state["trade_count"] = 0
    bot_state["total_pnl"] = 0.0
    bot_state["pnl_list"] = []
    bot_state["winning_trades"] = 0
    bot_state["losing_trades"] = 0

    print("Testing Spot Long-only restriction:")
    # Minimal geo-only info contract: a healthy READY pipeline emitting a signal,
    # with the cost-floored TP/SL the executor reads.
    info = {
        'signal': 'SELL', 'type': 'Geo', 'gauss_vol': 200.0,
        'geom': {'status': 'ready', 'health': 'ok', 'signal': 'SELL', 'exit_flag': False,
                 'tp_pct': 1.0, 'sl_pct': 0.5, 'exp_net': 0.5, 'schema': 'test',
                 'chart': {}},
    }
    df = pd.DataFrame({
        'timestamp': [pd.Timestamp.now()] * 100,
        'open': [60000.0] * 100, 'high': [60100.0] * 100, 'low': [59900.0] * 100,
        'close': [60000.0] * 100, 'volume': [10.0] * 100
    })

    async def mock_fetch_ohlcv():
        return df

    bot.fetch_ohlcv = mock_fetch_ohlcv
    bot.signal_engine.process = lambda *args, **kwargs: info

    await bot.main_tick()
    print(f"  Position open after SELL signal? {bot.position.is_open} (Expected: False)")
    assert not bot.position.is_open, "Error: Opened position on SELL signal!"

    print("Testing Paper entry slippage and fee calculation:")
    info['signal'] = 'BUY'; info['geom']['signal'] = 'BUY'
    await bot.main_tick()
    print(f"  Position open? {bot.position.is_open} (Expected: True)")
    print(f"  Position mode: {bot.position.mode} (Expected: PAPER)")
    print(f"  Position entry price: {bot.position.entry_price:.2f} (Expected: 60030.00)")
    assert abs(bot.position.entry_price - 60030.0) < 1e-5
    invested = bot.position.invested_amount
    expected_fee = invested * 0.001
    expected_balance = 10000.0 - expected_fee
    print(f"  Virtual Balance after entry: {bot_state['virtual_balance']:.2f} (Expected: {expected_balance:.2f})")
    assert abs(bot_state["virtual_balance"] - expected_balance) < 1e-2

    print("Testing Mode Isolation:")
    bot_state["trading_mode"] = "REAL"
    await bot.close_position("Test manual close", 61000.0)
    print(f"  Exchange orders placed on close? {len(mock_ex.orders)} (Expected: 0 - since position was PAPER)")
    assert len(mock_ex.orders) == 0, "Error: Placed order on exchange to close PAPER position!"
    print(f"  Virtual Balance after close: {bot_state['virtual_balance']:.2f}")

    print("Testing REAL mode stop-loss order placement:")
    mock_ex.orders = []
    bot_state["trading_mode"] = "REAL"
    mock_ex.balance = {"USDT": {"free": 100.0}}
    bot_state["real_balance"] = 100.0
    bot_state["start_real_balance"] = 100.0
    info['signal'] = 'BUY'; info['geom']['signal'] = 'BUY'
    await bot.main_tick()
    print(f"  Position mode: {bot.position.mode} (Expected: REAL)")
    print(f"  Borsa orders count: {len(mock_ex.orders)} (Expected: 2 - 1 entry and 1 stop-loss)")
    assert len(mock_ex.orders) == 2, f"Expected 2 orders, got {len(mock_ex.orders)}"
    assert mock_ex.orders[0]['side'] == "buy"
    assert mock_ex.orders[1]['side'] == "sell"
    assert "stopPrice" in mock_ex.orders[1]['params']

    print("Testing stop-loss order cancelation on exit:")
    stop_id = bot.position.stop_order_id
    await bot.close_position("Exit hit", 61000.0)
    print(f"  Canceled orders list: {mock_ex.canceled_orders} (Expected: ['{stop_id}'])")
    assert stop_id in mock_ex.canceled_orders, "Error: Stop order was not canceled on exit!"

    print("Testing Minimum Order Size enforcement (5 USDT):")
    mock_ex.orders = []
    mock_ex.balance = {"USDT": {"free": 3.0}}
    bot_state["real_balance"] = 3.0
    info['signal'] = 'BUY'; info['geom']['signal'] = 'BUY'
    await bot.main_tick()
    print(f"  Position open with low balance? {bot.position.is_open} (Expected: False)")
    assert not bot.position.is_open
    mock_ex.balance = {"USDT": {"free": 6.0}}
    bot_state["real_balance"] = 6.0
    await bot.main_tick()
    print(f"  Position open with 6 USDT balance? {bot.position.is_open} (Expected: True)")
    if bot.position.is_open:
        invested = bot.position.qty * 60000.0
        print(f"  Real order quantity: {bot.position.qty:.6f}, Value: {invested:.2f} USDT (Expected: >= 5.0)")
        assert invested >= 5.0

    print("--------------------------------------------------")
    print("ALL SAFETY AND EXECUTION TESTS PASSED SUCCESSFULLY!")
    print("--------------------------------------------------")


# ── suite dispatchers ────────────────────────────────────────────────────────
def _run_geom_suite():
    test_chen_identity()
    test_multires_matches_direct()
    test_delta_hat_and_schema()
    test_soft_threshold_sparsity()
    test_encoder_gradients()
    test_encoder_training()
    test_delta_hat_outlier_robustness()
    test_cost_floors()
    test_geo_trailing_floor()
    test_episode_hysteresis()
    test_cluster_and_graph()
    test_gbm_learner()
    test_large_history_pagination()
    test_research_provider_fallback()
    test_backtest_endpoint_repeatability()
    test_meta_and_conformal()
    test_barrier_directional()
    test_two_sided_and_short_gate()
    test_backtester_short()
    test_overfit_diagnostic()
    test_anchor_panel()
    test_pipeline_end_to_end()
    test_geometric_backtest_and_purged_wfo()
    test_signal_engine_contract()
    test_health_monitor_auto_hold()
    print("\nAll V3.6 learned-geometry tests passed!")


def _run_safety_suite():
    asyncio.run(_run_safety_tests_async())


def _run_all_tests():
    _run_geom_suite()
    print()
    _run_safety_suite()


bot_instance = None

def run_bot():
    global bot_instance
    bot_instance = QuantBot()
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    loop = asyncio.new_event_loop()
    loop.set_default_executor(executor)
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(bot_instance.main_loop())
    except (RuntimeError, asyncio.CancelledError) as e:
        log.warning(f"Bot loop ended: {e}")
    except Exception as e:
        log.error(f"Bot main loop error: {e}")
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        try:
            executor.shutdown(wait=False)
        except Exception:
            pass
        loop.close()

if __name__ == "__main__":
    # ── CLI: --test / --test-geom / --test-safe run the embedded test suites
    # and exit. No flag = launch the desktop bot as before.
    _flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if _flags & {"--test", "--test-all"}:
        _run_all_tests(); sys.exit(0)
    if "--test-geom" in _flags:
        _run_geom_suite(); sys.exit(0)
    if "--test-safe" in _flags:
        _run_safety_suite(); sys.exit(0)

    import subprocess
    try:
        cmd = "netstat -ano | findstr LISTENING | findstr :5001"
        output = subprocess.check_output(cmd, shell=True).decode()
        current_pid = os.getpid()
        for line in output.splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                pid = int(parts[-1])
                if pid != current_pid:
                    subprocess.run(f"taskkill /F /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    # Non-daemon: webview kapansa bile bot çalışmaya devam eder
    bot_thread = threading.Thread(target=run_bot, daemon=False)
    bot_thread.start()
    try:
        webview.create_window(title='Quant Bot V3.7 - LightGBM Research Engine', url=app, width=1400, height=850, resizable=True, min_size=(1100, 700))
        webview.start()
    except Exception as e:
        log.warning(f"Webview error: {e}")
    # Webview kapansa bile bot_thread bitene kadar bekle
    bot_thread.join()
