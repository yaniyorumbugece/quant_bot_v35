"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  QUANT BOT V3.6 — LEARNED GEOMETRY (nihai mimari)                          ║
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
from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template_string
import webview

try:
    from hmmlearn.hmm import GaussianHMM as HMMLearnGaussianHMM
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False

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
        logging.FileHandler(LOG_DIR / f"bot_v3.5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8")
    ]
)
log = logging.getLogger("QuantBot")

# ═══════════════════════════════════════════════════════════════════════════════
# AYARLAR & STATE
# ═══════════════════════════════════════════════════════════════════════════════
SYMBOL = "BTC/USDT"
OHLCV_LIMIT = 500
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
        "TP_PERCENT": 3.0
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

bot_state = {
    "is_trading_active": False,
    "trading_mode": "PAPER",
    "timeframe": "1m",
    "timeframe_changed": False,

    "symbol": SYMBOL,
    "price": 0.0,
    "fast_gauss": 0.0,
    "slow_gauss": 0.0,
    "upper_band": 0.0,
    "lower_band": 0.0,
    "gauss_vol": 0.0,
    "is_ranging": False,
    "regime": "Bilinmiyor",
    "signal": "HOLD",
    "signal_type": "",
    "obi": 0.0,

    "virtual_balance": 10000.0,
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
    "peak_balance": 10000.0,
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

    "hyp_direction": 0,
    "ou_theta": 0.0,
    "ou_mu": 0.0,
    "ou_half_life": 0.0,
    "ou_upper": 0.0,
    "ou_lower": 0.0,
    "ou_valid": False,
    "ou_jump_intensity": 0.0,
    "ou_jump_mean": 0.0,
    "ou_jump_std": 0.0,
    "ou_jump_detected": False,
    "ou_jump_cooldown": 0,
    
    # Shadow Mode keys
    "shadow_active": False,
    "shadow_parameters": {},
    "shadow_balance": 1000.0,
    "shadow_position_side": None,
    "shadow_position_entry": 0.0,
    "shadow_position_qty": 0.0,
    "shadow_position_pnl": 0.0,
    "shadow_total_pnl": 0.0,
    "shadow_trade_count": 0,
    "shadow_trades": [],
    
    # Backtest and WFO keys
    "wfo_report": None,
    "wfo_running": False,
    "backtest_report": None,
    "backtest_running": False,
    "active_parameters": get_active_parameters(),
    "parameters_store": get_all_parameters(),

    # V3.6 Learned Geometry state
    "geom": {"status": "collecting", "signal": "HOLD", "schema": "-", "kappa": 0.0,
             "a_score": 0.0, "a_gate": 0.5, "p_gbm": 0.5, "p_meta": 0.5,
             "episode": "NORMAL", "cluster": -1, "fold": 0, "dir": 0},

    # Two-sided trading toggles (opt-in short)
    "allow_short": ALLOW_SHORT,
    "trading_venue": TRADING_VENUE,
}

# ═══════════════════════════════════════════════════════════════════════════════
# METRİK MOTORU
# ═══════════════════════════════════════════════════════════════════════════════
def update_portfolio_metrics(pnl_usdt, mode):
    # Determine base capital for calculation
    if mode == "PAPER":
        base_capital = 10000.0
        current_balance = bot_state["virtual_balance"]
    else:
        current_balance = bot_state.get("real_balance", 0.0)
        base_capital = bot_state.get("start_real_balance", current_balance)
        if base_capital <= 0:
            base_capital = 10000.0

    # Calculate portfolio return % for this trade relative to capital base
    pnl_pct = (pnl_usdt / base_capital) * 100 if base_capital > 0 else 0.0

    bot_state["trade_count"] += 1
    bot_state["pnl_list"].append(pnl_pct)
    
    # Cumulative portfolio PnL % based on balance equity change
    if mode == "PAPER":
        bot_state["total_pnl"] = ((current_balance - 10000.0) / 10000.0) * 100
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
        bot_state["peak_balance"] = max(bot_state.get("peak_balance", 10000.0), current_balance)
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

class NumpyGaussianHMM:
    def __init__(self, n_components=2, n_iters=20, eps=1e-6):
        self.n_components, self.n_iters, self.eps = n_components, n_iters, eps
        self.means_ = self.covars_ = self.transmat_ = None

    def fit(self, X):
        T, D = X.shape; np.random.seed(42)
        # Initialize cluster means and covariances
        self.means_ = X[np.random.choice(T, self.n_components, replace=False)].copy()
        self.covars_ = np.array([np.eye(D) for _ in range(self.n_components)])
        self.transmat_ = np.ones((self.n_components, self.n_components)) / self.n_components
        
        # Iterative clustering updates (EM-like)
        Z = np.zeros(T, dtype=int)
        for _ in range(self.n_iters):
            ld = np.zeros((T, self.n_components))
            for j in range(self.n_components):
                diff = X - self.means_[j]
                _, logdet = np.linalg.slogdet(self.covars_[j])
                try: inv_cov = np.linalg.inv(self.covars_[j])
                except: inv_cov = np.eye(D)
                ld[:, j] = -0.5 * (D * np.log(2 * np.pi) + logdet + np.sum(diff @ inv_cov * diff, axis=1))
            Z = np.argmax(ld, axis=1)
            for i in range(self.n_components):
                Xi = X[Z == i]
                if len(Xi) > 1:
                    self.means_[i] = Xi.mean(0)
                    self.covars_[i] = np.cov(Xi.T, ddof=0) + self.eps * np.eye(D)
                    
        # Learn transition matrix from sequence Z
        self.transmat_ = np.zeros((self.n_components, self.n_components))
        for t in range(T - 1):
            self.transmat_[Z[t], Z[t+1]] += 1.0
        self.transmat_ += 1e-6  # Laplace smoothing
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)
        return self

    def predict(self, X):
        T_seq, D = X.shape
        # Log emissions for each observation and component
        log_emissions = np.zeros((T_seq, self.n_components))
        for j in range(self.n_components):
            diff = X - self.means_[j]
            _, logdet = np.linalg.slogdet(self.covars_[j])
            try: inv_cov = np.linalg.inv(self.covars_[j])
            except: inv_cov = np.eye(D)
            log_emissions[:, j] = -0.5 * (D * np.log(2 * np.pi) + logdet + np.sum(diff @ inv_cov * diff, axis=1))
            
        # Viterbi decoding algorithm
        log_V = np.zeros((T_seq, self.n_components))
        log_V[0] = log_emissions[0] + np.log(np.ones(self.n_components) / self.n_components)
        
        log_trans = np.log(self.transmat_ + 1e-12)
        for t in range(1, T_seq):
            for j in range(self.n_components):
                log_V[t, j] = log_emissions[t, j] + np.max(log_V[t-1] + log_trans[:, j])
                
        return int(np.argmax(log_V[-1]))

class HyperbolicClassifier:
    """
    Poincaré Ball Model hiperbolik uzay k-NN sınıflandırıcı.
    Feature'ları hiperbolik uzaya embed eder, hiperbolik mesafe ile k-NN sınıflandırma yapar.
    Trend sinyallerini doğrulamak için kullanılır.
    """
    def __init__(self, k=8, lookback=300):
        self.k = k
        self.lookback = lookback

    def _poincare_distance(self, u, v):
        """Poincaré Ball mesafesi: d(u,v) = arccosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))"""
        diff_sq = np.sum((u - v)**2)
        u_sq = np.sum(u**2)
        v_sq = np.sum(v**2)
        denom = (1.0 - u_sq) * (1.0 - v_sq)
        if denom <= 1e-10:
            return 100.0
        arg = 1.0 + 2.0 * diff_sq / denom
        return float(np.arccosh(max(arg, 1.0)))

    def _exp_map_origin(self, v, c=1.0):
        """Exponential map at origin: exp_0(v) = tanh(√c·||v||/2) · v / (√c·||v||)"""
        sqrt_c = math.sqrt(c)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            return v
        return np.tanh(sqrt_c * v_norm / 2.0) * v / (sqrt_c * v_norm)

    def _compute_features(self, closes, highs, lows):
        """Compute 5 features: RSI(14), CCI(20), ROC(9), Williams%R(14), Stochastic%K(14)"""
        n = len(closes)
        # RSI(14)
        rsi = np.full(n, 50.0)
        deltas = np.diff(closes, prepend=closes[0])
        for i in range(14, n):
            window = deltas[i-13:i+1]
            avg_gain = np.mean(np.maximum(window, 0))
            avg_loss = np.mean(np.maximum(-window, 0))
            rsi[i] = 100.0 - 100.0/(1.0 + avg_gain/avg_loss) if avg_loss > 0 else 100.0
        # CCI(20)
        cci = np.zeros(n)
        tp = (highs + lows + closes) / 3.0
        for i in range(20, n):
            w = tp[i-19:i+1]; sma = np.mean(w); mad = np.mean(np.abs(w - sma))
            cci[i] = (tp[i] - sma) / (0.015 * mad) if mad > 0 else 0.0
        # ROC(9)
        roc = np.zeros(n)
        for i in range(9, n):
            roc[i] = (closes[i] - closes[i-9]) / closes[i-9] * 100 if closes[i-9] > 0 else 0.0
        # Williams %R(14)
        wr = np.full(n, -50.0)
        for i in range(14, n):
            hh = np.max(highs[i-13:i+1]); ll = np.min(lows[i-13:i+1])
            wr[i] = -100*(hh - closes[i])/(hh - ll) if (hh - ll) > 0 else -50.0
        # Stochastic %K(14)
        stoch = np.full(n, 50.0)
        for i in range(14, n):
            hh = np.max(highs[i-13:i+1]); ll = np.min(lows[i-13:i+1])
            stoch[i] = 100*(closes[i] - ll)/(hh - ll) if (hh - ll) > 0 else 50.0
        return np.column_stack([rsi, cci, roc, wr, stoch])

    def classify(self, df, idx=-2):
        """
        Classify trend direction. Returns: 1 (bullish), -1 (bearish), 0 (neutral)
        """
        closes = df['close'].values; highs = df['high'].values; lows = df['low'].values
        n = len(closes)
        if n < 50: return 0
        features = self._compute_features(closes, highs, lows)
        start = max(0, n - self.lookback)
        wf = features[start:n]
        # Normalize to [-1, 1]
        mins = np.min(wf, axis=0); maxs = np.max(wf, axis=0)
        ranges = maxs - mins; ranges[ranges == 0] = 1.0
        normalized = 2 * (wf - mins) / ranges - 1
        # Embed into Poincaré ball via exp map
        embedded = np.array([self._exp_map_origin(row * 0.9) for row in normalized])
        # Labels: future return direction (+4 bars)
        labels = np.zeros(len(embedded))
        for i in range(len(embedded) - 4):
            ai = start + i
            if ai + 4 < n:
                labels[i] = 1.0 if closes[ai + 4] > closes[ai] else -1.0
        # Current point
        ci = len(embedded) - 1 + idx  # idx is typically -2 (use second-to-last completed bar)
        if ci < 0 or ci >= len(embedded): return 0
        current = embedded[ci]
        # k-NN with hyperbolic distance
        distances = []
        for i in range(max(0, ci - self.lookback), ci - 4):
            if labels[i] != 0:
                d = self._poincare_distance(current, embedded[i])
                distances.append((d, labels[i]))
        if len(distances) < self.k: return 0
        distances.sort(key=lambda x: x[0])
        top_k = distances[:self.k]
        bull_w = sum(1.0/(d+1e-10) for d, lbl in top_k if lbl > 0)
        bear_w = sum(1.0/(d+1e-10) for d, lbl in top_k if lbl < 0)
        if bull_w > bear_w * 1.2: return 1
        elif bear_w > bull_w * 1.2: return -1
        return 0

    def classify_fast(self, features, closes, current_idx, k=8, lookback=300):
        n = current_idx + 1
        if n < 50:
            return 0
        start = max(0, n - lookback)
        wf = features[start:n]
        mins = np.min(wf, axis=0)
        maxs = np.max(wf, axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        normalized = 2 * (wf - mins) / ranges - 1
        v_norm = np.linalg.norm(normalized, axis=1, keepdims=True)
        v_norm = np.maximum(v_norm, 1e-10)
        embedded = np.tanh(v_norm / 2.0) * normalized / v_norm
        labels = np.zeros(len(embedded))
        for i in range(len(embedded) - 4):
            ai = start + i
            if ai + 4 < n:
                labels[i] = 1.0 if closes[ai + 4] > closes[ai] else -1.0
        ci = len(embedded) - 1
        current = embedded[ci]
        train_start = max(0, ci - lookback)
        train_end = ci - 4
        if train_end <= train_start:
            return 0
        train_embedded = embedded[train_start:train_end]
        train_labels = labels[train_start:train_end]
        valid_mask = train_labels != 0
        if not np.any(valid_mask):
            return 0
        train_embedded = train_embedded[valid_mask]
        train_labels = train_labels[valid_mask]
        diff_sq = np.sum((train_embedded - current)**2, axis=1)
        u_sq = np.sum(current**2)
        v_sq = np.sum(train_embedded**2, axis=1)
        denom = (1.0 - u_sq) * (1.0 - v_sq)
        denom = np.maximum(denom, 1e-10)
        arg = 1.0 + 2.0 * diff_sq / denom
        distances = np.arccosh(np.maximum(arg, 1.0))
        if len(distances) < k:
            return 0
        top_k_indices = np.argsort(distances)[:k]
        top_k_dists = distances[top_k_indices]
        top_k_labels = train_labels[top_k_indices]
        weights = 1.0 / (top_k_dists + 1e-10)
        bull_w = np.sum(weights[top_k_labels > 0])
        bear_w = np.sum(weights[top_k_labels < 0])
        if bull_w > bear_w * 1.2:
            return 1
        elif bear_w > bull_w * 1.2:
            return -1
        return 0

class OUPingPong:
    """
    Jump-Diffusion Ornstein-Uhlenbeck süreci: dX = θ(μ - X)dt + σdW + J dN
    Yatay piyasada sıçramaları (jumps) filtreler, JD-OU parametrelerini tahmin eder ve sıçrama durumunda cooldown uygular.
    """
    def __init__(self, window=100):
        self.window = window
        self.theta = 0.0
        self.mu = 0.0
        self.sigma_ou = 0.0
        self.half_life = float('inf')
        self.ou_upper = 0.0
        self.ou_lower = 0.0
        self.ou_stop_upper = 0.0
        self.ou_stop_lower = 0.0
        self.is_valid = False
        self.jump_cooldown = 0
        self.jump_detected = False
        # Jump Diffusion Parametreleri
        self.jump_intensity = 0.0
        self.jump_mean = 0.0
        self.jump_std = 0.0

    def fit(self, prices):
        """OLS ile formal Jump Diffusion OU parametrelerini tahmin et"""
        if len(prices) < 30:
            self.is_valid = False; return
            
        # 1. Sıçrama Tespiti (Jump Detection)
        returns = np.diff(prices)
        median_ret = np.median(returns)
        mad = np.median(np.abs(returns - median_ret))
        robust_std = mad * 1.4826 if mad > 1e-8 else np.std(returns)
        if robust_std < 1e-8:
            robust_std = 1e-8
            
        # Sıçramaları bulalım
        diff_from_median = np.abs(returns - median_ret)
        jump_threshold = 3.0 * robust_std
        jump_mask = diff_from_median > jump_threshold
        jumps = returns[jump_mask]
        
        # Sıçrama yoğunluğu (jump intensity) lambda = sıçrama sayısı / veri uzunluğu
        self.jump_intensity = float(len(jumps) / len(returns))
        if len(jumps) > 0:
            self.jump_mean = float(np.mean(jumps))
            self.jump_std = float(np.std(jumps))
        else:
            self.jump_mean = 0.0
            self.jump_std = 0.0

        # Son bardaki değişim sıçrama mı?
        last_ret = returns[-1]
        self.jump_detected = bool(abs(last_ret - median_ret) > jump_threshold)
        
        if self.jump_detected:
            self.jump_cooldown = 4  # 4 bar boyunca sinyal engelle
            
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
            self.is_valid = False
            return

        # 2. Sıçramalardan arındırılmış veri ile OU fit etme (De-jumped estimation)
        p = prices[-self.window:]
        n_len = len(p)
        if n_len < 30:
            self.is_valid = False; return
            
        X = p[:-1]; dX = np.diff(p)
        
        # Filtreleme: OLS regresyonunu bozmaması için sıçramaları hariç tutalım
        median_dx = np.median(dX)
        mad_dx = np.median(np.abs(dX - median_dx))
        robust_std_dx = mad_dx * 1.4826 if mad_dx > 1e-8 else np.std(dX)
        if robust_std_dx < 1e-8: robust_std_dx = 1e-8
        
        valid_mask = np.abs(dX - median_dx) <= 3.0 * robust_std_dx
        if np.sum(valid_mask) < 20:
            self.is_valid = False; return
            
        X_clean = X[valid_mask]
        dX_clean = dX[valid_mask]
        
        Xm = np.mean(X_clean); dXm = np.mean(dX_clean)
        Sxx = np.sum((X_clean - Xm)**2)
        if Sxx == 0:
            self.is_valid = False; return
        Sxy = np.sum((X_clean - Xm) * (dX_clean - dXm))
        b = Sxy / Sxx; a = dXm - b * Xm
        self.theta = -b
        if self.theta <= 0.001:
            self.is_valid = False; return
        self.mu = -a / b if abs(b) > 1e-10 else np.mean(p)
        
        # Residuals ve sigma hesaplama (sıçramasız)
        residuals = dX_clean - (a + b * X_clean)
        self.sigma_ou = float(np.std(residuals))
        self.half_life = math.log(2) / self.theta
        ou_std = self.sigma_ou / math.sqrt(2 * self.theta)
        # Cap corridor range to prevent excessive widening (max 0.5% deviation)
        max_allowed_std = self.mu * 0.005
        ou_std = min(ou_std, max_allowed_std)
        
        self.ou_upper = self.mu + 2 * ou_std
        self.ou_lower = self.mu - 2 * ou_std
        self.ou_stop_upper = self.mu + 3 * ou_std
        self.ou_stop_lower = self.mu - 3 * ou_std
        self.is_valid = 2 <= self.half_life <= 50

    def get_signal(self, price):
        if not self.is_valid or self.jump_cooldown > 0: return "HOLD", ""
        if price <= self.ou_lower: return "BUY", "OU-Ping"
        return "HOLD", ""

class DynamicTargetOptimizer:
    """
    Geçmiş işlemlerin sonuçlarına göre TP/SL hedeflerini dinamik olarak optimize eden katman.
    """
    def __init__(self, lookback_trades=50):
        self.lookback_trades = lookback_trades
        self.trade_history = []  # list of dicts: {"regime": str, "volatility": float, "max_excursion_pct": float, "max_drawdown_pct": float, "pnl_pct": float}

    def record_trade(self, regime, volatility, max_excursion_pct, max_drawdown_pct, pnl_pct):
        self.trade_history.append({
            "regime": regime,
            "volatility": float(volatility),
            "max_excursion_pct": float(max_excursion_pct),
            "max_drawdown_pct": float(max_drawdown_pct),
            "pnl_pct": float(pnl_pct)
        })
        if len(self.trade_history) > self.lookback_trades:
            self.trade_history.pop(0)

    def get_optimal_targets(self, regime_type, current_volatility, default_tp=0.3, default_sl=0.3):
        relevant = [t for t in self.trade_history if t["regime"] == regime_type]
        if len(relevant) < 3:
            return default_tp, default_sl

        relevant.sort(key=lambda x: abs(x["volatility"] - current_volatility))
        neighbors = relevant[:5]

        excursions = [t["max_excursion_pct"] for t in neighbors if t["max_excursion_pct"] > 0]
        drawdowns = [t["max_drawdown_pct"] for t in neighbors if t["max_drawdown_pct"] > 0]

        opt_tp = np.mean(excursions) * 0.80 if len(excursions) > 0 else default_tp
        opt_sl = np.mean(drawdowns) * 1.20 if len(drawdowns) > 0 else default_sl

        if regime_type == "trend":
            opt_tp = np.clip(opt_tp, 0.15, 2.0)
            opt_sl = np.clip(opt_sl, 0.15, 1.5)
        else:
            opt_tp = np.clip(opt_tp, 0.10, 1.0)
            opt_sl = np.clip(opt_sl, 0.10, 1.0)

        return float(opt_tp), float(opt_sl)

class RoughPathClassifier:
    """
    Rough Path Signatures (Kaba Patika İmzaları) + Pure NumPy Ridge Regression Classifier.
    Fiyat patikasını saf NumPy ile 1., 2. ve 3. seviye imza özelliklerine (signatures) dönüştürür.
    Herhangi bir dış kütüphane bağımlılığı olmadan (scikit-learn/lightgbm) rejim ve trend tahmin eder.
    """
    def __init__(self, window=14):
        self.window = window
        self.model = None

    def _compute_signatures(self, prices):
        """
        Pure NumPy implementation of Path Signatures up to Level 3.
        """
        n = len(prices)
        if n < self.window:
            return np.zeros((n, 9))
            
        features = []
        for i in range(n):
            if i < self.window - 1:
                features.append(np.zeros(9))
                continue
                
            w = prices[i - self.window + 1 : i + 1]
            w_mean = np.mean(w)
            w_std = np.std(w)
            if w_std < 1e-8:
                w_std = 1e-8
            path = (w - w_mean) / w_std
            
            dx = np.diff(path)
            sig_l1 = float(np.sum(dx))
            sig_l2_1 = 0.5 * (sig_l1 ** 2)
            
            dt = 1.0 / (self.window - 1)
            t_grid = np.arange(self.window) * dt
            
            s_1_2 = float(np.sum(t_grid[1:] * dx))
            s_2_1 = float(np.sum(path[1:] * dt))
            area = s_1_2 - s_2_1
            
            qv = float(np.sum(dx ** 2))
            cv = float(np.sum(dx ** 3))
            sig_l3_2 = (sig_l1 ** 3) / 6.0
            
            len_path = float(np.sum(np.abs(dx)))
            sign_changes = float(np.sum(np.diff(np.sign(dx)) != 0))
            
            features.append(np.array([
                sig_l1,
                sig_l2_1,
                area,
                qv,
                cv,
                sig_l3_2,
                len_path,
                sign_changes,
                path[-1]
            ]))
            
        return np.array(features)

    def fit(self, df):
        """
        Train the classifier on the historical price dataframe using Ridge Regression.
        """
        closes = df['close'].values
        n = len(closes)
        if n < self.window + 20:
            return
            
        X = self._compute_signatures(closes)
        
        y = np.zeros(n)
        for i in range(n - 4):
            ret = (closes[i+4] - closes[i]) / closes[i] * 100
            if ret > 0.05:
                y[i] = 1.0
            elif ret < -0.05:
                y[i] = -1.0
            else:
                y[i] = 0.0
                
        train_idx = np.arange(self.window, n - 4)
        X_train = X[train_idx]
        y_train = y[train_idx]
        
        # Add bias column
        X_bias = np.column_stack([X_train, np.ones(len(X_train))])
        
        # Closed-form Ridge Regression: W = (X^T X + alpha * I)^-1 X^T Y
        d = X_bias.shape[1]
        alpha = 10.0
        XTX = np.dot(X_bias.T, X_bias)
        XTY = np.dot(X_bias.T, y_train)
        
        try:
            self.model = np.linalg.solve(XTX + alpha * np.eye(d), XTY)
            log.info("RoughPathClassifier successfully trained using pure NumPy Ridge Regression.")
        except Exception as e:
            self.model = None
            log.error(f"RoughPathClassifier training failed: {e}")

    def predict(self, prices):
        """
        Predict the regime for the last window.
        """
        if self.model is None:
            return 0
        if len(prices) < self.window:
            return 0
            
        feats = self._compute_signatures(prices)[-1]
        feats_bias = np.append(feats, 1.0)
        pred_val = float(np.dot(feats_bias, self.model))
        
        if pred_val > 0.35:
            return 1
        elif pred_val < -0.35:
            return -1
        return 0

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


def roundtrip_cost_pct(commission_rate=COMMISSION_RATE, slippage_rate=0.0005, spread_rate=0.0001):
    """Total roundtrip trading cost in percent: commission both ways plus
    entry/exit slippage and half-spread — the floor every target must clear."""
    return 2.0 * commission_rate * 100.0 + 2.0 * (slippage_rate + spread_rate / 2.0) * 100.0


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


class PureGradientBoosting:
    """
    LightGBM-style gradient boosted trees (binary logistic), dependency-free NumPy
    implementation (drops in for LightGBM in this box of the architecture; the
    real library is used instead when installed).
    """

    def __init__(self, n_trees=40, depth=3, lr=0.1, min_leaf=20, n_bins=24, seed=42):
        self.n_trees = n_trees
        self.depth = depth
        self.lr = lr
        self.min_leaf = min_leaf
        self.n_bins = n_bins
        self.seed = seed
        self.trees = []
        self.f0 = 0.0

    def _fit_tree(self, X, g, hset, depth, rng):
        n, d = X.shape
        node = {"leaf": True, "value": float(np.sum(g) / (np.sum(hset) + 1e-9))}
        if depth == 0 or n < 2 * self.min_leaf:
            return node
        best = None
        feat_idx = rng.choice(d, size=max(1, int(math.sqrt(d)) + 2), replace=False)
        base_score = (np.sum(g) ** 2) / (np.sum(hset) + 1e-9)
        for j in feat_idx:
            xs = X[:, j]
            qs = np.quantile(xs, np.linspace(0.05, 0.95, self.n_bins))
            for thr in np.unique(qs):
                m = xs <= thr
                nl = int(np.sum(m))
                if nl < self.min_leaf or n - nl < self.min_leaf:
                    continue
                gl, gr = np.sum(g[m]), np.sum(g[~m])
                hl, hr = np.sum(hset[m]), np.sum(hset[~m])
                gain = gl * gl / (hl + 1e-9) + gr * gr / (hr + 1e-9) - base_score
                if best is None or gain > best[0]:
                    best = (gain, j, thr, m)
        if best is None or best[0] <= 1e-9:
            return node
        _, j, thr, m = best
        return {"leaf": False, "feat": int(j), "thr": float(thr),
                "left": self._fit_tree(X[m], g[m], hset[m], depth - 1, rng),
                "right": self._fit_tree(X[~m], g[~m], hset[~m], depth - 1, rng)}

    def _predict_tree(self, node, X):
        if node["leaf"]:
            return np.full(X.shape[0], node["value"])
        m = X[:, node["feat"]] <= node["thr"]
        out = np.empty(X.shape[0])
        out[m] = self._predict_tree(node["left"], X[m])
        out[~m] = self._predict_tree(node["right"], X[~m])
        return out

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        p_mean = float(np.clip(np.mean(y), 1e-3, 1 - 1e-3))
        self.f0 = math.log(p_mean / (1 - p_mean))
        F = np.full(len(y), self.f0)
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for _ in range(self.n_trees):
            p = _sigmoid(F)
            g = y - p                      # negative gradient
            hset = p * (1 - p)             # hessian (Newton leaf values)
            tree = self._fit_tree(X, g, hset, self.depth, rng)
            self.trees.append(tree)
            F += self.lr * self._predict_tree(tree, X)
        return self

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.f0)
        for tree in self.trees:
            F += self.lr * self._predict_tree(tree, X)
        return _sigmoid(F)

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
        usable = np.array([vidx2[t] + GEO_BARRIER_HOLD < len(closes) for t in range(T)])
        y = np.zeros(T)
        for t in range(T):
            if usable[t]:
                y[t] = MetaLabeler.barrier_dir(
                    closes, vidx2[t], tp_arr[t], max_hold=GEO_BARRIER_HOLD)

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
            gb = PureGradientBoosting(n_trees=18, depth=2, seed=self.seed)
            gb.fit(Xf[tr_mask][:, cols], y[tr_mask])
            va = usable & (np.arange(T) >= cal_start)
            stab[f"{gname}_auc"] = PureGradientBoosting.auc(y[va], gb.predict_proba(Xf[va][:, cols])) if np.any(va) else 0.5
        self.feature_mask = np.ones(Xf.shape[1], dtype=bool)
        for gname, gmask in masks.items():
            if stab.get(f"{gname}_auc", 0.5) < 0.47:
                self.feature_mask[np.where(np.concatenate([gmask, np.zeros(Xf.shape[1] - n_innov, dtype=bool)]))[0]] = False

        # main classifier (LightGBM slot) — predicts P(up)
        self.gbm = PureGradientBoosting(n_trees=40, depth=3, seed=self.seed)
        self.gbm.fit(Xf[tr_mask][:, self.feature_mask], y[tr_mask])
        p_all = self.gbm.predict_proba(Xf[:, self.feature_mask])
        # symmetric directional thresholds: long if p_up high, short if p_up low
        self.p_hi = float(np.quantile(p_all[:cal_start], 0.70))
        self.p_lo = float(np.quantile(p_all[:cal_start], 0.30))

        # OVERFITTING diagnostic: primary-model AUC in-sample vs on the held-out
        # chronological tail. A large positive gap = the model memorized the train
        # window. Purely observational, surfaced in the dashboards.
        va_mask = usable & (np.arange(T) >= cal_start)
        train_auc = PureGradientBoosting.auc(y[tr_mask], p_all[tr_mask]) if np.any(tr_mask) else 0.5
        oos_auc = PureGradientBoosting.auc(y[va_mask], p_all[va_mask]) if np.any(va_mask) else 0.5
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
                closes, vidx2[t], tp_arr[t], sl_arr[t], side, max_hold=GEO_BARRIER_HOLD))
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
            self.conformal.calibrate(pm, dm)

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
        base_tp = float(get_active_parameters().get("TP_PERCENT", 0.3))
        cost = self.cost_pct
        bar_vol_pct = vol["rv"][vidx] * 100.0
        tp = np.maximum.reduce([
            np.full(len(vidx), base_tp),
            np.full(len(vidx), 3.0 * cost),
            2.0 * bar_vol_pct * math.sqrt(GEO_BARRIER_HOLD),
        ])
        sl = np.maximum(1.5 * cost, 0.5 * tp)
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

    def live_state(self, extra=None):
        s = {
            "status": self.status,
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
        self.embargo = max(GEOM_RESOLUTIONS) + GEO_BARRIER_HOLD

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
        combos = []
        for tp in (0.6, 1.0, 1.6):
            for tm in (2.0, 3.0, 4.0):
                cmb = dict(base)
                cmb["TP_PERCENT"] = tp
                cmb["TRAIL_MULT"] = tm
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

        best_ci, best_stable, best_var = 0, -1, float("inf")
        for ci in fold_results:
            rs = fold_results[ci]
            stable = sum(1 for r in rs if r["pf"] > 1.10 and r["calmar"] > 0.8)
            pfs = [min(r["pf"], 10.0) for r in rs]
            var = float(np.std(pfs)) if pfs else 0.0
            if stable > best_stable or (stable == best_stable and var < best_var):
                best_ci, best_stable, best_var = ci, stable, var
        challenger = dict(combos[best_ci])
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
        log.info(f"[WFO] overfit: mean IS-OOS AUC gap {overfit['mean_gap']:.3f} "
                 f"({overfit['verdict']}), mean OOS AUC {overfit['mean_oos_auc']:.3f}")
        return {
            "challenger": challenger,
            "stability_count": int(best_stable),
            "variance": float(best_var if best_var != float("inf") else 0.0),
            "slices_evaluated": self.n_folds,
            "engine": "geometric-purged-wfo",
            "diagnostics": pipeline.diagnostics,
            "d_anchor_log": pipeline.panel.d_anchor_log,
            "delta_hat_series": pipeline.delta_hat_series,
            "overfit": overfit,
            "allow_short": bool(pipeline.allow_short),
            "schema": pipeline.schema.label() if pipeline.schema else "-",
            "fold_results": {str(ci): fold_results[ci] for ci in fold_results},
        }


def run_geometric_backtest(df, params, timeframe="1m", seed=42):
    """Backtest stage of the flow: train on the first window, trade the purged remainder."""
    df = df.reset_index(drop=True)
    n = len(df)
    split = max(GeometricPipeline.MIN_TRAIN_BARS, int(n * 0.35))
    embargo = max(GEOM_RESOLUTIONS) + GEO_BARRIER_HOLD
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
    def __init__(self, enable_geometry=True):
        self.regime = RoughPathClassifier()
        self.ou = OUPingPong()
        # V3.6 learned-geometry pipeline; the legacy classifiers remain as the
        # fallback while the first window is still being collected/trained.
        self.geometry = GeometricPipeline(bot_state.get("timeframe", "1m")) if enable_geometry else None

    def process(self, df, params=None, force_retrain=False):
        if params is None:
            params = get_active_parameters()
        
        fast_len = int(params.get("FAST_LENGTH", 8))
        slow_len = int(params.get("SLOW_LENGTH", 21))
        vol_len = int(params.get("VOL_LENGTH", 14))
        cvd_len = int(params.get("CVD_LENGTH", 14))
        band_mult = float(params.get("BAND_MULT", 2.5))
        margin = float(params.get("MIN_PROFIT_MARGIN", 0.3))
        
        c, h, l, o, v = df['close'].values, df['high'].values, df['low'].values, df['open'].values, df['volume'].values
        fg = gaussian_filter(c, fast_len); sg = gaussian_filter(c, slow_len)
        gv = gaussian_filter(calc_true_range(h,l,c), vol_len)
        delta = np.where(c>o, v, np.where(c<o, -v, 0.0)); cvd_g = gaussian_filter(np.cumsum(delta), cvd_len)
        
        # Train classifier if not trained yet or forced
        if self.regime.model is None or force_retrain:
            self.regime.fit(df)
            
        i = len(df)-1
        ub, lb = sg[i]+gv[i]*band_mult, sg[i]-gv[i]*band_mult
        
        # Predict regime from Rough Path Signatures
        pred_direction = self.regime.predict(c)
        is_r = (pred_direction == 0)
        
        sig, st = "HOLD", ""
        # Trend: model predicts 1 (bullish) or -1 (bearish)
        if not is_r and pred_direction == 1:
            sig, st = "BUY", "Trend"
        elif not is_r and pred_direction == -1:
            sig, st = "SELL", "Trend"
        # Ranging: OU model primary, fallback to bands
        elif is_r and self.ou.is_valid:
            ou_sig, ou_type = self.ou.get_signal(c[i])
            if ou_sig != "HOLD":
                sig, st = ou_sig, ou_type
        elif is_r and l[i]<lb and c[i]>lb:
            ml = ((sg[i]-lb)/lb*100)>=margin if lb>0 else False
            if ml:
                sig, st = "BUY", "Ping"
        elif is_r and h[i]>ub and c[i]<ub:
            ms = ((ub-sg[i])/sg[i]*100)>=margin if sg[i]>0 else False
            if ms:
                sig, st = "SELL", "Pong"

        # Fit OU model for future/ranging calculations
        self.ou.fit(c[max(0, len(c)-100):])

        # ── V3.6 learned geometry: once trained, the geometric decision chain
        # (GBM → meta labeling → conformal gate) is the authoritative signal.
        geom_state = {}
        if self.geometry is not None:
            try:
                geom_state = self.geometry.on_bar(
                    df, force_retrain=force_retrain,
                    timeframe=bot_state.get("timeframe", self.geometry.timeframe))
            except Exception as e_geom:
                log.error(f"Geometry pipeline error: {e_geom}")
                geom_state = self.geometry.live_state() if self.geometry else {}
            if geom_state.get("status") == "ready":
                gsig = geom_state.get("signal", "HOLD")
                if gsig in ("BUY", "SELL"):
                    sig, st = gsig, "Geo"
                else:
                    sig, st = "HOLD", ""

        def sanitize(v, default=0.0):
            if v is None or math.isnan(v) or math.isinf(v):
                return default
            return float(v)

        return {
            "signal": sig,
            "type": st,
            "geom": geom_state,
            "price": float(c[i]),
            "gauss_vol": float(gv[i]),
            "slow_gauss": float(sg[i]),
            "upper_band": float(ub),
            "lower_band": float(lb),
            "is_ranging": bool(is_r),
            "fast_gauss": float(fg[i]),
            "fg_list": fg,
            "sg_list": sg,
            "ub_list": sg+gv*band_mult,
            "lb_list": sg-gv*band_mult,
            "hyp_direction": int(pred_direction),
            "ou_theta": sanitize(self.ou.theta),
            "ou_mu": sanitize(self.ou.mu),
            "ou_half_life": sanitize(self.ou.half_life, 999.0),
            "ou_upper": sanitize(self.ou.ou_upper),
            "ou_lower": sanitize(self.ou.ou_lower),
            "ou_stop_upper": sanitize(self.ou.ou_stop_upper),
            "ou_stop_lower": sanitize(self.ou.ou_stop_lower),
            "ou_valid": bool(self.ou.is_valid),
            "ou_jump_intensity": float(self.ou.jump_intensity),
            "ou_jump_mean": float(self.ou.jump_mean),
            "ou_jump_std": float(self.ou.jump_std),
            "ou_jump_detected": bool(self.ou.jump_detected),
            "ou_jump_cooldown": int(self.ou.jump_cooldown)
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
        if not self.is_open: return None
        if force_close: return "Zaman Dilimi Degisimi (Force Close)"
        if self.entry_type in ("Ping","Pong","OU-Ping","OU-Pong") and not info['is_ranging']: return "Acil Cikis (Rejim Degisti)"
        
        if self.side == "long":
            if self.has_taken_partial_tp:
                if price <= self.trail_stop_30: return "Trail Stop (30%)"
                return None
            
            if self.entry_type in ("Ping", "OU-Ping"):
                if price <= self.entry_price * (1 - self.sl_percent/100): return "Stop Loss (Ping)"
                if price >= self.entry_price * (1 + self.tp_percent/100): return "Ping TP"
            elif self.entry_type in ("Trend", "Geo"):
                if price <= self.entry_price * (1 - self.sl_percent/100): return "Stop Loss (Trend)"
                if price <= self.trail_stop_70:
                    if self.trail_stop_70 == self.trail_stop_30: return "Stop Loss (Trend)"
                    return "PARTIAL_TP"
                if price >= self.entry_price * (1 + self.tp_percent/100): return f"Trend TP ({self.tp_percent:.2f}%)"
                
        elif self.side == "short":
            if self.has_taken_partial_tp:
                if price >= self.trail_stop_30: return "Trail Stop (30%)"
                return None
            
            if self.entry_type in ("Pong", "OU-Pong"):
                if price >= self.entry_price * (1 + self.sl_percent/100): return "Stop Loss (Pong)"
                if price <= self.entry_price * (1 - self.tp_percent/100): return "Pong TP"
            elif self.entry_type in ("Trend", "Geo"):
                if price >= self.entry_price * (1 + self.sl_percent/100): return "Stop Loss (Trend)"
                if price >= self.trail_stop_70:
                    if self.trail_stop_70 == self.trail_stop_30: return "Stop Loss (Trend)"
                    return "PARTIAL_TP"
                if price <= self.entry_price * (1 - self.tp_percent/100): return f"Trend TP ({self.tp_percent:.2f}%)"
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
            ou_status = f"Valid (Half-life: {bot_state['ou_half_life']:.1f})" if bot_state["ou_valid"] else "Invalid"
            msg = (
                f"📊 *ANLIK DURUM RAPORU*\n\n"
                f"📈 *Fiyat:* {bot_state['price']:.2f} USDT\n"
                f"🤖 *Durum:* {active}\n"
                f"🔄 *Mod:* {mode} Mod\n"
                f"📊 *Rejim:* {bot_state['regime']}\n"
                f"🎯 *Sinyal:* {bot_state['signal']} ({bot_state['signal_type'] or 'Yok'})\n"
                f"⚖️ *OBI:* {bot_state['obi']:.2f}\n"
                f"💼 *Pozisyon:* {pos_desc}\n"
                f"💵 *Bakiye (Paper):* {bot_state['virtual_balance']:.2f} USDT\n"
                f"📈 *OU Model:* {ou_status}\n"
                f"📐 *OU μ (Denge):* {bot_state['ou_mu']:.2f}\n"
                f"📐 *OU θ (Hız):* {bot_state['ou_theta']:.4f}"
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
        self.public_binance = ccxt.binance({'enableRateLimit': True})  # Added for 10s sub-minute data
        self.signal_engine = SignalEngine()
        self.position = PositionManager()
        self.telegram = TelegramController(self)
        self.dynamic_target_optimizer = DynamicTargetOptimizer()
        self.shadow_dynamic_target_optimizer = DynamicTargetOptimizer()

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

    async def fetch_ohlcv_large(self, symbol, timeframe, limit=3000):
        all_ohlcv = []
        target_limit = limit
        tf_ms = 60 * 1000  # 1m default
        if timeframe == "5m": tf_ms = 5 * 60 * 1000
        elif timeframe == "15m": tf_ms = 15 * 60 * 1000
        elif timeframe == "1h": tf_ms = 60 * 60 * 1000
        elif timeframe == "4h": tf_ms = 4 * 60 * 60 * 1000

        now_ms = int(datetime.now().timestamp() * 1000)
        since = now_ms - (target_limit * tf_ms)

        for _ in range(5):
            try:
                batch = await asyncio.to_thread(self.exchange.fetch_ohlcv, symbol, timeframe, since, 1000)
                if not batch:
                    break
                all_ohlcv.extend(batch)
                all_ohlcv = sorted({x[0]: x for x in all_ohlcv}.values(), key=lambda x: x[0])
                if len(all_ohlcv) >= target_limit:
                    break
                since = all_ohlcv[-1][0] + tf_ms
            except Exception as e:
                log.error(f"Error in fetch_ohlcv_large page: {e}")
                break

        df = pd.DataFrame(all_ohlcv[-target_limit:], columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

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

            # Record completed trade metrics
            max_excursion = (max_price_seen - saved_entry_price) / saved_entry_price * 100 if pos_side == "long" else (saved_entry_price - min_price_seen) / saved_entry_price * 100
            max_drawdown = (saved_entry_price - min_price_seen) / saved_entry_price * 100 if pos_side == "long" else (max_price_seen - saved_entry_price) / saved_entry_price * 100
            self.dynamic_target_optimizer.record_trade(
                "ranging" if saved_entry_type in ("Ping", "Pong", "OU-Ping", "OU-Pong") else "trend",
                entry_volatility,
                max_excursion,
                max_drawdown,
                pnl_pct
            )

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
        
        margin = float(active_params.get("MIN_PROFIT_MARGIN", 0.3))
        band_mult = float(active_params.get("BAND_MULT", 2.5))
        ping_stop_mult = float(active_params.get("PING_STOP_MULT", 0.5))
        trail_mult = float(active_params.get("TRAIL_MULT", 3.0))
        
        # Repaint fix: process completed bars only
        df_completed = df.iloc[:-1].copy() if len(df) > 30 else df
        info = self.signal_engine.process(df_completed, params=active_params, force_retrain=force_retrain)
        price = float(df['close'].iloc[-1])
        is_r = info['is_ranging']
        log.info(f"Tick [{bot_state['timeframe']}]: Price={price:.2f}, Signal={info['signal']} ({info['type'] or 'None'}), Ranging={is_r}, Active={bot_state['is_trading_active']}, OBI={bot_state.get('obi', 0.0):.2f}")
        bot_state["fast_gauss"] = float(info['fast_gauss'])
        bot_state["slow_gauss"] = float(info['slow_gauss'])
        bot_state["upper_band"] = float(info['upper_band'])
        bot_state["lower_band"] = float(info['lower_band'])
        bot_state["gauss_vol"] = float(info['gauss_vol'])
        bot_state["is_ranging"] = is_r
        bot_state["regime"] = "Yatay (Ranging)" if is_r else "Trend"
        bot_state["signal"] = info['signal']
        bot_state["signal_type"] = info['type']
        
        bot_state["geom"] = info.get("geom", bot_state.get("geom", {"status": "collecting", "signal": "HOLD"}))
        bot_state["hyp_direction"] = info["hyp_direction"]
        bot_state["ou_theta"] = info["ou_theta"]
        bot_state["ou_mu"] = info["ou_mu"]
        bot_state["ou_half_life"] = info["ou_half_life"]
        bot_state["ou_upper"] = info["ou_upper"]
        bot_state["ou_lower"] = info["ou_lower"]
        bot_state["ou_valid"] = info["ou_valid"]
        bot_state["ou_jump_intensity"] = info["ou_jump_intensity"]
        bot_state["ou_jump_mean"] = info["ou_jump_mean"]
        bot_state["ou_jump_std"] = info["ou_jump_std"]
        bot_state["ou_jump_detected"] = info["ou_jump_detected"]
        bot_state["ou_jump_cooldown"] = info["ou_jump_cooldown"]

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
            # 1. Update stops using both normal and 1/6th lower timeframe volatility
            gv_normal = info['gauss_vol']
            gv_lower = gv_normal
            try:
                if bot_state["timeframe"] == "1m":
                    # For 1m timeframe, MEXC doesn't support sub-minute.
                    # We fetch 1s data from Binance, resample to 10s, and calculate volatility!
                    ohlcv_1s = await asyncio.to_thread(self.public_binance.fetch_ohlcv, SYMBOL, "1s", None, 600)
                    df_1s = pd.DataFrame(ohlcv_1s, columns=['timestamp','open','high','low','close','volume'])
                    df_1s['timestamp'] = pd.to_datetime(df_1s['timestamp'], unit='ms')
                    df_1s.set_index('timestamp', inplace=True)
                    df_10s = df_1s.resample('10s').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    c_l, h_l, l_l = df_10s['close'].values, df_10s['high'].values, df_10s['low'].values
                    gv_lower_list = gaussian_filter(calc_true_range(h_l, l_l, c_l), VOL_LENGTH)
                    gv_lower = float(gv_lower_list[-1])
                else:
                    lower_tf = get_lower_tf(bot_state["timeframe"])
                    if lower_tf != bot_state["timeframe"]:
                        if lower_tf in ("2m", "10m"):
                            # Local resampling using 1m data since 2m/10m are not natively supported by MEXC Spot
                            ohlcv_1m = await asyncio.to_thread(self.exchange.fetch_ohlcv, SYMBOL, "1m", None, 200)
                            df_1m = pd.DataFrame(ohlcv_1m, columns=['timestamp','open','high','low','close','volume'])
                            df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms')
                            df_1m.set_index('timestamp', inplace=True)
                            
                            resample_rule = "2Min" if lower_tf == "2m" else "10Min"
                            df_resampled = df_1m.resample(resample_rule).agg({
                                'open': 'first',
                                'high': 'max',
                                'low': 'min',
                                'close': 'last',
                                'volume': 'sum'
                            }).dropna()
                            c_l, h_l, l_l = df_resampled['close'].values, df_resampled['high'].values, df_resampled['low'].values
                            gv_lower_list = gaussian_filter(calc_true_range(h_l, l_l, c_l), VOL_LENGTH)
                            gv_lower = float(gv_lower_list[-1])
                            log.info(f"MTF: Resampled 1m to {lower_tf} locally. Volatility = {gv_lower:.4f}")
                        else:
                            ohlcv_lower = await asyncio.to_thread(self.exchange.fetch_ohlcv, SYMBOL, lower_tf, None, 100)
                            df_lower = pd.DataFrame(ohlcv_lower, columns=['timestamp','open','high','low','close','volume'])
                            c_l, h_l, l_l = df_lower['close'].values, df_lower['high'].values, df_lower['low'].values
                            gv_lower_list = gaussian_filter(calc_true_range(h_l, l_l, c_l), VOL_LENGTH)
                            gv_lower = float(gv_lower_list[-1])
            except Exception as ex_mtf:
                log.error(f"MTF gv fetch error: {ex_mtf}")

            self.position.update_stops(price, gv_normal, gv_lower)
            reason = self.position.check_exits(price, info)
            # Geometric positions also exit when the conformal score collapses while an
            # anomalous episode with negative expected return is active
            if reason is None and self.position.entry_type == "Geo" and info.get("geom", {}).get("exit_flag"):
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
                    close_qty = self.position.qty * 0.70
                    log.info(f"PARTIAL TP (70%) TRIGGERED! Qty to close: {close_qty:.6f} at {price:.2f}")

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
                    realized_pnl_usdt = (pos_invested * 0.70) * (pnl_pct / 100)
                    
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
                    self.position.qty *= 0.30
                    self.position.invested_amount *= 0.30
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
        if info['signal'] == "BUY" and obi_now < -0.3:
            obi_filter_pass = False
            log.info(f"BUY blocked by OBI ({obi_now:.2f} Sell Wall)")
        elif info['signal'] == "SELL" and obi_now > 0.3:
            obi_filter_pass = False
            log.info(f"SELL blocked by OBI ({obi_now:.2f} Buy Wall)")

        # CRITICAL Spot Restructure: Only open on BUY (no shorting on Spot)
        if not self.position.is_open and info['signal'] == "BUY" and obi_filter_pass:
            if bot_state["is_trading_active"]:
                try:
                    if info['type'] in ("OU-Ping", "OU-Pong"):
                        stop_dist = abs(price - info['ou_stop_lower'])
                    else:
                        stop_dist = info['gauss_vol'] * (ping_stop_mult if info['type'] in ("Ping","Pong") else trail_mult)
                    if stop_dist <= 0: stop_dist = price * 0.01

                    ou_target = info.get('ou_mu', 0.0)
                    ou_stop = info.get('ou_stop_lower', 0.0)

                    entry_vol = float(info['gauss_vol'])
                    reg_key = "ranging" if info['type'] in ("Ping", "Pong", "OU-Ping", "OU-Pong") else "trend"
                    active_p = get_active_parameters()
                    def_tp = float(active_p.get("TP_PERCENT", 0.3))
                    opt_tp, opt_sl = self.dynamic_target_optimizer.get_optimal_targets(
                        reg_key, entry_vol,
                        default_tp=def_tp if reg_key == "trend" else 0.3,
                        default_sl=def_tp if reg_key == "trend" else 0.3
                    )
                    min_trail_dist = 0.0
                    if info['type'] == "Geo":
                        # cost wall: learned-geometry targets are floored above the
                        # roundtrip cost and scaled to barrier-horizon volatility;
                        # the DTO can widen but never shrink them below the floor
                        g_state = info.get("geom", {}) or {}
                        cost = roundtrip_cost_pct()
                        bar_vol_pct = (entry_vol / price * 100.0) if price > 0 else 0.0
                        opt_tp = max(opt_tp, float(g_state.get("tp_pct") or 0.0), 3.0 * cost,
                                     2.0 * bar_vol_pct * math.sqrt(GEO_BARRIER_HOLD))
                        opt_sl = max(float(g_state.get("sl_pct") or 0.0), 1.5 * cost, 0.5 * opt_tp)
                        min_trail_dist = price * opt_tp / 100.0 * 0.5
                    active_mode = bot_state["trading_mode"]

                    if active_mode == "REAL":
                        balance = await asyncio.to_thread(self.exchange.fetch_balance)
                        real_usdt = float(balance.get('USDT', {}).get('free', 0) or 0)
                        bot_state["real_balance"] = real_usdt
                        # Record starting balance for first trade if not recorded
                        if bot_state.get("start_real_balance", 0.0) <= 0:
                            bot_state["start_real_balance"] = real_usdt
                            bot_state["peak_real_balance"] = real_usdt

                        if real_usdt >= 5:
                            side = "buy"
                            risk_amount = real_usdt * (TARGET_RISK_PERCENT / 100.0)
                            target_qty = risk_amount / stop_dist
                            max_qty = (real_usdt * MAX_CAPITAL_ALLOCATION) / price
                            qty = min(target_qty, max_qty)
                            
                            # Precision check & min 5 USDT enforcement
                            formatted_qty = self.exchange.amount_to_precision(SYMBOL, qty)
                            order_qty = float(formatted_qty)
                            
                            if order_qty * price < 5.0:
                                # Scale up to minimum order size if real balance allows
                                min_qty = 5.2 / price
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

                            # Place Native Stop-loss Order on Exchange
                            stop_order_id = None
                            try:
                                stop_price_val = ou_stop if info['type'] == "OU-Ping" else (price - info['gauss_vol'] * (ping_stop_mult if info['type'] == "Ping" else trail_mult))
                                stop_order_id = await self.place_native_stop_loss(filled_qty, stop_price_val)
                            except Exception as ex_stop:
                                log.error(f"Failed to place native exchange stop order: {ex_stop}")

                            pos_side = "long"
                            self.position.open(pos_side, price, filled_qty, info['type'], info['gauss_vol'], invested, ou_target, ou_stop, mode="REAL", stop_order_id=stop_order_id, tp_percent=opt_tp, sl_percent=opt_sl, entry_volatility=entry_vol, min_trail_dist=min_trail_dist)
                            
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
                        if bot_state["virtual_balance"] >= 5:
                            side = "buy"
                            risk_amount = bot_state["virtual_balance"] * (TARGET_RISK_PERCENT / 100.0)
                            target_qty = risk_amount / stop_dist
                            max_qty = (bot_state["virtual_balance"] * MAX_CAPITAL_ALLOCATION) / price
                            qty = min(target_qty, max_qty)
                            
                            # Apply paper slippage on entry (0.05%)
                            slippage_price = price * 1.0005
                            invested = qty * slippage_price
                            
                            # Deduct entry commission fee (0.1%)
                            comm_entry = invested * 0.001
                            bot_state["virtual_balance"] -= comm_entry

                            log.info(f"PAPER ORDER: buy {info['type']} (Qty: {qty:.6f}, Entry Price: {slippage_price:.2f}, Invest: ${invested:.2f})")
                            pos_side = "long"
                            self.position.open(pos_side, slippage_price, qty, info['type'], info['gauss_vol'], invested, ou_target, ou_stop, mode="PAPER", tp_percent=opt_tp, sl_percent=opt_sl, entry_volatility=entry_vol, min_trail_dist=min_trail_dist)

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
                opt_tp = max(float(g_state.get("tp_pct") or 0.0), 3.0 * cost,
                             2.0 * bar_vol_pct * math.sqrt(GEO_BARRIER_HOLD))
                opt_sl = max(float(g_state.get("sl_pct") or 0.0), 1.5 * cost, 0.5 * opt_tp)
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
                elif bot_state["virtual_balance"] >= 5:
                    risk_amount = bot_state["virtual_balance"] * (TARGET_RISK_PERCENT / 100.0)
                    target_qty = risk_amount / stop_dist
                    max_qty = (bot_state["virtual_balance"] * MAX_CAPITAL_ALLOCATION) / price
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

        # Shadow Mode Processing
        shadow_challenger = bot_state["parameters_store"].get("shadow_challenger")
        if shadow_challenger:
            bot_state["shadow_active"] = True
            bot_state["shadow_parameters"] = shadow_challenger
            if not hasattr(self, "shadow_signal_engine"):
                # Shadow engine stays on the legacy core (no second geometry training)
                self.shadow_signal_engine = SignalEngine(enable_geometry=False)
                self.shadow_position = PositionManager()
        else:
            bot_state["shadow_active"] = False
            bot_state["shadow_parameters"] = {}

        if bot_state["shadow_active"]:
            try:
                shadow_info = self.shadow_signal_engine.process(df_completed, params=shadow_challenger, force_retrain=force_retrain)
                
                if self.shadow_position.is_open:
                    gv_normal = shadow_info['gauss_vol']
                    gv_lower = gv_normal
                    self.shadow_position.update_stops(price, gv_normal, gv_lower)
                    
                    shadow_reason = self.shadow_position.check_exits(price, shadow_info)
                    
                    bot_state["shadow_position_side"] = self.shadow_position.side
                    bot_state["shadow_position_entry"] = float(self.shadow_position.entry_price)
                    bot_state["shadow_position_qty"] = float(self.shadow_position.qty)
                    bot_state["shadow_position_pnl"] = float((price - self.shadow_position.entry_price)/self.shadow_position.entry_price*100 if self.shadow_position.side=="long" else (self.shadow_position.entry_price-price)/self.shadow_position.entry_price*100)
                    
                    if shadow_reason == "PARTIAL_TP":
                        close_qty = self.shadow_position.qty * 0.70
                        exit_price_slippage = price * 0.9995 if self.shadow_position.side == "long" else price * 1.0005
                        comm_exit = (close_qty * exit_price_slippage) * 0.001
                        
                        pnl_pct = (exit_price_slippage - self.shadow_position.entry_price)/self.shadow_position.entry_price*100 if self.shadow_position.side=="long" else (self.shadow_position.entry_price - exit_price_slippage)/self.shadow_position.entry_price*100
                        realized_pnl_usdt = (self.shadow_position.invested_amount * 0.70) * (pnl_pct / 100) - comm_exit
                        
                        bot_state["shadow_balance"] += realized_pnl_usdt
                        self.shadow_position.realized_pnl_usdt += realized_pnl_usdt
                        
                        bot_state["shadow_trades"].insert(0, {
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "type": self.shadow_position.entry_type,
                            "side": f"PART_TP_{self.shadow_position.side.upper()}",
                            "entry": float(self.shadow_position.entry_price),
                            "exit": float(exit_price_slippage),
                            "pnl": f"{pnl_pct:+.2f}%",
                            "reason": "First Trailing Stop Hit (70% TP)"
                        })
                        
                        self.shadow_position.qty *= 0.30
                        self.shadow_position.invested_amount *= 0.30
                        self.shadow_position.has_taken_partial_tp = True
                        bot_state["shadow_position_qty"] = float(self.shadow_position.qty)
                        
                    elif shadow_reason:
                        exit_price_slippage = price * 0.9995 if self.shadow_position.side == "long" else price * 1.0005
                        comm_exit = (self.shadow_position.qty * exit_price_slippage) * 0.001
                        
                        pnl_pct = (exit_price_slippage - self.shadow_position.entry_price)/self.shadow_position.entry_price*100 if self.shadow_position.side=="long" else (self.shadow_position.entry_price - exit_price_slippage)/self.shadow_position.entry_price*100
                        pnl_usdt = self.shadow_position.invested_amount * (pnl_pct / 100) - comm_exit
                        
                        total_trade_pnl_usdt = self.shadow_position.realized_pnl_usdt + pnl_usdt
                        bot_state["shadow_balance"] += pnl_usdt
                        
                        bot_state["shadow_trade_count"] += 1
                        bot_state["shadow_total_pnl"] += total_trade_pnl_usdt
                        
                        bot_state["shadow_trades"].insert(0, {
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "type": self.shadow_position.entry_type,
                            "side": "CLOSED",
                            "entry": float(self.shadow_position.entry_price),
                            "exit": float(exit_price_slippage),
                            "pnl": f"{(total_trade_pnl_usdt / (self.shadow_position.invested_amount / 0.30 if self.shadow_position.has_taken_partial_tp else self.shadow_position.invested_amount) * 100):+.2f}%",
                            "reason": shadow_reason
                        })
                        
                        # Record shadow completed trade metrics
                        max_excursion = (self.shadow_position.max_price_seen - self.shadow_position.entry_price) / self.shadow_position.entry_price * 100 if self.shadow_position.side == "long" else (self.shadow_position.entry_price - self.shadow_position.min_price_seen) / self.shadow_position.entry_price * 100
                        max_drawdown = (self.shadow_position.entry_price - self.shadow_position.min_price_seen) / self.shadow_position.entry_price * 100 if self.shadow_position.side == "long" else (self.shadow_position.max_price_seen - self.shadow_position.entry_price) / self.shadow_position.entry_price * 100
                        self.shadow_dynamic_target_optimizer.record_trade(
                            "ranging" if self.shadow_position.entry_type in ("Ping", "Pong", "OU-Ping", "OU-Pong") else "trend",
                            self.shadow_position.entry_volatility,
                            max_excursion,
                            max_drawdown,
                            pnl_pct
                        )
                        
                        self.shadow_position.close(shadow_reason, exit_price_slippage)
                        bot_state["shadow_position_side"] = None
                        bot_state["shadow_position_qty"] = 0.0
                
                if not self.shadow_position.is_open and shadow_info['signal'] == "BUY":
                    shadow_obi_pass = True
                    if bot_state.get("obi", 0) < -0.3:
                        shadow_obi_pass = False
                        
                    if shadow_obi_pass:
                        shadow_stop_dist = shadow_info['gauss_vol'] * (float(shadow_challenger.get('PING_STOP_MULT', 0.5)) if shadow_info['type'] in ("Ping","Pong") else float(shadow_challenger.get('TRAIL_MULT', 3.0)))
                        if shadow_stop_dist <= 0:
                            shadow_stop_dist = price * 0.01
                            
                        risk_amount = bot_state["shadow_balance"] * (TARGET_RISK_PERCENT / 100.0)
                        target_qty = risk_amount / shadow_stop_dist
                        max_qty = (bot_state["shadow_balance"] * MAX_CAPITAL_ALLOCATION) / price
                        qty = min(target_qty, max_qty)
                        invested = qty * price
                        
                        entry_fee = invested * 0.001
                        bot_state["shadow_balance"] -= entry_fee
                        
                        shadow_entry_vol = float(shadow_info['gauss_vol'])
                        shadow_reg_key = "ranging" if shadow_info['type'] in ("Ping", "Pong", "OU-Ping", "OU-Pong") else "trend"
                        shadow_def_tp = float(shadow_challenger.get("TP_PERCENT", 0.3))
                        shadow_opt_tp, shadow_opt_sl = self.shadow_dynamic_target_optimizer.get_optimal_targets(
                            shadow_reg_key, shadow_entry_vol, 
                            default_tp=shadow_def_tp if shadow_reg_key == "trend" else 0.3, 
                            default_sl=shadow_def_tp if shadow_reg_key == "trend" else 0.3
                        )
                        self.shadow_position.open("long", price, qty, shadow_info['type'], shadow_info['gauss_vol'], invested, params=shadow_challenger, tp_percent=shadow_opt_tp, sl_percent=shadow_opt_sl, entry_volatility=shadow_entry_vol)
                        bot_state["shadow_position_side"] = "long"
                        bot_state["shadow_position_entry"] = float(price)
                        bot_state["shadow_position_qty"] = float(qty)
                        bot_state["shadow_position_pnl"] = 0.0
                        
                        bot_state["shadow_trades"].insert(0, {
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "type": shadow_info['type'],
                            "side": "LONG",
                            "entry": float(price),
                            "exit": 0.0,
                            "pnl": "OPEN",
                            "reason": f"Signal: {shadow_info['signal']}"
                        })
            except Exception as e_shadow:
                log.error(f"Shadow Mode tick error: {e_shadow}")

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
        log.info("QUANT BOT V3.6 (Learned Geometry) - Paper Trading Active")
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
                entry_type = "Trend"
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
                    await asyncio.wait_for(tf_change_event.wait(), timeout=LOOP_INTERVAL)
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
    def __init__(self, df, commission_rate=0.001, slippage_rate=0.0005, spread_rate=0.0001):
        self.df = df
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.spread_rate = spread_rate

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

        dynamic_target_optimizer = DynamicTargetOptimizer()
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
        
        fast_len = int(params.get("FAST_LENGTH", 8))
        slow_len = int(params.get("SLOW_LENGTH", 21))
        vol_len = int(params.get("VOL_LENGTH", 14))
        cvd_len = int(params.get("CVD_LENGTH", 14))
        band_mult = float(params.get("BAND_MULT", 2.5))
        margin = float(params.get("MIN_PROFIT_MARGIN", 0.3))
        trail_mult = float(params.get("TRAIL_MULT", 3.0))
        ping_stop_mult = float(params.get("PING_STOP_MULT", 0.5))
        tp_percent = float(params.get("TP_PERCENT", 3.0))
        
        fg = gaussian_filter(c, fast_len)
        sg = gaussian_filter(c, slow_len)
        gv = gaussian_filter(calc_true_range(h, l, c), vol_len)
        delta = np.where(c > o, v, np.where(c < o, -v, 0.0))
        cvd_g = gaussian_filter(np.cumsum(delta), cvd_len)
        
        rough_path_classifier = RoughPathClassifier()
        features = rough_path_classifier._compute_signatures(c) if geo is None else None
        ou = OUPingPong()

        start_idx = max(slow_len + 100, 300) if geo is None else max(slow_len + 5, 65)
        if start_idx >= len(self.df):
            return {
                "trade_count": 0, "total_pnl_pct": 0.0, "total_pnl_usdt": 0.0,
                "profit_factor": 1.0, "max_drawdown_pct": 0.0, "calmar_ratio": 0.0,
                "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "recovery_factor": 0.0,
                "win_rate": 0.0, "expectancy": 0.0, "trades": []
            }
            
        if geo is None:
            rough_path_classifier.fit(self.df.iloc[:start_idx])

        for idx in range(start_idx, len(self.df)):
            price = float(c[idx])
            prev_price = float(c[idx-1])

            if geo is None:
                if rough_path_classifier.model is not None:
                    feats_bias = np.append(features[idx-1], 1.0)
                    pred_val = float(np.dot(feats_bias, rough_path_classifier.model))
                    if pred_val > 0.35: pred_direction = 1
                    elif pred_val < -0.35: pred_direction = -1
                    else: pred_direction = 0
                else:
                    pred_direction = 0
                is_r = (pred_direction == 0)
                hyp_direction = pred_direction
                ou.fit(c[max(0, idx - 100):idx])
            else:
                pred_direction = 0
                hyp_direction = 0
                is_r = False
            
            if position_side is not None:
                max_price_seen = max(max_price_seen, price)
                min_price_seen = min(min_price_seen, price)
                gv_normal = float(gv[idx-1])
                gv_lower = 0.7 * gv_normal
                d30 = max(gv_normal * trail_mult, min_trail_dist)
                d70 = max(gv_lower * trail_mult, min_trail_dist)

                if position_side == "long":
                    trail_stop_30 = max(trail_stop_30, price - d30)
                    if not has_taken_partial_tp:
                        trail_stop_70 = max(trail_stop_70, price - d70)
                    ping_stop = max(ping_stop, price - gv_normal * ping_stop_mult)
                else:
                    trail_stop_30 = min(trail_stop_30, price + d30)
                    if not has_taken_partial_tp:
                        trail_stop_70 = min(trail_stop_70, price + d70)
                    ping_stop = min(ping_stop, price + gv_normal * ping_stop_mult)
                    
                exit_reason = None
                if position_side == "long":
                    if has_taken_partial_tp:
                        if price <= trail_stop_30: exit_reason = "Trail Stop (30%)"
                    elif entry_type in ("Ping", "OU-Ping"):
                        if price <= entry_price * (1 - current_sl_percent/100): exit_reason = "Stop Loss (Ping)"
                        elif price >= entry_price * (1 + current_tp_percent/100): exit_reason = "Ping TP"
                    elif entry_type in ("Trend", "Geo"):
                        if price <= entry_price * (1 - current_sl_percent/100): exit_reason = "Stop Loss (Trend)"
                        elif price <= trail_stop_70: exit_reason = "PARTIAL_TP"
                        elif price >= entry_price * (1 + current_tp_percent/100): exit_reason = f"Trend TP ({current_tp_percent:.2f}%)"
                else:
                    if has_taken_partial_tp:
                        if price >= trail_stop_30: exit_reason = "Trail Stop (30%)"
                    elif entry_type in ("Pong", "OU-Pong"):
                        if price >= entry_price * (1 + current_sl_percent/100): exit_reason = "Stop Loss (Pong)"
                        elif price <= entry_price * (1 - current_tp_percent/100): exit_reason = "Pong TP"
                    elif entry_type in ("Trend", "Geo"):
                        if price >= entry_price * (1 + current_sl_percent/100): exit_reason = "Stop Loss (Trend)"
                        elif price >= trail_stop_70: exit_reason = "PARTIAL_TP"
                        elif price <= entry_price * (1 - current_tp_percent/100): exit_reason = f"Trend TP ({current_tp_percent:.2f}%)"

                if exit_reason is None:
                    if entry_type in ("Ping", "Pong", "OU-Ping", "OU-Pong") and not is_r:
                        exit_reason = "Acil Cikis (Rejim Degisti)"
                    elif geo is not None and entry_type == "Geo" and bool(geo["exit_flag"][idx-1]):
                        exit_reason = "Geo Exit (Conformal/Transition)"
                        
                if exit_reason == "PARTIAL_TP":
                    close_qty = qty * 0.70
                    slippage_factor = self.slippage_rate + self.spread_rate / 2
                    exec_price = price * (1 - slippage_factor) if position_side == "long" else price * (1 + slippage_factor)
                    pnl_pct = (exec_price - entry_price) / entry_price * 100 if position_side == "long" else (entry_price - exec_price) / entry_price * 100
                    fee = close_qty * exec_price * self.commission_rate
                    pnl_usdt = (invested_amount * 0.70) * (pnl_pct / 100) - fee
                    
                    balance += pnl_usdt
                    pnl_list.append(pnl_pct)
                    if pnl_pct > 0: gross_profit += pnl_usdt
                    else: gross_loss += abs(pnl_usdt)
                    
                    qty *= 0.30
                    invested_amount *= 0.30
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
                    
                    # Record completed trade
                    max_excursion = (max_price_seen - entry_price) / entry_price * 100 if position_side == "long" else (entry_price - min_price_seen) / entry_price * 100
                    max_drawdown = (entry_price - min_price_seen) / entry_price * 100 if position_side == "long" else (max_price_seen - entry_price) / entry_price * 100
                    dynamic_target_optimizer.record_trade(
                        "ranging" if entry_type in ("Ping", "Pong", "OU-Ping", "OU-Pong") else "trend",
                        entry_volatility,
                        max_excursion,
                        max_drawdown,
                        pnl_pct
                    )
                    
                    position_side = None
                    qty = 0.0
                    invested_amount = 0.0
                    has_taken_partial_tp = False
                    min_trail_dist = 0.0
            
            if position_side is None:
                sig, st = "HOLD", ""
                if geo is not None:
                    # learned-geometry decision chain on the last completed bar
                    gsig_bt = str(geo["signal"][idx-1])
                    if gsig_bt in ("BUY", "SELL"):
                        sig, st = gsig_bt, "Geo"
                else:
                    ub, lb = sg[idx-1] + gv[idx-1] * band_mult, sg[idx-1] - gv[idx-1] * band_mult
                    cu = fg[idx-1] > sg[idx-1] and fg[idx-2] <= sg[idx-2]
                    cd = fg[idx-1] < sg[idx-1] and fg[idx-2] >= sg[idx-2]
                    cb = cvd_g[idx-1] > cvd_g[idx-2]
                    cbe = cvd_g[idx-1] < cvd_g[idx-2]

                    ml = ((sg[idx-1] - lb) / lb * 100) >= margin if lb > 0 else False
                    ms = ((ub - sg[idx-1]) / sg[idx-1] * 100) >= margin if sg[idx-1] > 0 else False

                    if not is_r and cu and cb and hyp_direction == 1:
                        sig, st = "BUY", "Trend"
                    elif not is_r and cd and cbe and hyp_direction == -1:
                        sig, st = "SELL", "Trend"
                    elif is_r and ou.is_valid:
                        ou_sig, ou_type = ou.get_signal(price)
                        if ou_sig != "HOLD":
                            sig, st = ou_sig, ou_type
                    elif is_r and ml and l[idx-1] < lb and c[idx-1] > lb:
                        sig, st = "BUY", "Ping"
                    elif is_r and ms and h[idx-1] > ub and c[idx-1] < ub:
                        sig, st = "SELL", "Pong"
                    
                # SELL opens a short only in geo mode (futures/two-sided); legacy
                # spot SELL stays flat as before.
                open_long = sig == "BUY" and balance >= 5.0
                open_short = sig == "SELL" and geo is not None and balance >= 5.0
                if open_long or open_short:
                    position_side = "long" if open_long else "short"
                    entry_type = st

                    slippage_factor = self.slippage_rate + self.spread_rate / 2
                    entry_price = price * (1 + slippage_factor) if open_long else price * (1 - slippage_factor)

                    entry_volatility = float(gv[idx-1])
                    regime_key = "ranging" if entry_type in ("Ping", "Pong", "OU-Ping", "OU-Pong") else "trend"
                    current_tp_percent, current_sl_percent = dynamic_target_optimizer.get_optimal_targets(
                        regime_key, entry_volatility,
                        default_tp=tp_percent if regime_key == "trend" else 0.3,
                        default_sl=tp_percent if regime_key == "trend" else 0.3
                    )
                    min_trail_dist = 0.0
                    if geo is not None:
                        # cost-floored, vol-scaled barrier targets from the pipeline;
                        # trailing may never come closer than half the target
                        current_tp_percent = float(geo["tp"][idx-1]) if "tp" in geo else max(current_tp_percent, 3.0 * roundtrip_cost_pct())
                        current_sl_percent = float(geo["sl"][idx-1]) if "sl" in geo else max(current_sl_percent, 1.5 * roundtrip_cost_pct())
                        min_trail_dist = entry_price * current_tp_percent / 100.0 * 0.5
                    max_price_seen = entry_price
                    min_price_seen = entry_price

                    stop_dist = gv[idx-1] * (ping_stop_mult if entry_type in ("Ping", "Pong", "OU-Ping", "OU-Pong") else trail_mult)
                    if stop_dist <= 0: stop_dist = entry_price * 0.01

                    risk_amount = balance * (TARGET_RISK_PERCENT / 100.0)
                    target_qty = risk_amount / stop_dist
                    max_qty = (balance * MAX_CAPITAL_ALLOCATION) / entry_price
                    qty = min(target_qty, max_qty)
                    invested_amount = qty * entry_price

                    balance -= invested_amount * self.commission_rate
                    has_taken_partial_tp = False
                    trail_dist_init = max(gv[idx-1] * trail_mult, min_trail_dist)
                    if position_side == "long":
                        trail_stop_30 = entry_price - trail_dist_init
                        trail_stop_70 = entry_price - trail_dist_init
                        ping_stop = entry_price - gv[idx-1] * ping_stop_mult
                    else:
                        trail_stop_30 = entry_price + trail_dist_init
                        trail_stop_70 = entry_price + trail_dist_init
                        ping_stop = entry_price + gv[idx-1] * ping_stop_mult

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

class BacktestOptimizer:
    def __init__(self, df):
        self.df = df
        
    def run_wfo(self):
        L = len(self.df)
        window_size = int(L * 0.40)
        n_slices = 10
        step = int((L - window_size) / (n_slices - 1)) if n_slices > 1 else 0
        
        grid_fast = [5, 8, 12]
        grid_slow = [18, 21, 28]
        grid_band = [2.0, 2.5, 3.0]
        grid_margin = [0.2, 0.3, 0.4]
        
        combos = []
        for f in grid_fast:
            for s in grid_slow:
                for b in grid_band:
                    for m in grid_margin:
                        combos.append({
                            "FAST_LENGTH": f,
                            "SLOW_LENGTH": s,
                            "BAND_MULT": b,
                            "MIN_PROFIT_MARGIN": m,
                            "VOL_LENGTH": 14,
                            "CVD_LENGTH": 14,
                            "TRAIL_MULT": 3.0,
                            "PING_STOP_MULT": 0.5,
                            "TP_PERCENT": 3.0
                        })
                        
        best_candidate = None
        best_stability_count = -1
        lowest_variance = float('inf')
        candidate_reports = []
        
        slices = []
        for i in range(n_slices):
            start = i * step
            end = start + window_size
            is_end = start + int(0.70 * window_size)
            slices.append({
                "is_df": self.df.iloc[start:is_end].copy(),
                "oos_df": self.df.iloc[is_end:end].copy()
            })
            
        for combo in combos:
            slice_pfs = []
            slice_calmars = []
            stable_slices = 0
            
            for s_idx, sl in enumerate(slices):
                bt = Backtester(sl["oos_df"])
                res = bt.run(combo)
                
                pf = res["profit_factor"]
                calmar = res["calmar_ratio"]
                
                slice_pfs.append(pf)
                slice_calmars.append(calmar)
                
                if pf > 1.20 and calmar > 1.0:
                    stable_slices += 1
                    
            if stable_slices >= 8:
                pf_std = float(np.std(slice_pfs))
                if stable_slices > best_stability_count or (stable_slices == best_stability_count and pf_std < lowest_variance):
                    best_stability_count = stable_slices
                    lowest_variance = pf_std
                    best_candidate = combo
                    best_candidate["slice_pfs"] = slice_pfs
                    best_candidate["slice_calmars"] = slice_calmars
                    
            candidate_reports.append({
                "parameters": combo,
                "stable_slices": stable_slices,
                "avg_pf": float(np.mean(slice_pfs))
            })
            
        if best_candidate is None:
            candidate_reports.sort(key=lambda x: (x["stable_slices"], x["avg_pf"]), reverse=True)
            if candidate_reports:
                best_candidate = candidate_reports[0]["parameters"]
                best_candidate["slice_pfs"] = []
                best_candidate["slice_calmars"] = []
                best_stability_count = candidate_reports[0]["stable_slices"]
                
        return {
            "challenger": best_candidate,
            "stability_count": best_stability_count,
            "variance": lowest_variance if lowest_variance != float('inf') else 0.0,
            "slices_evaluated": n_slices
        }
app = Flask("QuantDesktopApp")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quant Bot V3.6 - Learned Geometry</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="/static/lightweight-charts.js"></script>
    <style>
        :root {
            --bg-color: #131722;
            --panel-bg: #1e222d;
            --border-color: #2a2e39;
            --text-main: #d1d4dc;
            --text-muted: #787b86;
            --tv-green: #2962ff;
            --profit-green: #089981;
            --loss-red: #f23645;
            --warning-yellow: #f5b041;
            --paper-blue: #00bcd4;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; user-select: none; }
        body { background-color: var(--bg-color); color: var(--text-main); height: 100vh; overflow: hidden; display: flex; flex-direction: column; }

        header { background-color: var(--panel-bg); border-bottom: 1px solid var(--border-color); padding: 8px 20px; display: flex; justify-content: space-between; align-items: center; flex-shrink: 0; }
        .header-title { font-size: 1rem; font-weight: 600; display: flex; align-items: center; gap: 8px; }
        .controls { display: flex; gap: 12px; align-items: center; }
        .mode-toggle { display: flex; background: rgba(0,0,0,0.2); border-radius: 6px; padding: 2px; border: 1px solid var(--border-color); }
        .mode-btn { padding: 5px 10px; border: none; background: transparent; color: var(--text-muted); font-size: 0.75rem; font-weight: 600; cursor: pointer; border-radius: 4px; transition: 0.2s; }
        .mode-btn.paper-active { background: rgba(0, 188, 212, 0.2); color: var(--paper-blue); border: 1px solid rgba(0, 188, 212, 0.5); }
        .mode-btn.real-active { background: rgba(242, 54, 69, 0.2); color: var(--loss-red); border: 1px solid rgba(242, 54, 69, 0.5); }

        .status-badge { padding: 5px 10px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; display: flex; align-items: center; gap: 6px; border: 1px solid;}
        .status-active { background-color: rgba(8, 153, 129, 0.1); color: var(--profit-green); border-color: rgba(8, 153, 129, 0.3);}
        .status-paused { background-color: rgba(245, 176, 65, 0.1); color: var(--warning-yellow); border-color: rgba(245, 176, 65, 0.3);}
        .dot-blink { width: 7px; height: 7px; border-radius: 50%; animation: blink 1.5s infinite; }
        .bg-green { background-color: var(--profit-green); }
        .bg-yellow { background-color: var(--warning-yellow); }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

        .btn { padding: 6px 14px; border-radius: 4px; font-weight: 600; cursor: pointer; border: none; font-size: 0.8rem; transition: 0.2s; }
        .btn-start { background-color: var(--profit-green); color: white; }
        .btn-stop { background-color: var(--loss-red); color: white; }

        /* Metrics Bar */
        .metrics-bar { background-color: var(--panel-bg); border-bottom: 1px solid var(--border-color); padding: 6px 20px; display: flex; gap: 20px; align-items: center; overflow-x: auto; flex-shrink: 0; }
        .m-item { display: flex; flex-direction: column; gap: 1px; min-width: 80px; }
        .m-label { color: var(--text-muted); font-size: 0.65rem; font-weight: 500; text-transform: uppercase; }
        .m-val { font-size: 0.9rem; font-weight: 600; }
        .color-green { color: var(--profit-green); }
        .color-red { color: var(--loss-red); }
        .color-blue { color: var(--paper-blue); }

        /* Main Layout */
        .main-content { display: grid; grid-template-columns: 1fr 280px; flex: 1; min-height: 0; }
        .chart-area { display: flex; flex-direction: column; border-right: 1px solid var(--border-color); overflow: hidden; }
        .tf-bar { display: flex; gap: 6px; padding: 6px 15px; align-items: center; border-bottom: 1px solid var(--border-color); background: var(--panel-bg); flex-shrink: 0; }
        .tf-btn { background: var(--bg-color); color: var(--text-muted); border: 1px solid var(--border-color); padding: 3px 8px; border-radius: 3px; font-size: 0.75rem; cursor: pointer; }
        .tf-btn.active { background: rgba(41, 98, 255, 0.15); color: var(--tv-green); border-color: rgba(41, 98, 255, 0.5); }
        .chart-wrap { flex: 1; position: relative; min-height: 0; overflow: hidden; }
        #tv-chart-container { position: absolute; top: 0; left: 0; right: 0; bottom: 0; }

        /* Right Panel */
        .right-panel { background-color: var(--panel-bg); display: flex; flex-direction: column; gap: 0; overflow-y: auto; }
        .panel-box { border-bottom: 1px solid var(--border-color); padding: 10px 12px; }
        .box-title { font-size: 0.7rem; color: var(--text-muted); margin-bottom: 8px; font-weight: 600; text-transform: uppercase; display: flex; justify-content: space-between;}
        .kv-row { display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 0.8rem; }
        .kv-key { color: var(--text-muted); }
        .kv-val { font-weight: 500; }

        /* Orderbook */
        .ob-container { display: flex; flex-direction: column; gap: 1px; font-size: 0.75rem; font-family: 'Roboto Mono', monospace; font-weight: 500;}
        .ob-row { display: flex; justify-content: space-between; padding: 1px 4px; position: relative; z-index: 1;}
        .ob-row span { z-index: 2; position: relative; }
        .ob-price-ask { color: var(--loss-red); }
        .ob-price-bid { color: var(--profit-green); }
        .ob-qty { color: var(--text-main); }
        .ob-bg-ask { position: absolute; right: 0; top: 0; height: 100%; background: rgba(242, 54, 69, 0.12); z-index: 0; }
        .ob-bg-bid { position: absolute; right: 0; top: 0; height: 100%; background: rgba(8, 153, 129, 0.12); z-index: 0; }
        .ob-spread { text-align: center; font-weight: 700; padding: 4px 0; border-top: 1px solid var(--border-color); border-bottom: 1px solid var(--border-color); margin: 2px 0; font-size: 0.85rem;}

        .regime-box { text-align: center; padding: 6px; border-radius: 4px; font-weight: 700; font-size: 0.8rem; border: 1px solid; background: rgba(0,0,0,0.2); margin-top: 4px; }
        .regime-trend { color: var(--profit-green); border-color: rgba(8,153,129,0.3); }
        .regime-range { color: var(--warning-yellow); border-color: rgba(245,176,65,0.3); }

        /* Trades Table */
        .trades-area { background-color: var(--panel-bg); border-top: 1px solid var(--border-color); height: 160px; min-height: 160px; max-height: 160px; display: flex; flex-direction: column; flex-shrink: 0; overflow: hidden; }
        .trades-area h3 { padding: 6px 20px; font-size: 0.8rem; border-bottom: 1px solid var(--border-color); color: var(--text-muted); display: flex; justify-content: space-between;}
        .trades-table-wrapper { flex: 1; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 0.75rem; }
        th, td { padding: 4px 20px; text-align: left; border-bottom: 1px solid var(--border-color); }
        th { color: var(--text-muted); font-weight: 500; background-color: rgba(0,0,0,0.1); }
        .badge-long { background-color: rgba(8, 153, 129, 0.15); color: var(--profit-green); padding: 1px 5px; border-radius: 3px; font-weight: 600;}
        .badge-short { background-color: rgba(242, 54, 69, 0.15); color: var(--loss-red); padding: 1px 5px; border-radius: 3px; font-weight: 600;}
        .price-up { color: var(--profit-green); }
        .price-down { color: var(--loss-red); }
        .price-neutral { color: var(--text-main); }
    </style>
</head>
<body>
    <header>
        <div class="header-title"><i class="fa-solid fa-robot" style="color: #2962ff;"></i> Quant Bot V3.6 — Learned Geometry</div>
        <div class="mode-toggle">
            <button id="btn-mode-paper" class="mode-btn paper-active" onclick="setTradingMode('PAPER')"><i class="fa-solid fa-flask"></i> PAPER</button>
            <button id="btn-mode-real" class="mode-btn" onclick="setTradingMode('REAL')"><i class="fa-solid fa-fire"></i> REAL</button>
        </div>
        <div class="controls">
            <div id="status-badge" class="status-badge status-paused"><div id="status-dot" class="dot-blink bg-yellow"></div><span id="status-text">OBSERVATION</span></div>
            <button id="toggle-btn" class="btn btn-start" onclick="toggleBot()">&#9654; BEGIN TRADE</button>
        </div>
    </header>

    <div class="metrics-bar">
        <div class="m-item"><span class="m-label">Balance</span><span class="m-val color-blue" id="st-balance">$10,000</span></div>
        <div class="m-item"><span class="m-label">Net Profit</span><span class="m-val" id="st-net-profit">0.00%</span></div>
        <div class="m-item"><span class="m-label">Win Rate</span><span class="m-val" id="st-win-rate">0.00%</span></div>
        <div class="m-item"><span class="m-label">Profit Factor</span><span class="m-val" id="st-profit-factor">0.00</span></div>
        <div class="m-item"><span class="m-label">Sharpe</span><span class="m-val" id="st-sharpe">0.00</span></div>
        <div class="m-item"><span class="m-label">OBI</span><span class="m-val" id="st-obi" style="font-weight:700;">0.00</span></div>
        <div class="m-item"><span class="m-label">Avg W/L</span><span class="m-val" id="st-avg-wl" style="font-size:0.8rem;">0% / 0%</span></div>
        <div class="m-item"><span class="m-label">Cons W/L</span><span class="m-val" id="st-cons" style="font-size:0.8rem;">0 / 0</span></div>
    </div>

    <div class="main-content">
        <div class="chart-area">
            <div class="tf-bar">
                <span style="font-size:0.85rem; font-weight:600; margin-right:8px;">BTC/USDT</span>
                <button class="tf-btn active" onclick="setTF('1m')">1m</button>
                <button class="tf-btn" onclick="setTF('5m')">5m</button>
                <button class="tf-btn" onclick="setTF('15m')">15m</button>
                <button class="tf-btn" onclick="setTF('1h')">1h</button>
                <button class="tf-btn" onclick="setTF('4h')">4h</button>
                <div style="flex:1;"></div>
                <div id="current-price" class="price-neutral" style="font-size:1.3rem; font-weight:700;">$0.00</div>
            </div>
            <div class="chart-wrap">
                <div id="tv-chart-container"></div>
                <div id="chart-legend" style="position: absolute; top: 10px; left: 10px; z-index: 10; background: rgba(30, 34, 45, 0.85); border: 1px solid var(--border-color); padding: 6px 10px; border-radius: 4px; font-size: 0.72rem; color: var(--text-main); pointer-events: none; display: flex; flex-direction: column; gap: 3px;">
                    <div><span style="color:#089981; font-weight:700;">▮</span> Conformal A ≥ kapı</div>
                    <div><span style="color:#787b86; font-weight:700;">▮</span> Conformal A &lt; kapı</div>
                    <div><span style="color:#f5b041; font-weight:700;">●</span> Epizot başlangıcı (δ≠0, A-küme)</div>
                    <div><span style="color:#089981; font-weight:700;">▲</span> GEO Long &nbsp;·&nbsp; <span style="color:#e040fb; font-weight:700;">▼</span> GEO Short &nbsp;·&nbsp; <span style="color:#f23645; font-weight:700;">▼</span> Exit</div>
                    <div id="legend-geo-status" style="color:#787b86;">Geometri: -</div>
                </div>
            </div>
        </div>

        <div class="right-panel">
            <div class="panel-box">
                <div class="box-title"><span><i class="fa-solid fa-list"></i> Order Book</span><span style="color:var(--profit-green); font-size:0.65rem;">LIVE</span></div>
                <div style="display:flex; justify-content:space-between; color:var(--text-muted); font-size:0.65rem; padding:0 4px; margin-bottom:3px;"><span>Price (USDT)</span><span>Qty (BTC)</span></div>
                <div id="ob-asks" class="ob-container"></div>
                <div id="ob-spread" class="ob-spread price-neutral">$0.00</div>
                <div id="ob-bids" class="ob-container"></div>
            </div>

            <div class="panel-box">
                <div class="box-title">Active Position</div>
                <div id="pos-info" style="text-align:center; color:var(--text-muted); padding:4px 0; font-size:0.8rem;">No Open Position</div>
            </div>

            <div class="panel-box">
                <div class="box-title">Signal & Regime</div>
                <div class="kv-row"><span class="kv-key">Signal:</span><span class="kv-val" id="signal-val">HOLD</span></div>
                <div id="regime-box" class="regime-box">LOADING...</div>
            </div>

            <div class="panel-box">
                <div class="box-title">Learned Geometry <span id="geom-status" style="font-size:0.65rem; color:var(--warning-yellow);">COLLECTING</span></div>
                <div class="kv-row"><span class="kv-key">Schema M:</span><span class="kv-val" id="geom-schema" style="font-size:0.68rem;">-</span></div>
                <div class="kv-row"><span class="kv-key">κ (learned/init):</span><span class="kv-val" id="geom-kappa">- / -</span></div>
                <div class="kv-row"><span class="kv-key">η (speed):</span><span class="kv-val" id="geom-eta">-</span></div>
                <div class="kv-row"><span class="kv-key">P(δ≠0):</span><span class="kv-val" id="geom-pdelta">-</span></div>
                <div class="kv-row"><span class="kv-key">Episode:</span><span class="kv-val" id="geom-episode">NORMAL</span></div>
                <div class="kv-row"><span class="kv-key">Direction:</span><span class="kv-val" id="geom-dir">FLAT</span></div>
                <div class="kv-row"><span class="kv-key">p GBM / Meta:</span><span class="kv-val" id="geom-p">- / -</span></div>
                <div class="kv-row"><span class="kv-key">Conformal A:</span><span class="kv-val" id="geom-a">-</span></div>
                <div class="kv-row"><span class="kv-key">E[net] · TP/SL:</span><span class="kv-val" id="geom-expnet" style="font-size:0.7rem;">-</span></div>
                <div class="kv-row"><span class="kv-key">Overfit gap:</span><span class="kv-val" id="geom-overfit">-</span></div>
                <div class="kv-row"><span class="kv-key">D_anchor:</span><span class="kv-val" id="geom-danchor">-</span></div>
                <div class="kv-row"><span class="kv-key">δ̂ (5/15/30/60):</span><span class="kv-val" id="geom-dhat" style="font-size:0.68rem;">-</span></div>
                <div class="kv-row"><span class="kv-key">r·√|κ| (5/15/30/60):</span><span class="kv-val" id="geom-reff" style="font-size:0.68rem;">-</span></div>
                <div class="kv-row"><span class="kv-key">Heldout Recon:</span><span class="kv-val" id="geom-recon">-</span></div>
                <div class="kv-row"><span class="kv-key">Anchor Panel:</span><span class="kv-val" id="geom-panel" style="font-size:0.7rem;">-</span></div>
                <div class="kv-row"><span class="kv-key">Fold:</span><span class="kv-val" id="geom-fold">0</span></div>
                <div class="kv-row" style="margin-top:6px; border-top:1px dashed var(--border-color); padding-top:6px;">
                    <span class="kv-key">Short (Futures):</span>
                    <button id="short-toggle" onclick="toggleShort()" class="btn" style="padding:2px 8px; font-size:0.68rem; background:#2a2e39; color:var(--text-muted);">OFF</button>
                </div>
                <div style="font-size:0.6rem; color:var(--text-muted); margin-top:3px;">REAL futures short otomatik açılmaz — önce PAPER/backtest.</div>
            </div>

            <div class="panel-box">
                <div class="box-title">Champion Params (Live) <span style="font-size:0.65rem; color:var(--profit-green);" id="champion-version">v1</span></div>
                <div id="champion-params" style="font-size:0.72rem; color:var(--text-main); display:grid; grid-template-columns:1fr 1fr; gap:2px 8px;">
                    <!-- JS populated -->
                </div>
            </div>

            <div class="panel-box">
                <div class="box-title">Shadow Mode & WFO</div>
                <div class="kv-row"><span class="kv-key">Challenger:</span><span class="kv-val color-blue" id="shadow-challenger-title">None</span></div>
                <div id="challenger-params" style="font-size:0.7rem; color:var(--text-muted); display:grid; grid-template-columns:1fr 1fr; gap:2px 8px; margin-bottom:6px; display:none;">
                    <!-- JS populated -->
                </div>
                <div class="kv-row"><span class="kv-key">Shadow Bal:</span><span class="kv-val color-green" id="shadow-balance">$1000.00</span></div>
                <div class="kv-row"><span class="kv-key">Shadow PnL:</span><span class="kv-val" id="shadow-pnl">$0.00</span></div>
                <div class="kv-row"><span class="kv-key">Shadow Pos:</span><span class="kv-val" id="shadow-position">None</span></div>
                
                <div style="display:flex; flex-direction:column; gap:6px; margin-top:8px;">
                    <button class="btn" style="background:#2a2e39; color:white; font-size:0.7rem; padding:4px;" onclick="runBacktest()" id="btn-backtest">Run Backtest (3000b)</button>
                    <button class="btn" style="background:#2a2e39; color:white; font-size:0.7rem; padding:4px;" onclick="runWFO()" id="btn-wfo">Run WFO (10 slices)</button>
                    <button class="btn" style="background:var(--tv-green); color:white; font-size:0.7rem; padding:4px; display:none;" onclick="promoteChallenger()" id="btn-promote">Promote Challenger</button>
                </div>
            </div>

            <div class="panel-box" id="wfo-report-box" style="display:none; font-size:0.75rem;">
                <div class="box-title">WFO / Backtest Report</div>
                <div id="wfo-report-content"></div>
            </div>
        </div>
    </div>

    <div class="trades-area">
        <h3><span>Trade History</span> <span id="th-mode-label" style="font-size:0.65rem; color:var(--paper-blue);">[PAPER]</span></h3>
        <div class="trades-table-wrapper">
            <table>
                <thead><tr><th>Time</th><th>Strategy</th><th>Side</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Reason</th></tr></thead>
                <tbody id="trades-body"><tr><td colspan="7" style="text-align:center;">No trades yet</td></tr></tbody>
            </table>
        </div>
    </div>

    <script>
        let tvChart = null;
        let candleSeries = null;
        let aHist = null;          // Conformal A histogram (learned-geometry overlay)
        let lastPrice = 0;
        let activeTF = '1m';

        function initChart() {
            const container = document.getElementById('tv-chart-container');
            tvChart = LightweightCharts.createChart(container, {
                layout: { textColor: '#d1d4dc', background: { type: 'solid', color: '#131722' } },
                grid: { vertLines: { color: 'rgba(42, 46, 57, 0.3)' }, horzLines: { color: 'rgba(42, 46, 57, 0.3)' } },
                crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
                rightPriceScale: { borderColor: 'rgba(197, 203, 206, 0.4)' },
                timeScale: { borderColor: 'rgba(197, 203, 206, 0.4)', timeVisible: true, secondsVisible: false },
                width: container.clientWidth,
                height: container.clientHeight
            });

            candleSeries = tvChart.addCandlestickSeries({
                upColor: '#089981', downColor: '#f23645', borderVisible: false,
                wickUpColor: '#089981', wickDownColor: '#f23645'
            });

            // Learned-geometry overlay: Conformal A score as a bottom histogram pane
            aHist = tvChart.addHistogramSeries({
                priceScaleId: 'geo',
                priceLineVisible: false,
                lastValueVisible: false,
                priceFormat: { type: 'price', precision: 2, minMove: 0.01 }
            });
            tvChart.priceScale('geo').applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });

            new ResizeObserver(entries => {
                const { width, height } = entries[0].contentRect;
                tvChart.applyOptions({ width, height });
            }).observe(container);
        }

        function toggleBot() {
            fetch('/api/toggle_bot', {method: 'POST'}).then(r => r.json()).then(() => updateUI());
        }

        function toggleShort() {
            if (!window._shortOn) {
                if(!confirm("SHORT sinyallerini aç (futures / iki yönlü)?\n\nPAPER'da ve backtest'te tam simüle edilir. REAL modda canlı futures short OTOMATİK AÇILMAZ — önce doğrula.")) return;
            }
            fetch('/api/toggle_short', {method: 'POST'}).then(r => r.json()).then(() => updateUI());
        }

        function setTF(tf) {
            fetch('/api/set_timeframe', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({tf: tf}) })
            .then(() => {
                activeTF = tf;
                document.querySelectorAll('.tf-btn').forEach(b => { b.classList.remove('active'); if(b.innerText === tf) b.classList.add('active'); });
                // Timeframe değişince eski grafik verilerini temizle
                if (candleSeries && tvChart) {
                    tvChart.removeSeries(candleSeries);
                    tvChart.removeSeries(aHist);
                    candleSeries = tvChart.addCandlestickSeries({
                        upColor: '#089981', downColor: '#f23645', borderVisible: false,
                        wickUpColor: '#089981', wickDownColor: '#f23645'
                    });
                    aHist = tvChart.addHistogramSeries({
                        priceScaleId: 'geo', priceLineVisible: false,
                        lastValueVisible: false,
                        priceFormat: { type: 'price', precision: 2, minMove: 0.01 }
                    });
                    tvChart.priceScale('geo').applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
                }
            });
        }

        function setTradingMode(mode) {
            if (mode === 'REAL') {
                if(!confirm("WARNING: Switching to REAL TRADING mode. This will use real USDT from your MEXC account. Are you sure?")) return;
            }
            fetch('/api/set_trading_mode', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({mode: mode}) })
            .then(() => updateUI());
        }

        function updateUI() {
            fetch('/api/state')
                .then(r => r.json())
                .then(s => {
                    // Mode
                    const pBtn = document.getElementById('btn-mode-paper');
                    const rBtn = document.getElementById('btn-mode-real');
                    if (s.trading_mode === "PAPER") {
                        pBtn.className = "mode-btn paper-active"; rBtn.className = "mode-btn";
                        document.getElementById('st-balance').innerText = "$" + s.virtual_balance.toFixed(2);
                        document.getElementById('th-mode-label').innerText = "[PAPER]";
                    } else {
                        pBtn.className = "mode-btn"; rBtn.className = "mode-btn real-active";
                        document.getElementById('st-balance').innerText = "$" + s.real_balance.toFixed(2);
                        document.getElementById('th-mode-label').innerText = "[REAL]";
                    }

                    // Header state
                    const btn = document.getElementById('toggle-btn');
                    const sBadge = document.getElementById('status-badge');
                    const sText = document.getElementById('status-text');
                    const sDot = document.getElementById('status-dot');
                    if (s.is_trading_active) {
                        btn.className = "btn btn-stop"; btn.innerHTML = "&#9632; STOP";
                        sBadge.className = "status-badge status-active"; sText.innerText = "LIVE TRADING"; sDot.className = "dot-blink bg-green";
                    } else {
                        btn.className = "btn btn-start"; btn.innerHTML = "&#9654; BEGIN TRADE";
                        sBadge.className = "status-badge status-paused"; sText.innerText = "OBSERVATION"; sDot.className = "dot-blink bg-yellow";
                    }

                    // Metrics
                    const winRate = s.trade_count > 0 ? (s.winning_trades / s.trade_count * 100) : 0;
                    const pf = s.gross_loss < 0 ? Math.abs(s.gross_profit / s.gross_loss) : (s.gross_profit > 0 ? 99.99 : 0);
                    document.getElementById('st-net-profit').innerText = (s.total_pnl >= 0 ? "+" : "") + s.total_pnl.toFixed(2) + "%";
                    document.getElementById('st-net-profit').className = "m-val " + (s.total_pnl >= 0 ? "color-green" : "color-red");
                    document.getElementById('st-win-rate').innerText = winRate.toFixed(1) + "%";
                    document.getElementById('st-profit-factor').innerText = pf.toFixed(2);
                    document.getElementById('st-sharpe').innerText = s.sharpe_ratio.toFixed(2);
                    document.getElementById('st-sharpe').className = "m-val " + (s.sharpe_ratio >= 0 ? "color-green" : "color-red");
                    document.getElementById('st-obi').innerText = s.obi > 0 ? "+" + s.obi.toFixed(2) : s.obi.toFixed(2);
                    document.getElementById('st-obi').className = "m-val " + (s.obi > 0.3 ? "color-green" : (s.obi < -0.3 ? "color-red" : "price-neutral"));
                    document.getElementById('st-avg-wl').innerHTML = '<span class="color-green">+' + s.avg_win.toFixed(2) + '%</span> / <span class="color-red">' + s.avg_loss.toFixed(2) + '%</span>';
                    document.getElementById('st-cons').innerHTML = '<span class="color-green">' + s.max_cons_wins + '</span> / <span class="color-red">' + s.max_cons_losses + '</span>';

                    // Price
                    const cpDiv = document.getElementById('current-price');
                    if (s.price > lastPrice) cpDiv.className = "price-up";
                    else if (s.price < lastPrice) cpDiv.className = "price-down";
                    else cpDiv.className = "price-neutral";
                    cpDiv.innerText = "$" + s.price.toFixed(2);
                    lastPrice = s.price;

                    // Signal & Regime with Hyperbolic trend confirmation indicators
                    let hypText = "";
                    if (s.hyp_direction === 1) {
                        hypText = " <span style='color:var(--profit-green); font-weight:bold;'>▲ Hyp</span>";
                    } else if (s.hyp_direction === -1) {
                        hypText = " <span style='color:var(--loss-red); font-weight:bold;'>▼ Hyp</span>";
                    }
                    
                    let sigText = s.signal;
                    if (s.signal_type) sigText += " (" + s.signal_type + ")";
                    document.getElementById('signal-val').innerHTML = sigText + hypText;
                    document.getElementById('signal-val').style.color = s.signal === 'BUY' ? 'var(--profit-green)' : (s.signal === 'SELL' ? 'var(--loss-red)' : 'var(--text-main)');
                    
                    // Regime box: geometric episode state once the pipeline is ready,
                    // legacy trend/ranging only during warm-up
                    if (s.geom && s.geom.status === 'ready') {
                        const inEp = s.geom.episode === 'EPISODE';
                        document.getElementById('regime-box').innerText = inEp
                            ? ('EPİZOT' + (s.geom.cluster >= 0 ? ' (A' + s.geom.cluster + ')' : ''))
                            : 'NORMAL (GEO)';
                        document.getElementById('regime-box').className = "regime-box " + (inEp ? "regime-range" : "regime-trend");
                    } else {
                        document.getElementById('regime-box').innerText = s.regime.toUpperCase();
                        document.getElementById('regime-box').className = "regime-box " + (s.is_ranging ? "regime-range" : "regime-trend");
                    }

                    // Learned Geometry panel (V3.6)
                    if (s.geom) {
                        const g = s.geom;
                        const gst = document.getElementById('geom-status');
                        gst.innerText = (g.status || 'collecting').toUpperCase();
                        gst.style.color = g.status === 'ready' ? 'var(--profit-green)' : 'var(--warning-yellow)';
                        document.getElementById('geom-schema').innerText = g.schema || '-';
                        document.getElementById('geom-kappa').innerText = (g.kappa || 0).toFixed(3) + ' / ' + (g.kappa_init || 0).toFixed(3);
                        document.getElementById('geom-eta').innerText = (g.eta || 0).toFixed(3);
                        const gd = g.diag || {};
                        document.getElementById('geom-pdelta').innerText = (gd.p_delta_nonzero !== undefined) ? (gd.p_delta_nonzero * 100).toFixed(1) + '%' : '-';
                        const epEl = document.getElementById('geom-episode');
                        if (g.episode === 'EPISODE') {
                            epEl.innerText = 'EPISODE' + (g.cluster >= 0 ? ' (A' + g.cluster + ')' : '');
                            epEl.style.color = 'var(--warning-yellow)';
                        } else {
                            epEl.innerText = 'NORMAL';
                            epEl.style.color = 'var(--profit-green)';
                        }
                        document.getElementById('geom-p').innerText = (g.p_gbm !== undefined ? g.p_gbm.toFixed(2) : '-') + ' / ' + (g.p_meta !== undefined ? g.p_meta.toFixed(2) : '-');
                        const aEl = document.getElementById('geom-a');
                        aEl.innerText = (g.a_score || 0).toFixed(2) + ' (gate ' + (g.a_gate !== undefined ? g.a_gate.toFixed(2) : '0.50') + ')';
                        aEl.style.color = (g.a_score || 0) >= (g.a_gate || 0.5) ? 'var(--profit-green)' : 'var(--text-muted)';
                        const enEl = document.getElementById('geom-expnet');
                        if (g.exp_net !== undefined && g.tp_pct) {
                            enEl.innerText = (g.exp_net >= 0 ? '+' : '') + g.exp_net.toFixed(2) + '% · ' +
                                g.tp_pct.toFixed(2) + '/' + g.sl_pct.toFixed(2) + '%';
                            enEl.style.color = g.exp_net > 0 ? 'var(--profit-green)' : 'var(--loss-red)';
                        } else {
                            enEl.innerText = '-';
                            enEl.style.color = 'var(--text-muted)';
                        }
                        document.getElementById('geom-danchor').innerText = (gd.d_anchor !== undefined) ? gd.d_anchor.toFixed(4) : '-';
                        const resKeys = ['5', '15', '30', '60'];
                        const dh = g.delta_hat || {};
                        document.getElementById('geom-dhat').innerText = resKeys.every(k => dh[k] !== undefined)
                            ? resKeys.map(k => dh[k].toFixed(2)).join(' / ') : '-';
                        const re = gd.r_eff || {};
                        document.getElementById('geom-reff').innerText = resKeys.every(k => re[k] !== undefined)
                            ? resKeys.map(k => re[k].toFixed(2)).join(' / ') : '-';
                        document.getElementById('geom-recon').innerText = (gd.heldout_recon !== undefined) ? gd.heldout_recon.toFixed(4) : '-';
                        const pn = g.panel || {};
                        document.getElementById('geom-panel').innerText = (pn.core_active !== undefined)
                            ? (pn.core_active + ' act / ' + (pn.core_retired || 0) + ' ret')
                            : '-';
                        document.getElementById('geom-fold').innerText = g.fold || 0;

                        // Direction badge
                        const dirEl = document.getElementById('geom-dir');
                        if (g.dir === 1) { dirEl.innerText = '▲ LONG'; dirEl.style.color = 'var(--profit-green)'; }
                        else if (g.dir === -1) { dirEl.innerText = '▼ SHORT'; dirEl.style.color = 'var(--loss-red)'; }
                        else { dirEl.innerText = 'FLAT'; dirEl.style.color = 'var(--text-muted)'; }

                        // Overfit gap (train-OOS AUC)
                        const ofEl = document.getElementById('geom-overfit');
                        if (gd.overfit_gap !== undefined) {
                            ofEl.innerText = gd.overfit_gap.toFixed(3) + ' (OOS ' + (gd.oos_auc !== undefined ? gd.oos_auc.toFixed(2) : '-') + ')';
                            ofEl.style.color = gd.overfit_gap > 0.15 ? 'var(--loss-red)' : (gd.overfit_gap > 0.08 ? 'var(--warning-yellow)' : 'var(--profit-green)');
                        } else { ofEl.innerText = '-'; ofEl.style.color = 'var(--text-muted)'; }

                        // Short toggle button reflects backend state
                        const stEl = document.getElementById('short-toggle');
                        if (g.allow_short) { stEl.innerText = 'ON'; stEl.style.background = 'rgba(242,54,69,0.2)'; stEl.style.color = 'var(--loss-red)'; }
                        else { stEl.innerText = 'OFF'; stEl.style.background = '#2a2e39'; stEl.style.color = 'var(--text-muted)'; }

                        document.getElementById('legend-geo-status').innerText = 'Geometri: ' + (g.status || '-').toUpperCase() +
                            (g.status === 'ready' && g.schema ? ' · ' + g.schema : '');
                    }

                    // Position
                    const pInfo = document.getElementById('pos-info');
                    if (s.position_side) {
                        pInfo.innerHTML = '<div class="kv-row"><span class="kv-key">Side:</span><span class="' + (s.position_side === 'long' ? 'badge-long' : 'badge-short') + '">' + s.position_side.toUpperCase() + '</span></div>' +
                            '<div class="kv-row"><span class="kv-key">Entry:</span><span class="kv-val">$' + s.position_entry.toFixed(2) + '</span></div>' +
                            '<div class="kv-row" style="margin-top:3px; border-top:1px dashed var(--border-color); padding-top:3px;"><span class="kv-key">PnL:</span><span class="kv-val ' + (s.position_pnl >= 0 ? 'color-green' : 'color-red') + '">' + (s.position_pnl >= 0 ? '+' : '') + s.position_pnl.toFixed(2) + '%</span></div>';
                    } else {
                        pInfo.innerHTML = '<div style="text-align:center; color:var(--text-muted); padding:4px 0; font-size:0.8rem;">No Open Position</div>';
                    }

                    // Champion Params
                    const champPStore = s.parameters_store || {};
                    document.getElementById('champion-version').innerText = 'v' + (champPStore.active_version || 1);
                    const activeP = s.active_parameters || {};
                    document.getElementById('champion-params').innerHTML = Object.entries(activeP).map(([k, v]) => 
                        '<div><span class="kv-key">' + k.replace('_LENGTH', '').replace('_MULT', '') + ':</span> <span class="kv-val">' + v + '</span></div>'
                    ).join('');

                    // Shadow challenger & buttons
                    const challengerP = champPStore.shadow_challenger;
                    const cTitle = document.getElementById('shadow-challenger-title');
                    const cParams = document.getElementById('challenger-params');
                    const promoteBtn = document.getElementById('btn-promote');
                    if (challengerP) {
                        cTitle.innerText = "Active Challenger";
                        cTitle.className = "kv-val color-blue";
                        cParams.style.display = "grid";
                        cParams.innerHTML = Object.entries(challengerP).map(([k, v]) => 
                            '<div><span class="kv-key">' + k.replace('_LENGTH', '').replace('_MULT', '') + ':</span> <span class="kv-val">' + v + '</span></div>'
                        ).join('');
                        
                        if (!s.position_side) {
                            promoteBtn.style.display = "block";
                        } else {
                            promoteBtn.style.display = "none";
                        }
                    } else {
                        cTitle.innerText = "None";
                        cTitle.className = "kv-val color-red";
                        cParams.style.display = "none";
                        promoteBtn.style.display = "none";
                    }

                    // Shadow metrics
                    document.getElementById('shadow-balance').innerText = "$" + s.shadow_balance.toFixed(2);
                    document.getElementById('shadow-pnl').innerText = (s.shadow_total_pnl >= 0 ? "+" : "") + s.shadow_total_pnl.toFixed(2) + " USDT (" + s.shadow_trade_count + " trades)";
                    document.getElementById('shadow-pnl').className = "kv-val " + (s.shadow_total_pnl >= 0 ? "color-green" : "color-red");
                    
                    if (s.shadow_position_side) {
                        document.getElementById('shadow-position').innerHTML = '<span class="' + (s.shadow_position_side === 'long' ? 'badge-long' : 'badge-short') + '">' + s.shadow_position_side.toUpperCase() + '</span> @ $' + s.shadow_position_entry.toFixed(2) + ' (' + (s.shadow_position_pnl >= 0 ? '+' : '') + s.shadow_position_pnl.toFixed(2) + '%)';
                    } else {
                        document.getElementById('shadow-position').innerText = "None";
                    }

                    // Task spinners
                    const btBtn = document.getElementById('btn-backtest');
                    if (s.backtest_running) {
                        btBtn.innerText = "Running Backtest...";
                        btBtn.disabled = true;
                    } else {
                        btBtn.innerText = "Run Backtest (3000b)";
                        btBtn.disabled = false;
                    }

                    const wfoBtn = document.getElementById('btn-wfo');
                    if (s.wfo_running) {
                        wfoBtn.innerText = "Running WFO Grid...";
                        wfoBtn.disabled = true;
                    } else {
                        wfoBtn.innerText = "Run WFO (10 slices)";
                        wfoBtn.disabled = false;
                    }

                    // Reports display
                    const wfoBox = document.getElementById('wfo-report-box');
                    const wfoContent = document.getElementById('wfo-report-content');
                    if (s.wfo_report) {
                        wfoBox.style.display = "block";
                        let wfoHtml =
                            '<div style="font-weight:600; color:var(--paper-blue); margin-bottom:4px;">WFO Result (Challenger Chosen):</div>' +
                            '<div class="kv-row"><span class="kv-key">Engine:</span><span class="kv-val" style="font-size:0.65rem;">' + (s.wfo_report.engine || 'legacy-grid') + '</span></div>' +
                            '<div class="kv-row"><span class="kv-key">Stability folds:</span><span class="kv-val">' + s.wfo_report.stability_count + '/' + (s.wfo_report.slices_evaluated || 10) + '</span></div>' +
                            '<div class="kv-row"><span class="kv-key">PF Variance:</span><span class="kv-val">' + s.wfo_report.variance.toFixed(4) + '</span></div>';
                        if (s.wfo_report.schema && s.wfo_report.schema !== '-') {
                            wfoHtml += '<div class="kv-row"><span class="kv-key">Geometry:</span><span class="kv-val" style="font-size:0.65rem;">' + s.wfo_report.schema + '</span></div>';
                        }
                        const ov = s.wfo_report.overfit;
                        if (ov && ov.mean_gap !== undefined) {
                            const ovColor = ov.verdict === 'high' ? 'var(--loss-red)' : (ov.verdict === 'moderate' ? 'var(--warning-yellow)' : 'var(--profit-green)');
                            wfoHtml += '<div class="kv-row"><span class="kv-key">Overfit (IS-OOS AUC):</span><span class="kv-val" style="color:' + ovColor + ';">' +
                                ov.mean_gap.toFixed(3) + ' · OOS ' + ov.mean_oos_auc.toFixed(2) + ' [' + ov.verdict + ']</span></div>';
                        }
                        const dg = (s.wfo_report.diagnostics || []);
                        if (dg.length > 0) {
                            const dl = dg[dg.length - 1];
                            wfoHtml += '<div style="font-size:0.65rem; color:var(--text-muted); margin-top:3px;">' +
                                'P(δ≠0)=' + (dl.p_delta_nonzero * 100).toFixed(1) + '% · κ=' + dl.kappa.toFixed(2) +
                                ' · heldout=' + dl.heldout_recon.toFixed(3) +
                                ' · D_anchor=' + dl.d_anchor.toFixed(4) + '</div>';
                        }
                        wfoHtml += '<div style="font-size:0.65rem; color:var(--text-muted); margin-top:4px; word-break:break-all;">' + JSON.stringify(s.wfo_report.challenger) + '</div>';
                        wfoContent.innerHTML = wfoHtml;
                    } else if (s.backtest_report) {
                        wfoBox.style.display = "block";
                        const r = s.backtest_report;
                        if (r.status === "error") {
                            wfoContent.innerHTML = '<div style="color:var(--loss-red);">' + r.message + '</div>';
                        } else {
                            wfoContent.innerHTML =
                                '<div style="font-weight:600; color:var(--profit-green); margin-bottom:4px;">Backtest Result (' + (r.engine || 'legacy') + '):</div>' +
                                (r.schema && r.schema !== '-' ? '<div class="kv-row"><span class="kv-key">Geometry:</span><span class="kv-val" style="font-size:0.65rem;">' + r.schema + '</span></div>' : '') +
                                '<div class="kv-row"><span class="kv-key">Trades / Win rate:</span><span class="kv-val">' + r.trade_count + ' / ' + r.win_rate.toFixed(1) + '%</span></div>' +
                                '<div class="kv-row"><span class="kv-key">Net Profit:</span><span class="kv-val ' + (r.total_pnl_usdt >= 0 ? "color-green" : "color-red") + '">' + (r.total_pnl_usdt >= 0 ? "+" : "") + r.total_pnl_usdt.toFixed(2) + ' USDT (' + (r.total_pnl_pct >= 0 ? "+" : "") + r.total_pnl_pct.toFixed(2) + '%)</span></div>' +
                                '<div class="kv-row"><span class="kv-key">Profit Factor:</span><span class="kv-val">' + r.profit_factor.toFixed(2) + '</span></div>' +
                                '<div class="kv-row"><span class="kv-key">Calmar / Sharpe:</span><span class="kv-val">' + r.calmar_ratio.toFixed(2) + ' / ' + r.sharpe_ratio.toFixed(2) + '</span></div>' +
                                '<div class="kv-row"><span class="kv-key">Max Drawdown:</span><span class="kv-val color-red">' + r.max_drawdown_pct.toFixed(2) + '%</span></div>';
                        }
                    } else {
                        wfoBox.style.display = "none";
                    }

                    // Trades
                    if (s.trades.length > 0) {
                        document.getElementById('trades-body').innerHTML = s.trades.map(t =>
                            '<tr><td style="color:var(--text-muted);">' + t.time + '</td><td>' + t.type + '</td>' +
                            '<td><span class="' + (t.side === 'LONG' || t.side === 'long' ? 'badge-long' : 'badge-short') + '">' + t.side + '</span></td>' +
                            '<td>$' + parseFloat(t.entry).toFixed(2) + '</td><td>' + (t.exit > 0 ? '$' + parseFloat(t.exit).toFixed(2) : '-') + '</td>' +
                            '<td class="' + (String(t.pnl).startsWith('+') ? 'color-green' : (String(t.pnl).startsWith('-') ? 'color-red' : '')) + '">' + t.pnl + '</td><td>' + t.reason + '</td></tr>'
                        ).join('');
                    }

                    // TradingView Chart — learned-geometry overlay
                    if (!tvChart) initChart();
                    if (s.chart_data && s.chart_data.length > 0) {
                        candleSeries.setData(s.chart_data.map(d => ({ time: d.time, open: d.open, high: d.high, low: d.low, close: d.close })));

                        // Conformal A histogram: yellow = episode, green = A above the gate
                        aHist.setData(s.chart_data
                            .filter(d => d.geo_a !== null && d.geo_a !== undefined)
                            .map(d => ({
                                time: d.time,
                                value: d.geo_a,
                                color: d.geo_episode
                                    ? 'rgba(245, 176, 65, 0.55)'
                                    : (d.geo_a >= d.geo_gate ? 'rgba(8, 153, 129, 0.65)' : 'rgba(120, 123, 134, 0.40)')
                            })));

                        // Markers: episode starts (with anomaly cluster), GEO entries, geometric exits
                        let markers = [];
                        let prevEp = false;
                        s.chart_data.forEach(d => {
                            if (d.geo_episode && !prevEp) {
                                markers.push({
                                    time: d.time, position: 'aboveBar', color: '#f5b041',
                                    shape: 'circle',
                                    text: d.geo_state > 0 ? 'A' + (d.geo_state - 1) : 'EP'
                                });
                            }
                            prevEp = !!d.geo_episode;
                            if (d.geo_buy) {
                                markers.push({ time: d.time, position: 'belowBar', color: '#089981', shape: 'arrowUp', text: 'GEO L' });
                            }
                            if (d.geo_sell) {
                                markers.push({ time: d.time, position: 'aboveBar', color: '#e040fb', shape: 'arrowDown', text: 'GEO S' });
                            }
                            if (d.geo_exit) {
                                markers.push({ time: d.time, position: 'aboveBar', color: '#f23645', shape: 'arrowDown', text: 'EXIT' });
                            }
                        });
                        candleSeries.setMarkers(markers);
                    }

                    // Order Book
                    if (s.orderbook && s.orderbook.bids.length > 0 && s.orderbook.asks.length > 0) {
                        const topAsks = s.orderbook.asks.slice(0, 7).reverse();
                        const topBids = s.orderbook.bids.slice(0, 7);
                        let maxVol = 0;
                        [...topAsks, ...topBids].forEach(arr => { if (arr[1] > maxVol) maxVol = arr[1]; });
                        document.getElementById('ob-asks').innerHTML = topAsks.map(arr => {
                            const pct = Math.min((arr[1] / maxVol) * 100, 100);
                            return '<div class="ob-row"><div class="ob-bg-ask" style="width:' + pct + '%;"></div><span class="ob-price-ask">' + parseFloat(arr[0]).toFixed(2) + '</span><span class="ob-qty">' + parseFloat(arr[1]).toFixed(4) + '</span></div>';
                        }).join('');
                        document.getElementById('ob-bids').innerHTML = topBids.map(arr => {
                            const pct = Math.min((arr[1] / maxVol) * 100, 100);
                            return '<div class="ob-row"><div class="ob-bg-bid" style="width:' + pct + '%;"></div><span class="ob-price-bid">' + parseFloat(arr[0]).toFixed(2) + '</span><span class="ob-qty">' + parseFloat(arr[1]).toFixed(4) + '</span></div>';
                        }).join('');
                        document.getElementById('ob-spread').innerText = "$" + s.price.toFixed(2);
                    }
                }).catch(err => {});
        }

        function runBacktest() {
            fetch('/api/backtest', { method: 'POST' }).then(() => updateUI());
        }

        function runWFO() {
            fetch('/api/run_wfo', { method: 'POST' }).then(() => updateUI());
        }

        function promoteChallenger() {
            if(!confirm("Are you sure you want to promote the shadow challenger parameters to champion (active live parameter set)?")) return;
            fetch('/api/promote_challenger', { method: 'POST' }).then(r => r.json()).then(data => {
                if (data.status === 'success') {
                    alert('Challenger successfully promoted!');
                } else {
                    alert('Promotion failed: ' + data.message);
                }
                updateUI();
            });
        }

        setTimeout(() => { updateUI(); setInterval(updateUI, 500); }, 300);
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
        return jsonify(state_copy)

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

@app.route('/api/backtest', methods=['POST'])
def run_backtest_endpoint():
    global bot_instance
    if not bot_instance:
        return jsonify({"status": "error", "message": "Bot not initialized"}), 400
        
    def _worker():
        try:
            log.info("Backtest worker thread started. Fetching historical OHLCV data...")
            bot_state["backtest_running"] = True
            bot_state["backtest_report"] = None
            
            loop = bot_state.get("loop")
            if loop:
                future = asyncio.run_coroutine_threadsafe(
                    bot_instance.fetch_ohlcv_large(SYMBOL, bot_state["timeframe"], 3000),
                    loop
                )
                df = future.result()
            else:
                import asyncio as local_asyncio
                df = local_asyncio.run(bot_instance.fetch_ohlcv_large(SYMBOL, bot_state["timeframe"], 3000))
                
            log.info(f"Historical OHLCV data fetched successfully: {len(df)} bars. Running backtest simulation...")
            active_params = get_active_parameters()
            # V3.6: geometric backtest (train on first window, trade the purged
            # remainder); legacy engine only as fallback for short histories.
            try:
                report = run_geometric_backtest(df, active_params, timeframe=bot_state["timeframe"])
            except Exception as e_geo_bt:
                log.error(f"Geometric backtest unavailable ({e_geo_bt}); falling back to legacy engine.")
                report = Backtester(df).run(active_params)
                report["engine"] = "legacy"
            log.info(f"Backtest simulation completed ({report.get('engine','legacy')}). Trade Count: {report['trade_count']}, Net PnL: {report['total_pnl_usdt']:.2f} USDT")

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
                "trades": report["trades"][-10:]
            }
        except Exception as e:
            log.error(f"Backtest API error: {e}")
            bot_state["backtest_report"] = {"status": "error", "message": str(e)}
        finally:
            bot_state["backtest_running"] = False

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"status": "running"})

@app.route('/api/run_wfo', methods=['POST'])
def run_wfo_endpoint():
    global bot_instance
    if not bot_instance:
        return jsonify({"status": "error", "message": "Bot not initialized"}), 400
        
    def _worker():
        try:
            log.info("WFO worker thread started. Fetching historical OHLCV data...")
            bot_state["wfo_running"] = True
            bot_state["wfo_report"] = None
            
            loop = bot_state.get("loop")
            if loop:
                future = asyncio.run_coroutine_threadsafe(
                    bot_instance.fetch_ohlcv_large(SYMBOL, bot_state["timeframe"], 3000),
                    loop
                )
                df = future.result()
            else:
                import asyncio as local_asyncio
                df = local_asyncio.run(bot_instance.fetch_ohlcv_large(SYMBOL, bot_state["timeframe"], 3000))
                
            log.info(f"Historical OHLCV data fetched successfully: {len(df)} bars. Running purged walk-forward...")
            # V3.6: purged walk-forward over the geometric pipeline (warm start +
            # neighbour-fold RKD, geometry schema fixed). Legacy grid as fallback.
            try:
                engine = PurgedWalkForwardEngine(df, timeframe=bot_state["timeframe"])
                result = engine.run()
            except Exception as e_geo_wfo:
                log.error(f"Purged WFO unavailable ({e_geo_wfo}); falling back to legacy grid WFO.")
                result = BacktestOptimizer(df).run_wfo()
                result["engine"] = "legacy-grid"
            log.info(f"WFO completed ({result.get('engine','legacy-grid')}). Challenger: {result['challenger']}, "
                     f"Stability: {result['stability_count']}/{result['slices_evaluated']}")

            challenger = result["challenger"]
            if challenger:
                challenger = {k: v for k, v in challenger.items() if not k.startswith("slice_")}

            p_store = get_all_parameters()
            p_store["shadow_challenger"] = challenger

            with open(PARAMETERS_STORE_PATH, "w", encoding="utf-8") as f:
                import json
                json.dump(p_store, f, indent=4)

            bot_state["parameters_store"] = p_store

            bot_state["wfo_report"] = {
                "stability_count": result["stability_count"],
                "variance": float(result["variance"]),
                "slices_evaluated": result["slices_evaluated"],
                "challenger": challenger,
                "engine": result.get("engine", "legacy-grid"),
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
            bot_state["wfo_running"] = False

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"status": "running"})

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
            
        p_store["history"].append({
            "version": p_store.get("active_version", 1),
            "parameters": p_store.get("champion"),
            "retired_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        p_store["champion"] = shadow
        p_store["shadow_challenger"] = None
        p_store["active_version"] = p_store.get("active_version", 1) + 1
        
        with open(PARAMETERS_STORE_PATH, "w", encoding="utf-8") as f:
            import json
            json.dump(p_store, f, indent=4)
            
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
#   python quant_bot_v35.py --test-comp    # V3.5 component tests only
#   python quant_bot_v35.py --test-safe    # V3.5 safety/execution tests only
#
# The suites don't need extra installs — same NumPy/pandas the bot already uses.
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
    print("Testing pure-NumPy gradient boosting (LightGBM slot)...")
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (600, 8))
    y = ((X[:, 0] + 0.5 * X[:, 1] + 0.1 * rng.normal(size=600)) > 0).astype(float)
    gb = PureGradientBoosting(n_trees=30, depth=3, seed=2).fit(X[:450], y[:450])
    p = gb.predict_proba(X[450:])
    auc = PureGradientBoosting.auc(y[450:], p)
    print(f"  holdout AUC = {auc:.3f}")
    assert auc > 0.85
    assert np.all((p > 0) & (p < 1))


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
    assert PureGradientBoosting.auc(y, p) > 0.9
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
    print("Testing SignalEngine info contract (legacy keys preserved + geom block)...")
    df = _make_synth_df(300)
    eng = SignalEngine(enable_geometry=True)
    info = eng.process(df)
    for key in ("signal", "type", "price", "gauss_vol", "slow_gauss", "upper_band",
                "lower_band", "is_ranging", "fast_gauss", "hyp_direction",
                "ou_theta", "ou_mu", "ou_valid", "geom"):
        assert key in info, f"missing info key: {key}"
    assert info["geom"]["status"] in ("collecting", "training")
    assert info["signal"] in ("BUY", "SELL", "HOLD")
    eng2 = SignalEngine(enable_geometry=False)
    info2 = eng2.process(df)
    assert info2["geom"] == {}
    print("  legacy contract intact, geometry block attached ✓")


# ── V3.5 component tests ─────────────────────────────────────────────────────
def test_ou_component():
    print("Testing OUPingPong...")
    ou = OUPingPong()
    np.random.seed(42)
    prices = [100.0]
    mu = 100.0; theta = 0.1; sigma = 0.2
    for _ in range(150):
        dp = theta * (mu - prices[-1]) + np.random.normal(0, sigma)
        prices.append(prices[-1] + dp)
    prices = np.array(prices)
    ou.fit(prices)
    print(f"Fit results: theta={ou.theta:.4f}, mu={ou.mu:.2f}, half_life={ou.half_life:.2f}, valid={ou.is_valid}")
    print(f"Corridor: lower={ou.ou_lower:.2f}, upper={ou.ou_upper:.2f}")
    sig, sig_type = ou.get_signal(ou.ou_lower - 1.0)
    print(f"Price below lower band signal: {sig} ({sig_type})")
    assert sig == "BUY"


def test_jump_diffusion_component():
    print("Testing Jump Diffusion detection...")
    ou = OUPingPong()
    np.random.seed(42)
    prices = [100.0]
    mu = 100.0; theta = 0.1; sigma = 0.2
    for _ in range(150):
        dp = theta * (mu - prices[-1]) + np.random.normal(0, sigma)
        prices.append(prices[-1] + dp)
    ou.fit(np.array(prices))
    print(f"Clean fit valid: {ou.is_valid}, upper_band: {ou.ou_upper:.2f}")
    prices.append(prices[-1] + 5.0)
    ou.fit(np.array(prices))
    print(f"Post-jump valid (should be False due to cooldown): {ou.is_valid}")
    print(f"Jump detected: {ou.jump_detected}, cooldown: {ou.jump_cooldown}")
    print(f"Jump intensity (lambda): {ou.jump_intensity:.4f}, Jump Mean: {ou.jump_mean:.4f}, Jump Std: {ou.jump_std:.4f}")
    assert ou.jump_intensity > 0.0
    assert ou.jump_mean > 0.0
    sig, sig_type = ou.get_signal(90.0)
    print(f"Signal during cooldown (should be HOLD): {sig}")
    assert sig == "HOLD"


def test_rough_path_classifier():
    print("Testing RoughPathClassifier...")
    clf = RoughPathClassifier(window=14)
    np.random.seed(42)
    n = 100
    closes = 100 + np.cumsum(np.random.normal(0, 0.5, n))
    sigs = clf._compute_signatures(closes)
    print(f"Computed signatures shape: {sigs.shape}")
    assert sigs.shape == (n, 9)
    df = pd.DataFrame({'close': closes})
    clf.fit(df)
    pred = clf.predict(closes)
    print(f"Predicted direction: {pred}")
    assert pred in (-1, 0, 1)


def test_dynamic_target_optimizer():
    print("Testing DynamicTargetOptimizer...")
    dto = DynamicTargetOptimizer()
    tp, sl = dto.get_optimal_targets("trend", 1.5, default_tp=0.5, default_sl=0.5)
    print(f"Default targets: tp={tp:.2f}, sl={sl:.2f}")
    assert tp == 0.5
    assert sl == 0.5
    dto.record_trade("trend", 1.0, 1.2, 0.5, 1.0)
    dto.record_trade("trend", 1.1, 1.0, 0.4, 0.8)
    dto.record_trade("trend", 0.9, 1.4, 0.6, 1.2)
    dto.record_trade("trend", 1.0, 1.1, 0.5, 0.9)
    tp, sl = dto.get_optimal_targets("trend", 1.0)
    print(f"Optimal targets for vol 1.0: tp={tp:.4f}, sl={sl:.4f}")
    assert abs(tp - 0.94) < 1e-5
    assert abs(sl - 0.60) < 1e-5


def test_backtester_and_wfo():
    print("Testing Backtester and BacktestOptimizer...")
    np.random.seed(42)
    n = 500
    closes = 100 + np.cumsum(np.random.normal(0, 0.5, n))
    highs = closes + np.random.uniform(0.05, 0.2, n)
    lows = closes - np.random.uniform(0.05, 0.2, n)
    opens = closes + np.random.uniform(-0.1, 0.1, n)
    volumes = np.random.uniform(10, 100, n)
    df = pd.DataFrame({'open': opens, 'high': highs, 'low': lows,
                       'close': closes, 'volume': volumes})
    params = {"FAST_LENGTH": 8, "SLOW_LENGTH": 21, "VOL_LENGTH": 14, "CVD_LENGTH": 14,
              "BAND_MULT": 2.5, "MIN_PROFIT_MARGIN": 0.3, "TRAIL_MULT": 3.0,
              "PING_STOP_MULT": 0.5, "TP_PERCENT": 3.0}
    bt = Backtester(df)
    report = bt.run(params)
    print(f"Backtest run complete: Trade Count={report['trade_count']}, Net PnL={report['total_pnl_usdt']:.2f}")
    assert "trade_count" in report
    assert "total_pnl_pct" in report
    assert "total_pnl_usdt" in report
    assert "profit_factor" in report
    optimizer = BacktestOptimizer(df)
    wfo_res = optimizer.run_wfo()
    print(f"WFO run complete: Challenger={wfo_res['challenger']}, Stability={wfo_res['stability_count']}/10")
    assert "challenger" in wfo_res
    assert "stability_count" in wfo_res


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
    info = {
        'signal': 'SELL', 'type': 'Trend', 'gauss_vol': 200.0, 'slow_gauss': 60000.0,
        'upper_band': 60500.0, 'lower_band': 59500.0, 'is_ranging': False, 'fast_gauss': 60000.0,
        'sg_list': [60000.0] * 100, 'ub_list': [60500.0] * 100, 'lb_list': [59500.0] * 100,
        'hyp_direction': 0, 'ou_theta': 0.0, 'ou_mu': 0.0, 'ou_half_life': 999.0,
        'ou_upper': 0.0, 'ou_lower': 0.0, 'ou_stop_lower': 0.0, 'ou_stop_upper': 0.0,
        'ou_valid': False, 'ou_jump_intensity': 0.0, 'ou_jump_mean': 0.0,
        'ou_jump_std': 0.0, 'ou_jump_detected': False, 'ou_jump_cooldown': 0
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
    info['signal'] = 'BUY'
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
    info['signal'] = 'BUY'
    info['type'] = 'Ping'
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
    info['signal'] = 'BUY'
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
    test_meta_and_conformal()
    test_barrier_directional()
    test_two_sided_and_short_gate()
    test_backtester_short()
    test_overfit_diagnostic()
    test_anchor_panel()
    test_pipeline_end_to_end()
    test_geometric_backtest_and_purged_wfo()
    test_signal_engine_contract()
    print("\nAll V3.6 learned-geometry tests passed!")


def _run_components_suite():
    test_ou_component()
    test_jump_diffusion_component()
    test_rough_path_classifier()
    test_dynamic_target_optimizer()
    test_backtester_and_wfo()
    print("All component tests passed!")


def _run_safety_suite():
    asyncio.run(_run_safety_tests_async())


def _run_all_tests():
    _run_geom_suite()
    print()
    _run_components_suite()
    print()
    _run_safety_suite()


bot_instance = None

def run_bot():
    global bot_instance
    bot_instance = QuantBot()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(bot_instance.main_loop())
    except Exception as e:
        log.error(f"Bot main loop error: {e}")
    finally:
        try:
            # Cancel all pending tasks gracefully
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        loop.close()

if __name__ == "__main__":
    # ── CLI: --test / --test-geom / --test-comp / --test-safe run the embedded
    # test suites and exit. No flag = launch the desktop bot as before.
    _flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if _flags & {"--test", "--test-all"}:
        _run_all_tests(); sys.exit(0)
    if "--test-geom" in _flags:
        _run_geom_suite(); sys.exit(0)
    if "--test-comp" in _flags:
        _run_components_suite(); sys.exit(0)
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

    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    webview.create_window(title='Quant Bot V3.6 - Learned Geometry', url=app, width=1400, height=850, resizable=True, min_size=(1100, 700))
    webview.start()