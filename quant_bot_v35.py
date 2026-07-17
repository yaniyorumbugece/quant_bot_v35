"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  QUANT BOT V3.1 — Scale-Out (70/30) + MTF Micro-Trailing Stop              ║
║  MEXC Spot · Gaussian Crossover · Order Book Depth · PyWebView              ║
║                                                                            ║
║  Mimarlar: Profesör + Antigravity AI + Kullanıcı                           ║
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
    "parameters_store": get_all_parameters()
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

class SignalEngine:
    def __init__(self):
        self.regime = RoughPathClassifier()
        self.ou = OUPingPong()

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

        def sanitize(v, default=0.0):
            if v is None or math.isnan(v) or math.isinf(v):
                return default
            return float(v)

        return {
            "signal": sig,
            "type": st,
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

    @property
    def is_open(self): return self.side is not None

    def open(self, side, price, qty, sig_type, gauss_vol, invested_amount, ou_target=0.0, ou_stop=0.0, mode="PAPER", stop_order_id=None, params=None, tp_percent=0.3, sl_percent=0.3, entry_volatility=0.0):
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
        
        self.trail_mult = float(params.get("TRAIL_MULT", 3.0))
        self.ping_stop_mult = float(params.get("PING_STOP_MULT", 0.5))
        
        if side == "long":
            self.trail_stop_30 = price - gauss_vol*self.trail_mult
            self.trail_stop_70 = price - gauss_vol*self.trail_mult
            self.ping_stop = price - gauss_vol*self.ping_stop_mult
        else:
            self.trail_stop_30 = price + gauss_vol*self.trail_mult
            self.trail_stop_70 = price + gauss_vol*self.trail_mult
            self.ping_stop = price + gauss_vol*self.ping_stop_mult

    def close(self, reason, price):
        pnl_pct = (price-self.entry_price)/self.entry_price*100 if self.side=="long" else (self.entry_price-price)/self.entry_price*100
        pnl_usdt = self.invested_amount * (pnl_pct / 100)
        self.side = None; self.entry_price = 0; self.qty = 0
        self.has_taken_partial_tp = False
        self.mode = "PAPER"
        self.stop_order_id = None
        self.realized_pnl_usdt = 0.0
        return pnl_pct, pnl_usdt

    def update_stops(self, price, gv_normal, gv_lower):
        self.max_price_seen = max(self.max_price_seen, price)
        self.min_price_seen = min(self.min_price_seen, price)
        if self.side == "long":
            self.trail_stop_30 = max(self.trail_stop_30, price-gv_normal*self.trail_mult)
            if not self.has_taken_partial_tp:
                self.trail_stop_70 = max(self.trail_stop_70, price-gv_lower*self.trail_mult)
            self.ping_stop = max(self.ping_stop, price-gv_normal*self.ping_stop_mult)
        elif self.side == "short":
            self.trail_stop_30 = min(self.trail_stop_30, price+gv_normal*self.trail_mult)
            if not self.has_taken_partial_tp:
                self.trail_stop_70 = min(self.trail_stop_70, price+gv_lower*self.trail_mult)
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
            elif self.entry_type == "Trend":
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
            elif self.entry_type == "Trend":
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

        # Safely pad the indicator list for the final open bar to prevent index errors
        def get_padded_val(lst, index, default=0.0):
            if lst is None or len(lst) == 0: return default
            if index < len(lst): return float(lst[index])
            return float(lst[-1])

        # Robust historical jump detection for chart markers
        jumps_flag = np.zeros(len(df), dtype=bool)
        if len(df) > 30:
            c_vals = df['close'].values
            returns = np.diff(c_vals)
            median_ret = np.median(returns)
            mad = np.median(np.abs(returns - median_ret))
            robust_std = mad * 1.4826 if mad > 1e-8 else np.std(returns)
            if robust_std < 1e-8: robust_std = 1e-8
            jump_threshold = 3.0 * robust_std
            for j in range(1, len(df)):
                ret = float(df['close'].iloc[j] - df['close'].iloc[j-1])
                if abs(ret - median_ret) > jump_threshold:
                    jumps_flag[j] = True

        # OHLC chart data for Lightweight Charts (unix timestamps)
        chart_data = []
        for j in range(len(df)):
            t_val = int(df['timestamp'].iloc[j].timestamp()) + UTC_OFFSET
            chart_data.append({
                "time": t_val,
                "open": float(df['open'].iloc[j]),
                "high": float(df['high'].iloc[j]),
                "low": float(df['low'].iloc[j]),
                "close": float(df['close'].iloc[j]),
                "slow": get_padded_val(info.get('sg_list'), j, price),
                "upper": get_padded_val(info.get('ub_list'), j, price),
                "lower": get_padded_val(info.get('lb_list'), j, price),
                "ou_upper": float(info.get('ou_upper', 0)) if info.get('ou_valid', False) else float(df['close'].iloc[j]),
                "ou_lower": float(info.get('ou_lower', 0)) if info.get('ou_valid', False) else float(df['close'].iloc[j]),
                "jump": bool(jumps_flag[j])
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

        # OBI & Risk Parity filtering (Spot is Long-Only, so BUY only)
        obi_filter_pass = True
        if info['signal'] == "BUY" and bot_state.get("obi", 0) < -0.3:
            obi_filter_pass = False
            log.info(f"BUY blocked by OBI ({bot_state['obi']:.2f} Sell Wall)")

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
                            self.position.open(pos_side, price, filled_qty, info['type'], info['gauss_vol'], invested, ou_target, ou_stop, mode="REAL", stop_order_id=stop_order_id, tp_percent=opt_tp, sl_percent=opt_sl, entry_volatility=entry_vol)
                            
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
                            self.position.open(pos_side, slippage_price, qty, info['type'], info['gauss_vol'], invested, ou_target, ou_stop, mode="PAPER", tp_percent=opt_tp, sl_percent=opt_sl, entry_volatility=entry_vol)

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

        # Shadow Mode Processing
        shadow_challenger = bot_state["parameters_store"].get("shadow_challenger")
        if shadow_challenger:
            bot_state["shadow_active"] = True
            bot_state["shadow_parameters"] = shadow_challenger
            if not hasattr(self, "shadow_signal_engine"):
                self.shadow_signal_engine = SignalEngine()
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
        log.info("QUANT BOT V3.5 - Paper Trading Active")
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

    def run(self, params):
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
        features = rough_path_classifier._compute_signatures(c)
        ou = OUPingPong()
            
        start_idx = max(slow_len + 100, 300)
        if start_idx >= len(self.df):
            return {
                "trade_count": 0, "total_pnl_pct": 0.0, "total_pnl_usdt": 0.0,
                "profit_factor": 1.0, "max_drawdown_pct": 0.0, "calmar_ratio": 0.0,
                "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "recovery_factor": 0.0,
                "win_rate": 0.0, "expectancy": 0.0, "trades": []
            }
            
        rough_path_classifier.fit(self.df.iloc[:start_idx])
            
        for idx in range(start_idx, len(self.df)):
            price = float(c[idx])
            prev_price = float(c[idx-1])
            
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
            
            if position_side is not None:
                max_price_seen = max(max_price_seen, price)
                min_price_seen = min(min_price_seen, price)
                gv_normal = float(gv[idx-1])
                gv_lower = 0.7 * gv_normal
                
                if position_side == "long":
                    trail_stop_30 = max(trail_stop_30, price - gv_normal * trail_mult)
                    if not has_taken_partial_tp:
                        trail_stop_70 = max(trail_stop_70, price - gv_lower * trail_mult)
                    ping_stop = max(ping_stop, price - gv_normal * ping_stop_mult)
                else:
                    trail_stop_30 = min(trail_stop_30, price + gv_normal * trail_mult)
                    if not has_taken_partial_tp:
                        trail_stop_70 = min(trail_stop_70, price + gv_lower * trail_mult)
                    ping_stop = min(ping_stop, price + gv_normal * ping_stop_mult)
                    
                exit_reason = None
                if position_side == "long":
                    if has_taken_partial_tp:
                        if price <= trail_stop_30: exit_reason = "Trail Stop (30%)"
                    elif entry_type in ("Ping", "OU-Ping"):
                        if price <= entry_price * (1 - current_sl_percent/100): exit_reason = "Stop Loss (Ping)"
                        elif price >= entry_price * (1 + current_tp_percent/100): exit_reason = "Ping TP"
                    elif entry_type == "Trend":
                        if price <= entry_price * (1 - current_sl_percent/100): exit_reason = "Stop Loss (Trend)"
                        elif price <= trail_stop_70: exit_reason = "PARTIAL_TP"
                        elif price >= entry_price * (1 + current_tp_percent/100): exit_reason = f"Trend TP ({current_tp_percent:.2f}%)"
                else:
                    if has_taken_partial_tp:
                        if price >= trail_stop_30: exit_reason = "Trail Stop (30%)"
                    elif entry_type in ("Pong", "OU-Pong"):
                        if price >= entry_price * (1 + current_sl_percent/100): exit_reason = "Stop Loss (Pong)"
                        elif price <= entry_price * (1 - current_tp_percent/100): exit_reason = "Pong TP"
                    elif entry_type == "Trend":
                        if price >= entry_price * (1 + current_sl_percent/100): exit_reason = "Stop Loss (Trend)"
                        elif price >= trail_stop_70: exit_reason = "PARTIAL_TP"
                        elif price <= entry_price * (1 - current_tp_percent/100): exit_reason = f"Trend TP ({current_tp_percent:.2f}%)"
                            
                if exit_reason is None:
                    if entry_type in ("Ping", "Pong", "OU-Ping", "OU-Pong") and not is_r:
                        exit_reason = "Acil Cikis (Rejim Degisti)"
                        
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
            
            if position_side is None:
                ub, lb = sg[idx-1] + gv[idx-1] * band_mult, sg[idx-1] - gv[idx-1] * band_mult
                cu = fg[idx-1] > sg[idx-1] and fg[idx-2] <= sg[idx-2]
                cd = fg[idx-1] < sg[idx-1] and fg[idx-2] >= sg[idx-2]
                cb = cvd_g[idx-1] > cvd_g[idx-2]
                cbe = cvd_g[idx-1] < cvd_g[idx-2]
                
                ml = ((sg[idx-1] - lb) / lb * 100) >= margin if lb > 0 else False
                ms = ((ub - sg[idx-1]) / sg[idx-1] * 100) >= margin if sg[idx-1] > 0 else False
                
                sig, st = "HOLD", ""
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
                    
                if sig == "BUY" and balance >= 5.0:
                    position_side = "long"
                    entry_type = st
                    
                    slippage_factor = self.slippage_rate + self.spread_rate / 2
                    entry_price = price * (1 + slippage_factor)
                    
                    entry_volatility = float(gv[idx-1])
                    regime_key = "ranging" if entry_type in ("Ping", "Pong", "OU-Ping", "OU-Pong") else "trend"
                    current_tp_percent, current_sl_percent = dynamic_target_optimizer.get_optimal_targets(
                        regime_key, entry_volatility, 
                        default_tp=tp_percent if regime_key == "trend" else 0.3, 
                        default_sl=tp_percent if regime_key == "trend" else 0.3
                    )
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
                    trail_stop_30 = entry_price - gv[idx-1] * trail_mult
                    trail_stop_70 = entry_price - gv[idx-1] * trail_mult
                    ping_stop = entry_price - gv[idx-1] * ping_stop_mult

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
    <title>Quant Bot V3.5 - TradingView Charts</title>
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
        <div class="header-title"><i class="fa-solid fa-robot" style="color: #2962ff;"></i> Quant Bot V3.5</div>
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
                    <div><span style="color:#f5b041; font-weight:700;">■</span> Gauss Slow (MA)</div>
                    <div><span style="color:#2962ff; font-weight:700;">■</span> Gauss Bands (Volatility)</div>
                    <div><span style="color:#e040fb; font-weight:700;">■</span> OU Corridor (Mean Reversion)</div>
                    <div><span style="color:#ff5722; font-weight:700;">▲</span> JD Jump (Sıçrama)</div>
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
                <div class="box-title">OU Parameters (JD-OU)</div>
                <div class="kv-row"><span class="kv-key">θ (Speed):</span><span class="kv-val" id="ou-theta">0.00</span></div>
                <div class="kv-row"><span class="kv-key">μ (Equil.):</span><span class="kv-val" id="ou-mu">$0.00</span></div>
                <div class="kv-row"><span class="kv-key">Half-Life:</span><span class="kv-val" id="ou-halflife">∞ bars</span></div>
                <div class="kv-row"><span class="kv-key">Jump Intensity:</span><span class="kv-val" id="ou-jump-intensity">0.00%</span></div>
                <div class="kv-row"><span class="kv-key">Jump Mean/Std:</span><span class="kv-val" id="ou-jump-stats">0/0</span></div>
                <div class="kv-row"><span class="kv-key">Jump Cooldown:</span><span class="kv-val" id="ou-cooldown">NO</span></div>
                <div class="kv-row"><span class="kv-key">OU Valid:</span><span class="kv-val" id="ou-valid">NO</span></div>
                <div class="kv-row"><span class="kv-key">Corridor:</span><span class="kv-val" id="ou-corridor" style="font-size:0.7rem;">-</span></div>
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
        let slowLine = null;
        let upperLine = null;
        let lowerLine = null;
        let ouUpperLine = null;
        let ouLowerLine = null;
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

            slowLine = tvChart.addLineSeries({ color: '#f5b041', lineWidth: 2, crosshairMarkerVisible: false, priceLineVisible: false });
            upperLine = tvChart.addLineSeries({ color: '#2962ff', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, crosshairMarkerVisible: false, priceLineVisible: false });
            lowerLine = tvChart.addLineSeries({ color: '#2962ff', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, crosshairMarkerVisible: false, priceLineVisible: false });
            
            // OU Bands (magenta dashed)
            ouUpperLine = tvChart.addLineSeries({ color: '#e040fb', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, crosshairMarkerVisible: false, priceLineVisible: false });
            ouLowerLine = tvChart.addLineSeries({ color: '#e040fb', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, crosshairMarkerVisible: false, priceLineVisible: false });

            new ResizeObserver(entries => {
                const { width, height } = entries[0].contentRect;
                tvChart.applyOptions({ width, height });
            }).observe(container);
        }

        function toggleBot() {
            fetch('/api/toggle_bot', {method: 'POST'}).then(r => r.json()).then(() => updateUI());
        }

        function setTF(tf) {
            fetch('/api/set_timeframe', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({tf: tf}) })
            .then(() => {
                activeTF = tf;
                document.querySelectorAll('.tf-btn').forEach(b => { b.classList.remove('active'); if(b.innerText === tf) b.classList.add('active'); });
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
                    
                    document.getElementById('regime-box').innerText = s.regime.toUpperCase();
                    document.getElementById('regime-box').className = "regime-box " + (s.is_ranging ? "regime-range" : "regime-trend");

                    // OU Parameters panel details
                    document.getElementById('ou-theta').innerText = s.ou_theta.toFixed(4);
                    document.getElementById('ou-mu').innerText = "$" + s.ou_mu.toFixed(2);
                    document.getElementById('ou-halflife').innerText = s.ou_valid ? s.ou_half_life.toFixed(1) + " bars" : "∞ bars";
                    document.getElementById('ou-jump-intensity').innerText = (s.ou_jump_intensity * 100).toFixed(1) + "%";
                    document.getElementById('ou-jump-stats').innerText = s.ou_jump_mean.toFixed(2) + " / " + s.ou_jump_std.toFixed(2);
                    document.getElementById('ou-cooldown').innerText = s.ou_jump_cooldown > 0 ? "YES (" + s.ou_jump_cooldown + "b)" : (s.ou_jump_detected ? "DETECTED" : "NO");
                    document.getElementById('ou-cooldown').style.color = (s.ou_jump_cooldown > 0 || s.ou_jump_detected) ? "var(--loss-red)" : "var(--text-muted)";
                    document.getElementById('ou-valid').innerText = s.ou_valid ? "YES" : "NO";
                    document.getElementById('ou-valid').style.color = s.ou_valid ? "var(--profit-green)" : "var(--text-muted)";
                    
                    if (s.ou_valid) {
                        document.getElementById('ou-corridor').innerHTML = 
                            "<span style='color:var(--paper-blue);'>$" + s.ou_lower.toFixed(2) + "</span> - " +
                            "<span style='color:var(--paper-blue);'>$" + s.ou_upper.toFixed(2) + "</span>";
                    } else {
                        document.getElementById('ou-corridor').innerText = "-";
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
                        wfoContent.innerHTML = 
                            '<div style="font-weight:600; color:var(--paper-blue); margin-bottom:4px;">WFO Result (Challenger Chosen):</div>' +
                            '<div class="kv-row"><span class="kv-key">Stability slices:</span><span class="kv-val">' + s.wfo_report.stability_count + '/10</span></div>' +
                            '<div class="kv-row"><span class="kv-key">PF Variance:</span><span class="kv-val">' + s.wfo_report.variance.toFixed(4) + '</span></div>' +
                            '<div style="font-size:0.65rem; color:var(--text-muted); margin-top:4px; word-break:break-all;">' + JSON.stringify(s.wfo_report.challenger) + '</div>';
                    } else if (s.backtest_report) {
                        wfoBox.style.display = "block";
                        const r = s.backtest_report;
                        if (r.status === "error") {
                            wfoContent.innerHTML = '<div style="color:var(--loss-red);">' + r.message + '</div>';
                        } else {
                            wfoContent.innerHTML = 
                                '<div style="font-weight:600; color:var(--profit-green); margin-bottom:4px;">Backtest Result:</div>' +
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

                    // TradingView Chart
                    if (!tvChart) initChart();
                    if (s.chart_data && s.chart_data.length > 0) {
                        candleSeries.setData(s.chart_data.map(d => ({ time: d.time, open: d.open, high: d.high, low: d.low, close: d.close })));
                        slowLine.setData(s.chart_data.map(d => ({ time: d.time, value: d.slow })));
                        upperLine.setData(s.chart_data.map(d => ({ time: d.time, value: d.upper })));
                        lowerLine.setData(s.chart_data.map(d => ({ time: d.time, value: d.lower })));
                        ouUpperLine.setData(s.chart_data.map(d => ({ time: d.time, value: d.ou_upper })));
                        ouLowerLine.setData(s.chart_data.map(d => ({ time: d.time, value: d.ou_lower })));
                        
                        // Set markers for Jump Diffusion jumps
                        let markers = [];
                        s.chart_data.forEach(d => {
                            if (d.jump) {
                                markers.push({
                                    time: d.time,
                                    position: 'aboveBar',
                                    color: '#ff5722',
                                    shape: 'arrowDown',
                                    text: 'JD Jump'
                                });
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
            bt = Backtester(df)
            report = bt.run(active_params)
            log.info(f"Backtest simulation completed. Trade Count: {report['trade_count']}, Net PnL: {report['total_pnl_usdt']:.2f} USDT")
            
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
                
            log.info(f"Historical OHLCV data fetched successfully: {len(df)} bars. Running WFO grid optimization...")
            optimizer = BacktestOptimizer(df)
            result = optimizer.run_wfo()
            log.info(f"WFO grid optimization completed. Challenger: {result['challenger']}, Stability: {result['stability_count']}/10")
            
            challenger = result["challenger"]
            
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
                "challenger": challenger
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

bot_instance = None

def run_bot():
    global bot_instance
    bot_instance = QuantBot()
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(bot_instance.main_loop())

if __name__ == "__main__":
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
    webview.create_window(title='Quant Bot V3.5 - TradingView', url=app, width=1400, height=850, resizable=True, min_size=(1100, 700))
    webview.start()
