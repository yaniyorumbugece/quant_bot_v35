import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
from quant_bot_v35 import HyperbolicClassifier, OUPingPong

def test_ou():
    print("Testing OUPingPong...")
    ou = OUPingPong()
    # Generate some mean reverting synthetic data (AR(1) process)
    np.random.seed(42)
    prices = [100.0]
    mu = 100.0
    theta = 0.1
    sigma = 0.2
    for _ in range(150):
        dp = theta * (mu - prices[-1]) + np.random.normal(0, sigma)
        prices.append(prices[-1] + dp)
    
    prices = np.array(prices)
    ou.fit(prices)
    print(f"Fit results: theta={ou.theta:.4f}, mu={ou.mu:.2f}, half_life={ou.half_life:.2f}, valid={ou.is_valid}")
    print(f"Corridor: lower={ou.ou_lower:.2f}, upper={ou.ou_upper:.2f}")
    
    # Check signal
    sig, sig_type = ou.get_signal(ou.ou_lower - 1.0)
    print(f"Price below lower band signal: {sig} ({sig_type})")
    assert sig == "BUY"

def test_jump_diffusion():
    print("Testing Jump Diffusion detection...")
    ou = OUPingPong()
    np.random.seed(42)
    prices = [100.0]
    mu = 100.0
    theta = 0.1
    sigma = 0.2
    for _ in range(150):
        dp = theta * (mu - prices[-1]) + np.random.normal(0, sigma)
        prices.append(prices[-1] + dp)
    
    # Fit clean data
    ou.fit(np.array(prices))
    print(f"Clean fit valid: {ou.is_valid}, upper_band: {ou.ou_upper:.2f}")
    
    # Append a massive spike (jump)
    prices.append(prices[-1] + 5.0)  # huge jump relative to sigma=0.2
    ou.fit(np.array(prices))
    print(f"Post-jump valid (should be False due to cooldown): {ou.is_valid}")
    print(f"Jump detected: {ou.jump_detected}, cooldown: {ou.jump_cooldown}")
    print(f"Jump intensity (lambda): {ou.jump_intensity:.4f}, Jump Mean: {ou.jump_mean:.4f}, Jump Std: {ou.jump_std:.4f}")
    
    assert ou.jump_intensity > 0.0
    assert ou.jump_mean > 0.0
    
    # Check that signals are blocked during cooldown
    sig, sig_type = ou.get_signal(90.0)  # extremely low price
    print(f"Signal during cooldown (should be HOLD): {sig}")
    assert sig == "HOLD"

def test_rough_path_classifier():
    print("Testing RoughPathClassifier...")
    from quant_bot_v35 import RoughPathClassifier
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
    from quant_bot_v35 import DynamicTargetOptimizer
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
    
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    params = {
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
    
    from quant_bot_v35 import Backtester, BacktestOptimizer
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

if __name__ == "__main__":
    test_ou()
    test_jump_diffusion()
    test_rough_path_classifier()
    test_dynamic_target_optimizer()
    test_backtester_and_wfo()
    print("All component tests passed!")
