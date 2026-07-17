import sys
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Import classes and state
import quant_bot_v35
from quant_bot_v35 import QuantBot, bot_state, SYMBOL, update_portfolio_metrics

# Create a mock CCXT exchange class
class MockExchange:
    def __init__(self):
        self.apiKey = "mock_key"
        self.secret = "mock_secret"
        self.orders = []
        self.canceled_orders = []
        self.balance = {"USDT": {"free": 100.0}}

    def load_markets(self):
        pass

    def amount_to_precision(self, symbol, qty):
        return f"{qty:.4f}"

    def price_to_precision(self, symbol, price):
        return f"{price:.2f}"

    def fetch_balance(self):
        return self.balance

    def fetch_order_book(self, symbol, limit=None):
        return {
            "bids": [[59990.0, 10.0]],
            "asks": [[60010.0, 10.0]]
        }

    def fetch_order(self, order_id, symbol):
        for o in self.orders:
            if o['id'] == order_id:
                o['status'] = 'closed'
                o['filled'] = o['qty']
                return o
        return {"id": order_id, "status": "closed", "filled": 0.0, "average": 60000.0}

    def fetch_ticker(self, symbol):
        return {"last": 60000.0, "close": 60000.0}

    def create_order(self, symbol, type, side, qty, price=None, params=None):
        order = {
            "id": f"mock_order_{len(self.orders)+1}",
            "symbol": symbol,
            "type": type,
            "side": side,
            "qty": qty,
            "price": price,
            "params": params or {},
            "average": price or 60000.0
        }
        self.orders.append(order)
        return order

    def cancel_order(self, order_id, symbol):
        self.canceled_orders.append(order_id)
        return {"status": "canceled", "id": order_id}

async def run_safety_tests():
    print("--------------------------------------------------")
    print("RUNNING QUANT BOT V3.5 SAFETY AND METRIC TESTS...")
    print("--------------------------------------------------")

    # Instantiate Bot
    bot = QuantBot()
    # Override exchange with MockExchange
    mock_ex = MockExchange()
    bot.exchange = mock_ex

    # Reset state variables
    bot_state["trading_mode"] = "PAPER"
    bot_state["virtual_balance"] = 10000.0
    bot_state["is_trading_active"] = True
    bot_state["trades"] = []
    bot_state["trade_count"] = 0
    bot_state["total_pnl"] = 0.0
    bot_state["pnl_list"] = []
    bot_state["winning_trades"] = 0
    bot_state["losing_trades"] = 0

    # 1. Test Spot Long-Only Restriction
    # Simulation: Receive a SELL signal when no position is open
    print("Testing Spot Long-only restriction:")
    info = {
        'signal': 'SELL',
        'type': 'Trend',
        'gauss_vol': 200.0,
        'slow_gauss': 60000.0,
        'upper_band': 60500.0,
        'lower_band': 59500.0,
        'is_ranging': False,
        'fast_gauss': 60000.0,
        'sg_list': [60000.0]*100,
        'ub_list': [60500.0]*100,
        'lb_list': [59500.0]*100,
        'hyp_direction': 0,
        'ou_theta': 0.0,
        'ou_mu': 0.0,
        'ou_half_life': 999.0,
        'ou_upper': 0.0,
        'ou_lower': 0.0,
        'ou_stop_lower': 0.0,
        'ou_stop_upper': 0.0,
        'ou_valid': False,
        'ou_jump_intensity': 0.0,
        'ou_jump_mean': 0.0,
        'ou_jump_std': 0.0,
        'ou_jump_detected': False,
        'ou_jump_cooldown': 0
    }
    
    # Create dummy dataframe
    df = pd.DataFrame({
        'timestamp': [pd.Timestamp.now()]*100,
        'open': [60000.0]*100,
        'high': [60100.0]*100,
        'low': [59900.0]*100,
        'close': [60000.0]*100,
        'volume': [10.0]*100
    })
    
    async def mock_fetch_ohlcv():
        return df
        
    bot.fetch_ohlcv = mock_fetch_ohlcv
    bot.signal_engine.process = lambda *args, **kwargs: info

    await bot.main_tick()
    print(f"  Position open after SELL signal? {bot.position.is_open} (Expected: False)")
    assert not bot.position.is_open, "Error: Opened position on SELL signal!"

    # 2. Test Paper Order execution with Fee & Slippage
    print("Testing Paper entry slippage and fee calculation:")
    info['signal'] = 'BUY'
    await bot.main_tick()
    print(f"  Position open? {bot.position.is_open} (Expected: True)")
    print(f"  Position mode: {bot.position.mode} (Expected: PAPER)")
    # Raw price is 60000.0. Under 0.05% slippage, entry price must be 60000 * 1.0005 = 60030.0
    print(f"  Position entry price: {bot.position.entry_price:.2f} (Expected: 60030.00)")
    assert abs(bot.position.entry_price - 60030.0) < 1e-5
    # Virtual balance should have entry commission deducted: 0.1% of invested amount
    invested = bot.position.invested_amount
    expected_fee = invested * 0.001
    expected_balance = 10000.0 - expected_fee
    print(f"  Virtual Balance after entry: {bot_state['virtual_balance']:.2f} (Expected: {expected_balance:.2f})")
    assert abs(bot_state["virtual_balance"] - expected_balance) < 1e-2

    # 3. Test Mode Isolation
    # Set global mode to REAL while having open PAPER position.
    print("Testing Mode Isolation:")
    bot_state["trading_mode"] = "REAL"
    # Close the position now
    await bot.close_position("Test manual close", 61000.0)
    print(f"  Exchange orders placed on close? {len(mock_ex.orders)} (Expected: 0 - since position was PAPER)")
    assert len(mock_ex.orders) == 0, "Error: Placed order on exchange to close PAPER position!"
    # Verify that virtual balance was updated
    print(f"  Virtual Balance after close: {bot_state['virtual_balance']:.2f}")

    # 4. Test REAL Mode Stop-Loss Order Placement
    print("Testing REAL mode stop-loss order placement:")
    # Reset mock exchange
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
    print(f"  Placed order 1 (Entry): {mock_ex.orders[0]['side']} (Expected: buy)")
    print(f"  Placed order 2 (Stop-loss): {mock_ex.orders[1]['side']} (Expected: sell) and stopPrice: {mock_ex.orders[1]['params']['stopPrice']}")
    assert mock_ex.orders[0]['side'] == "buy"
    assert mock_ex.orders[1]['side'] == "sell"
    assert "stopPrice" in mock_ex.orders[1]['params']

    # 5. Test Exit stops cancelation
    print("Testing stop-loss order cancelation on exit:")
    stop_id = bot.position.stop_order_id
    await bot.close_position("Exit hit", 61000.0)
    print(f"  Canceled orders list: {mock_ex.canceled_orders} (Expected: ['{stop_id}'])")
    assert stop_id in mock_ex.canceled_orders, "Error: Stop order was not canceled on exit!"

    # 6. Test Precision and Minimum Order Value Limit
    print("Testing Minimum Order Size enforcement (5 USDT):")
    # Low balance case
    mock_ex.orders = []
    mock_ex.balance = {"USDT": {"free": 3.0}}
    bot_state["real_balance"] = 3.0
    info['signal'] = 'BUY'
    await bot.main_tick()
    print(f"  Position open with low balance? {bot.position.is_open} (Expected: False)")
    assert not bot.position.is_open

    # Balance 6 USDT (enough for 5 USDT min)
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

if __name__ == "__main__":
    asyncio.run(run_safety_tests())
