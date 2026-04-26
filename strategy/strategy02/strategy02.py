from decimal import Decimal
import numpy as np
import pybroker
import talib
from pybroker import Strategy, StrategyConfig, ExecContext
import KLineForm as klf
from strategy.dataTransfer.DataTransfer import ZszqDataSource

# 全局状态存储（每个标的独立）
_state = {}


def trend_rsi_strategy(ctx: ExecContext):
    # 参数设置
    LOOKBACK = 150
    RSI_BUY_THRESHOLD = 25
    RSI_SELL_THRESHOLD = 75
    TRAILING_STOP_PCT = Decimal("0.08")
    POSITION_RISK = Decimal("0.2")
    MAX_POSITION_PCT = Decimal("0.6")

    if ctx.bars < LOOKBACK:
        return

    # 初始化当前标的状态
    symbol = ctx.symbol
    if symbol not in _state:
        _state[symbol] = {'highest_since_entry': Decimal("0")}
    state = _state[symbol]

    # 数据转换
    close = np.asarray(ctx.close, dtype=float)
    open_ = np.asarray(ctx.open, dtype=float)
    high = np.asarray(ctx.high, dtype=float)
    low = np.asarray(ctx.low, dtype=float)
    volume = np.asarray(ctx.volume, dtype=float)
    cur_price = ctx.close[-1]
    cur_price_f = float(cur_price)
    pos = ctx.long_pos()

    # 更新持仓期间最高价
    if pos is not None:
        if cur_price > state['highest_since_entry']:
            state['highest_since_entry'] = cur_price

    # 技术指标
    ma20_list = talib.MA(close, 20)
    if len(ma20_list) < 2:
        return
    ma20 = np.array([float(v) for v in ma20_list])
    ma20_up = ma20[-1] > ma20[-2]
    price_above_ma20 = cur_price_f > ma20[-1]
    trend_ok = ma20_up and price_above_ma20

    rsi14_list = talib.RSI(close, 14)
    rsi14_val = float(rsi14_list[-1]) if len(rsi14_list) > 0 else 50.0

    # 卖出逻辑
    if pos is not None:
        sell_volume = 0
        # 1. 趋势破坏
        if (len(ma20) >= 2 and ma20[-1] < ma20[-2]) or (cur_price_f < ma20[-1]):
            sell_volume = pos.shares
        # 2. 移动止损
        elif state['highest_since_entry'] > 0:
            drawdown = (state['highest_since_entry'] - cur_price) / state['highest_since_entry']
            if drawdown >= TRAILING_STOP_PCT:
                sell_volume = pos.shares
        # 3. RSI 超买减仓
        elif rsi14_val > RSI_SELL_THRESHOLD:
            sell_volume = int(pos.shares / 3)

        if sell_volume >= 100:
            ctx.sell_shares = min(sell_volume, pos.shares)
            if ctx.sell_shares == pos.shares:
                state['highest_since_entry'] = Decimal("0")
            return

    # 买入逻辑
    if not trend_ok:
        return
    if not klf.buy.is_rsi_oversold(close, threshold=RSI_BUY_THRESHOLD):
        return
    bullish = (klf.buy.is_bullish_engulfing(open_, high, low, close) or
               klf.buy.is_morning_star(open_, high, low, close) or
               klf.buy.is_piercing(open_, high, low, close) or
               klf.buy.is_three_white_soldiers(open_, high, low, close))
    if not bullish:
        return

    available_cash = ctx.calc_available_cash()
    target_value = available_cash * POSITION_RISK
    current_position_value = pos.shares * cur_price if pos is not None else Decimal("0")
    total_value = ctx.calc_total_value()
    if current_position_value + target_value > total_value * MAX_POSITION_PCT:
        return

    shares = int(target_value / (cur_price * Decimal("1.001")) // 100 * 100)
    if shares >= 100:
        ctx.buy_shares = shares
        state['highest_since_entry'] = cur_price


# ========== 回测设置 ==========
config = StrategyConfig(
    initial_cash=1_000_000,
    fee_amount=0.0001,
    position_mode=pybroker.PositionMode.LONG_ONLY
)
strategy = Strategy(ZszqDataSource(), "2020-01-01", "2026-03-20", config)
strategy.add_execution(trend_rsi_strategy, ["sh603787"])
result = strategy.backtest(timeframe="1d", adjust="")

# 输出结果
print(result)

# 保存交易记录
if result.trades is not None and not result.trades.empty:
    result.trades.to_csv('trades_log.csv', index=False, encoding='utf-8-sig')
    print("交易记录已保存至 trades_log.csv")

if result.orders is not None and not result.orders.empty:
    result.orders.to_csv('orders_log.csv', index=False, encoding='utf-8-sig')
    print("订单记录已保存至 orders_log.csv")

if result.portfolio is not None and not result.portfolio.empty:
    result.portfolio.to_csv('portfolio_nav.csv', index=True, encoding='utf-8-sig')
    print("净值曲线已保存至 portfolio_nav.csv")