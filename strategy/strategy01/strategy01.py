from decimal import Decimal

import numpy as np
import pybroker
from pybroker import Strategy, StrategyConfig

import KLineForm as klf
from strategy.dataTransfer.DataTransfer import ZszqDataSource


def ma_strategy(ctx: pybroker.context.ExecContext):
    # 确保有足够的数据（至少 20 日均线 + 均线对最大周期 + 缓冲）
    if ctx.bars < 30:
        return

    close = np.asarray(ctx.close)

    cur_price = close[-1]
    pos = ctx.long_pos()

    # ----- 20 日均线方向判断 -----
    if klf.sell.is_moving_average_down(close, ma_periods=[20]):
        if pos is not None:
            ctx.sell_shares = pos.shares
            return  # 清仓后本 bar 不再执行其他信号

    # ----- 正常交易逻辑（仅在 20 日线非向下时执行）-----
    # 买入：均线金叉（默认使用 [5,10] 和 [10,20] 对，任意一对金叉即触发）
    if klf.buy.is_macd_golden_cross(close,"8,16,6"):
        if pos is None:
            shares = ctx.calc_target_shares(1.0, price=cur_price)
            ctx.buy_shares = (shares // 100) * 100

    # 卖出：均线死叉
    elif klf.sell.ma_death_cross(close):
        if pos is not None:
            ctx.sell_shares = pos.shares

    # 止损：价格低于平均成本 5%
    if pos is not None:
        total_cost = sum(entry.price * entry.shares for entry in pos.entries)
        avg_price = total_cost / pos.shares
        if cur_price < avg_price * Decimal("0.95"):
            ctx.sell_shares = pos.shares




# ========== 回测设置 ==========
config = StrategyConfig(initial_cash=1_000_000,fee_amount=0.0001,position_mode=pybroker.PositionMode.LONG_ONLY)
strategy = Strategy(ZszqDataSource(), "2023-01-01", "2026-03-20", config)
strategy.add_execution(ma_strategy, ["sh603787"])
result = strategy.backtest(timeframe="1d", adjust="")

# 输出结果
print(result)

# 1. 保存所有交易记录
trades_df = result.trades          # DataFrame，包含每笔交易的详细信息
if trades_df is not None and not trades_df.empty:
    trades_df.to_csv('trades_log.csv', index=False, encoding='utf-8-sig')
    print("交易记录已保存至 trades_log.csv")

# 2. 保存订单记录（如果有）
orders_df = result.orders
if orders_df is not None and not orders_df.empty:
    orders_df.to_csv('orders_log.csv', index=False, encoding='utf-8-sig')
    print("订单记录已保存至 orders_log.csv")

# 3. 保存净值曲线
portfolio_df = result.portfolio
if portfolio_df is not None and not portfolio_df.empty:
    portfolio_df.to_csv('portfolio_nav.csv', index=True, encoding='utf-8-sig')
    print("净值曲线已保存至 portfolio_nav.csv")
