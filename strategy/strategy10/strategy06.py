import numpy as np
import pandas as pd
from datetime import datetime
import pybroker
from pybroker import Strategy, StrategyConfig
from strategy.dataTransfer.DataTransfer import ZszqDataSource
from AKshareDataLoader.api import get_stock_info_a_code_name_by_file
from KLineForm import buy, sell

# ==================== 策略定义 ====================
def rsi_oversold_strategy(ctx):
    """RSI超卖买入，固定止盈止损出场"""
    if ctx.bars < 70:
        return

    close = np.asarray(ctx.close, dtype=np.float64)
    high  = np.asarray(ctx.high, dtype=np.float64)
    low   = np.asarray(ctx.low, dtype=np.float64)
    open_ = np.asarray(ctx.open, dtype=np.float64)

    cur_price = float(close[-1])
    pos = ctx.long_pos()

    # 卖出条件：持仓后按止盈止损
    if pos is not None:
        # 获取持仓均价
        total_cost = sum(e.price * e.shares for e in pos.entries)
        avg_cost = float(total_cost) / float(pos.shares)

        # 5% 止盈
        if cur_price >= avg_cost * 1.02:
            ctx.sell_shares = pos.shares
            return
        # 3% 止损
        if cur_price <= avg_cost * 0.98:
            ctx.sell_shares = pos.shares
            return

    # 买入条件：改进版 RSI 超卖
    if pos is None:
        # 使用改进的超买超卖函数（已自动判断趋势和加速）
        signal = buy.is_rsi_oversold(
            close,
            rsi_periods='6,12,24',
            threshold=27.0,
            lookback=14,
            accel_factor=1.8
        )
        if signal:   # ManagerBoolean 对象，直接可用于 if 判断
            shares = ctx.calc_target_shares(1.0)
            ctx.buy_shares = (shares // 100) * 100


# ==================== 主流程 ====================
def main():
    # 1. 获取股票池（主板，非ST，非科创/创业/北证）
    print("获取股票池...")
    stock_df = get_stock_info_a_code_name_by_file()
    stock_df = stock_df[~stock_df['market'].str.contains('科创|创业|北京|北证', na=False)]
    if 'name' in stock_df.columns:
        stock_df = stock_df[~stock_df['name'].str.contains('ST', case=False, na=False)]

    codes = stock_df['code'].values
    symbols = []
    for code in codes:
        code_str = str(int(code)).zfill(6)
        symbols.append('sh' + code_str if code_str.startswith('6') else 'sz' + code_str)
    print(f"符合条件的股票数量: {len(symbols)}")

    # 测试时可只取前500只（加快速度）
    # symbols = symbols[:500]

    backtest_start = '2020-01-01'
    backtest_end   = '2026-03-20'

    all_trades = []
    all_orders = []
    all_portfolio = []

    for idx, sym in enumerate(symbols):
        print(f"[{idx+1}/{len(symbols)}] 回测 {sym} ...")
        try:
            config = StrategyConfig(initial_cash=1_000_000, fee_amount=0.0001)
            strategy = Strategy(ZszqDataSource(), backtest_start, backtest_end, config)
            strategy.add_execution(rsi_oversold_strategy, [sym])
            result = strategy.backtest(timeframe='1d', adjust='')

            if result.trades is not None and not result.trades.empty:
                metrics = result.metrics
                print(f"  总收益: {metrics.total_return_pct:.2f}% | "
                      f"最大回撤: {metrics.max_drawdown_pct:.2f}% | "
                      f"胜率: {metrics.win_rate:.1f}% | "
                      f"交易次数: {len(result.trades)}")
                all_trades.append(result.trades)
            else:
                print("  无交易")

            if result.orders is not None and not result.orders.empty:
                all_orders.append(result.orders)
            if result.portfolio is not None and not result.portfolio.empty:
                port = result.portfolio.copy()
                port['symbol'] = sym
                all_portfolio.append(port)

        except Exception as e:
            print(f"  回测失败: {e}")

    # 2. 保存结果
    if all_trades:
        final_trades = pd.concat(all_trades, ignore_index=True)
        final_trades.to_csv('trades_rsi_oversold.csv', index=False)
    if all_orders:
        pd.concat(all_orders, ignore_index=True).to_csv('orders_rsi_oversold.csv', index=False)
    if all_portfolio:
        final_portfolio = pd.concat(all_portfolio, ignore_index=True)
        final_portfolio.to_csv('portfolio_rsi_oversold.csv', index=False)

    # 3. 汇总统计
    if all_trades:
        trades = pd.concat(all_trades, ignore_index=True)
        total_pnl = trades['pnl'].sum()
        win_rate = (trades['pnl'] > 0).mean() * 100
        avg_win = trades[trades['pnl'] > 0]['pnl'].mean()
        avg_loss = trades[trades['pnl'] <= 0]['pnl'].mean()
        print("\n========== RSI超卖策略全A回测汇总 ==========")
        print(f"总交易笔数: {len(trades)}")
        print(f"总盈亏: {total_pnl:.2f}")
        print(f"胜率: {win_rate:.2f}%")
        if not pd.isna(avg_win):
            print(f"平均盈利: {avg_win:.2f}")
        else:
            print("平均盈利: N/A")
        if not pd.isna(avg_loss):
            print(f"平均亏损: {avg_loss:.2f}")
        else:
            print("平均亏损: N/A")
        if not pd.isna(avg_win) and not pd.isna(avg_loss) and abs(avg_loss) > 0:
            print(f"盈亏比: {avg_win/abs(avg_loss):.2f}")
        if all_portfolio:
            port_all = pd.concat(all_portfolio, ignore_index=True)
            daily_total = port_all.groupby('date')['market_value'].sum()
            peak = daily_total.cummax()
            dd = (daily_total - peak) / peak
            print(f"组合最大回撤: {dd.min()*100:.2f}%")
            print(f"组合总收益率: {(daily_total.iloc[-1]/daily_total.iloc[0]-1)*100:.2f}%")


if __name__ == '__main__':
    main()