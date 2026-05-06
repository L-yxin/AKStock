import numpy as np
import pandas as pd
import pybroker
import talib
from pybroker import Strategy, StrategyConfig

# 你的本地模块
from AKshareDataLoader.api import get_stock_info_a_code_name_by_file
from strategy.dataTransfer.DataTransfer import ZszqDataSource

# ==================== 1. 用回测方式获取上证指数信号 ====================
def create_index_signal():
    """回测 sh000001 获取大盘 60 日均线方向，返回 pd.Series，索引为日期"""
    ds = ZszqDataSource()
    config = StrategyConfig(initial_cash=1)
    strategy = Strategy(ds, '2022-01-01', '2026-03-20', config)

    close_prices = []
    def collect(ctx):
        close_prices.append(float(ctx.close[-1]))
    strategy.add_execution(collect, ['sh000001'])
    strategy.backtest(timeframe='1d', adjust='')

    closes = np.array(close_prices, dtype=np.float64)
    ma60 = talib.SMA(closes, timeperiod=60)
    is_up = np.zeros(len(ma60), dtype=bool)
    for i in range(60, len(ma60)):
        if not np.isnan(ma60[i]) and not np.isnan(ma60[i-1]):
            is_up[i] = ma60[i] > ma60[i-1]

    dates = pd.bdate_range(start='2022-01-01', end='2026-03-20')
    if len(dates) > len(is_up):
        dates = dates[:len(is_up)]
    index_signal = pd.Series(is_up, index=dates, name='index_ma60_up')
    return index_signal

INDEX_SIGNAL = create_index_signal()

# ==================== 2. 大盘过滤策略函数 ====================
def macd_strategy_index_filter(ctx: pybroker.context.ExecContext):
    if ctx.bars < 60:
        return

    # 获取当前日期
    cur_date = ctx.date[-1]
    cur_date = pd.Timestamp(cur_date)

    close = np.asarray(ctx.close, dtype=np.float64)
    high  = np.asarray(ctx.high, dtype=np.float64)
    vol   = np.asarray(ctx.volume, dtype=np.float64)
    cur_price = float(close[-1])

    pos = ctx.long_pos()

    # ---- 大盘方向 ----
    try:
        index_up = INDEX_SIGNAL.loc[cur_date]
    except KeyError:
        index_up = INDEX_SIGNAL.asof(cur_date)

    # 大盘向下：强制清仓
    if pos is not None and not index_up:
        ctx.sell_shares = pos.shares
        return

    # ---- 技术指标 ----
    ma20 = talib.SMA(close, timeperiod=20)
    macd, signal, _ = talib.MACD(close, fastperiod=8, slowperiod=16, signalperiod=6)
    vol_ma5 = talib.SMA(vol, timeperiod=5)

    # ---- 持仓风控 ----
    if pos is not None:
        entry = pos.entries[0]
        if hasattr(ctx, 'bars_since_entry'):
            hold_bars = ctx.bars_since_entry(entry)
        elif hasattr(ctx, 'bars_since'):
            hold_bars = ctx.bars_since(entry.date)
        else:
            hold_bars = 1

        # 1) 移动止损：持仓最高点回撤 10%
        if hold_bars > 0:
            highest = high[-hold_bars:].max()
            if cur_price < highest * 0.90:
                ctx.sell_shares = pos.shares
                return

        # 2) 时间止损：持仓 ≥20 日且浮亏
        if hold_bars >= 20:
            avg_cost = float(sum(e.price * e.shares for e in pos.entries) / pos.shares)
            if cur_price < avg_cost:
                ctx.sell_shares = pos.shares
                return

        # 3) MACD 死叉离场
        if macd[-2] >= signal[-2] and macd[-1] < signal[-1]:
            ctx.sell_shares = pos.shares
            return

    # ---- 买入条件 ----
    if pos is None:
        # 大盘必须向上
        if not index_up:
            return
        # 价格在 20 日线上方
        if ma20[-1] is not None and not np.isnan(ma20[-1]):
            if cur_price <= ma20[-1]:
                return
        # MACD 金叉
        if not (macd[-2] <= signal[-2] and macd[-1] > signal[-1]):
            return
        # 成交量过滤：不是地量（ >5日均量的一半 ）
        if vol[-1] < vol_ma5[-1] * 0.5:
            return

        shares = ctx.calc_target_shares(1.0, price=cur_price)
        ctx.buy_shares = (shares // 100) * 100

# ==================== 3. 单线程回测 ====================
stock_info = get_stock_info_a_code_name_by_file()
stock_codes = []
for code in stock_info['code']:
    code_str = str(int(code)).zfill(6)
    symbol = 'sh' + code_str if code_str.startswith('6') else 'sz' + code_str
    stock_codes.append(symbol)

# 若需要快速测试，可只取前 N 只（此行可注释掉跑全部）
# stock_codes = stock_codes[:500]

all_trades = []
all_orders = []
all_portfolio = []

print(f"开始单线程回测，共 {len(stock_codes)} 只股票...")
for idx, symbol in enumerate(stock_codes, 1):
    try:
        ds = ZszqDataSource()
        config = StrategyConfig(initial_cash=1_000_000, fee_amount=0.0001, position_mode=pybroker.PositionMode.LONG_ONLY)
        strategy = Strategy(ds, "2023-01-01", "2026-03-20", config)
        strategy.add_execution(macd_strategy_index_filter, [symbol])
        result = strategy.backtest(timeframe="1d", adjust="")

        if result.trades is not None and not result.trades.empty:
            all_trades.append(result.trades)
        if result.orders is not None and not result.orders.empty:
            all_orders.append(result.orders)
        if result.portfolio is not None and not result.portfolio.empty:
            port = result.portfolio.copy()
            port['symbol'] = symbol
            all_portfolio.append(port)

        if idx % 50 == 0:
            print(f"进度: {idx}/{len(stock_codes)}")
    except Exception as e:
        print(f"回测 {symbol} 失败: {e}")

# ==================== 4. 保存结果 ====================
if all_trades:
    final_trades = pd.concat(all_trades, ignore_index=True)
    final_trades.to_csv('trades_log.csv', index=False, encoding='utf-8-sig')
    print(f"交易记录已保存，总笔数: {len(final_trades)}")
if all_orders:
    pd.concat(all_orders, ignore_index=True).to_csv('orders_log.csv', index=False, encoding='utf-8-sig')
if all_portfolio:
    final_portfolio = pd.concat(all_portfolio, ignore_index=True)
    final_portfolio.reset_index(inplace=True)
    final_portfolio.rename(columns={'index': 'date'}, inplace=True)
    final_portfolio.to_csv('portfolio_nav.csv', index=False, encoding='utf-8-sig')
    print(f"净值曲线已保存，总行数: {len(final_portfolio)}")