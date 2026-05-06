# coding=utf-8
import pandas as pd
from gm.api import *

def init(context):
    context.symbol = 'SHSE.000001'
    context.lookback = 1
    subscribe(context.symbol, '1d', count=context.lookback,
              fields='open,high,low,close,volume')

    csv_path = 'D:/Users/lyx/Desktop/量化/AKStock/strategy/strategy05/ml_trades_fixed.csv'
    df = pd.read_csv(csv_path)
    df['entry_dt'] = pd.to_datetime(df['entry_date']).dt.date
    df['exit_dt']  = pd.to_datetime(df['exit_date']).dt.date
    df['shares']   = df['shares'].astype(int)

    events = []
    for _, row in df.iterrows():
        events.append({'date': row['entry_dt'], 'action': 'buy', 'shares': int(row['shares'])})
        events.append({'date': row['exit_dt'],  'action': 'sell', 'shares': int(row['shares'])})
    events.sort(key=lambda x: (x['date'], 0 if x['action'] == 'buy' else 1))

    context.events = events
    context.event_idx = 0

def on_bar(context, bars):
    today = context.now.date()
    # 处理当天所有事件
    while context.event_idx < len(context.events):
        evt = context.events[context.event_idx]
        if evt['date'] != today:
            break

        account = context.account()
        if evt['action'] == 'buy':
            price = bars[0].close
            # 用全部可用资金计算可买股数（向下取整100股）
            max_shares = int(account.cash.available / (price * 1.001) / 100) * 100
            buy_shares = max_shares
            if buy_shares >= 100:
                order_volume(symbol=context.symbol, volume=buy_shares,
                             side=OrderSide_Buy, order_type=OrderType_Market,
                             position_effect=PositionEffect_Open)
                print(f'{context.now}: 买入 {buy_shares} 股')
            else:
                print(f'{context.now}: 资金不足，跳过买入')
        else:  # sell
            pos = get_position(context.symbol)
            if pos and pos[0].volume >= 100:
                order_volume(symbol=context.symbol, volume=pos[0].volume,
                             side=OrderSide_Sell, order_type=OrderType_Market,
                             position_effect=PositionEffect_Close)
                print(f'{context.now}: 卖出 {pos[0].volume} 股')
        context.event_idx += 1

def on_backtest_finished(context, indicator):
    print('*' * 50)
    print('LSTM 策略演示回测完成')
    print('最终收益率：{:.2%}'.format(indicator['pnl_ratio']))
    print('最大回撤：  {:.2%}'.format(indicator['max_drawdown']))
    print('*' * 50)

if __name__ == '__main__':
    run(strategy_id='c3cbd239-e2e3-11f0-8396-de46284d0201',
        filename='01.py',
        mode=MODE_BACKTEST,
        token='e6740ed56c26779ea954bb72affe37d85434a479',
        backtest_start_time='2020-01-01 08:00:00',
        backtest_end_time='2026-04-24 15:30:00',
        backtest_adjust=ADJUST_NONE,
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1)