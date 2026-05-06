import numpy as np
import pandas as pd
import numba
import vectorbt as vbt
import talib
from AKshareDataLoader.api import get_stock_info_a_code_name_by_file
from zszqDataLoader import ZSZQDataLoader

# ==================== 1. 生成上证 60 日线信号 ====================
def create_index_signal(start, end):
    print("计算上证指数 60 日均线方向...")
    loader = ZSZQDataLoader()
    df = loader.select('sh000001', '1d', '', start, end)
    if df.empty:
        raise ValueError("无法获取 sh000001 数据")
    if 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    closes = df['close'].values.astype(np.float64)
    ma60 = talib.SMA(closes, timeperiod=60)
    is_up = np.zeros(len(ma60), dtype=bool)
    for i in range(60, len(ma60)):
        if not np.isnan(ma60[i]) and not np.isnan(ma60[i-1]):
            is_up[i] = ma60[i] > ma60[i-1]
    index_signal = pd.Series(is_up, index=df.index, name='index_ma60_up')
    return index_signal

# ==================== 2. 数据加载 ====================
def load_data_wide(symbols, start, end, loader):
    frames = {}
    for i, sym in enumerate(symbols):
        try:
            df = loader.select(sym, '1d', '', start, end)
            if df is None or df.empty:
                continue
            if 'datetime' in df.columns:
                df['date'] = pd.to_datetime(df['datetime'])
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            else:
                continue
            df = df.set_index('date')
            needed = ['open', 'high', 'low', 'close', 'volume']
            if not all(c in df.columns for c in needed):
                continue
            frames[sym] = df[needed]
        except:
            pass
        if (i + 1) % 200 == 0:
            print(f"已加载 {i+1}/{len(symbols)} 只股票")
    if not frames:
        raise ValueError("未加载到任何数据")
    wide = pd.concat(frames, axis=1, keys=frames.keys()).sort_index()
    return wide

# ==================== 3. Numba 加速 MACD 信号 ====================
@numba.njit(parallel=True, fastmath=True)
def compute_macd_signals(close, vol, fast=8, slow=16, signal=6, vol_ma=20, vol_factor=1.0):
    n_times, n_stocks = close.shape
    entries = np.zeros((n_times, n_stocks), dtype=np.bool_)
    exits   = np.zeros((n_times, n_stocks), dtype=np.bool_)

    for j in numba.prange(n_stocks):
        c = close[:, j]
        v = vol[:, j]
        ema_fast = np.empty(n_times, dtype=np.float64)
        ema_slow = np.empty(n_times, dtype=np.float64)
        sig_line = np.empty(n_times, dtype=np.float64)
        vol_ema  = np.empty(n_times, dtype=np.float64)

        ema_fast[0] = c[0]
        ema_slow[0] = c[0]
        sig_line[0] = 0.0
        vol_ema[0] = v[0]

        k_fast = 2.0 / (fast + 1)
        k_slow = 2.0 / (slow + 1)
        k_sig  = 2.0 / (signal + 1)
        k_vol  = 2.0 / (vol_ma + 1)

        for i in range(1, n_times):
            ema_fast[i] = c[i] * k_fast + ema_fast[i-1] * (1 - k_fast)
            ema_slow[i] = c[i] * k_slow + ema_slow[i-1] * (1 - k_slow)
            vol_ema[i]  = v[i] * k_vol + vol_ema[i-1] * (1 - k_vol)

        macd = ema_fast - ema_slow
        sig_line[0] = macd[0]
        for i in range(1, n_times):
            sig_line[i] = macd[i] * k_sig + sig_line[i-1] * (1 - k_sig)

        for i in range(1, n_times):
            if macd[i-1] <= sig_line[i-1] and macd[i] > sig_line[i]:
                if v[i] > vol_ema[i] * vol_factor:
                    entries[i, j] = True
            elif macd[i-1] >= sig_line[i-1] and macd[i] < sig_line[i]:
                exits[i, j] = True
    return entries, exits

# ==================== 4. 主回测流程 ====================
def main():
    print("获取股票池...")
    stock_df = get_stock_info_a_code_name_by_file()
    stock_df = stock_df[~stock_df['market'].str.contains('科创|创业|北京|北证', na=False)]
    if 'name' in stock_df.columns:
        stock_df = stock_df[~stock_df['name'].str.contains('ST', case=False, na=False)]

    symbols = []
    for code in stock_df['code']:
        c = str(int(code)).zfill(6)
        symbols.append('sh' + c if c.startswith('6') else 'sz' + c)
    print(f"符合条件的股票数量: {len(symbols)}")

    # 测试时可限制数量，全量时注释掉下一行
    symbols = symbols[:500]

    print("加载数据...")
    loader = ZSZQDataLoader()
    start = '2020-01-01'
    end   = '2026-03-20'
    wide_df = load_data_wide(symbols, start, end, loader)
    print("数据形状:", wide_df.shape)

    close_df = wide_df.xs('close', axis=1, level=1)
    vol_df   = wide_df.xs('volume', axis=1, level=1)

    # 本次跳过流动性过滤，保留所有股票，用原始数据
    # 如果想保留过滤，可取消下面注释，并修正阈值
    # avg_vol = vol_df.rolling(200).mean().iloc[-1]
    # liquid = (avg_vol * close_df.iloc[-1]) > 20_000_000
    # liquid_cols = liquid[liquid].index
    # close_df = close_df[liquid_cols]
    # vol_df   = vol_df[liquid_cols]

    print(f"回测股票数量: {len(close_df.columns)}")

    # ---------- 强制统一索引 ----------
    # 去除时区
    close_df.index = close_df.index.tz_localize(None)
    vol_df.index = vol_df.index.tz_localize(None)
    close_df = close_df.sort_index()
    vol_df = vol_df.sort_index()

    # 列名转为字符串
    close_df.columns = close_df.columns.astype(str)
    vol_df.columns = vol_df.columns.astype(str)

    # 生成指数信号并对齐
    index_signal = create_index_signal(start, end)
    index_signal = index_signal.tz_localize(None).reindex(close_df.index, method='ffill').fillna(False).astype(bool)

    close_arr = close_df.values.astype(np.float64)
    vol_arr   = vol_df.values.astype(np.float64)

    print("计算 MACD(8,16,6) 信号（Numba 加速）...")
    entries_arr, exits_arr = compute_macd_signals(close_arr, vol_arr, vol_factor=1.0)

    entries = pd.DataFrame(entries_arr, index=close_df.index, columns=close_df.columns)
    exits   = pd.DataFrame(exits_arr,   index=close_df.index, columns=close_df.columns)

    # 大盘过滤：买入信号在大盘向上时才有效，大盘向下时强制清仓
    entries = entries.where(index_signal, False)      # 这里的广播会正确处理
    exits   = exits | (~index_signal)                 # 大盘向下时，所有列触发卖出

    # ---------- 最终对齐（彻底消除 vectorbt 索引错误） ----------
    # 确保所有 DataFrame 拥有完全相同的 Index 和 Columns 对象
    master_index = close_df.index
    master_cols  = close_df.columns
    entries = pd.DataFrame(entries.reindex(index=master_index, columns=master_cols, fill_value=False).values,
                          index=master_index, columns=master_cols)
    exits   = pd.DataFrame(exits.reindex(index=master_index, columns=master_cols, fill_value=False).values,
                          index=master_index, columns=master_cols)

    print("开始向量化回测...")
    pf = vbt.Portfolio.from_signals(
        close=close_df,
        entries=entries,
        exits=exits,
        sl_stop=0.08,
        tp_stop=0.15,
        direction='longonly',
        freq='1d',
        init_cash=1_000_000,
        cash_sharing=True,
        accumulate=False,
        fees=0.0001
    )

    print("\n========== MACD(8,16,6) + 成交量 + 大盘过滤 回测结果 ==========")
    print(pf.stats())

    trades_df = pf.trades.records_readable
    if trades_df is not None and not trades_df.empty:
        trades_df.to_csv('trades_macd_final.csv', index=False)
        print("交易记录已保存至 trades_macd_final.csv")

if __name__ == '__main__':
    main()