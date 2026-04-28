from typing import Optional, List, Dict, Tuple
import numpy as np
import talib
from KLineForm.managerTool import MacdConfig, MaPeriodsConfig, MaPairsConfig, RsiConfig, manager_boolean

from pydantic import validate_call



@manager_boolean("看涨吞没", lambda arr: arr[-1] == 100)
def is_bullish_engulfing(open_, high, low, close) -> np.ndarray:
    """返回完整吞噬信号数组（100/-100/0）"""
    return talib.CDLENGULFING(open_, high, low, close)


@manager_boolean("一阳穿三线", lambda arr: arr[-1])
def is_one_barrier_three_lines(open_, close, ma_periods: Optional[MaPeriodsConfig] = None) -> np.ndarray:
    """返回每日是否一阳穿三线"""
    if ma_periods is None:
        ma_periods = MaPeriodsConfig()
    periods = ma_periods.value
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(n):
        if i < max(periods) + 1:
            continue
        if close[i] <= open_[i]:
            continue
        ok = True
        for p in periods:
            if i < p:
                ok = False
                break
            ma = talib.MA(close[:i+1], p)
            if not (open_[i] < ma[-1] < close[i]):
                ok = False
                break
        res[i] = ok
    return res


@manager_boolean("MACD金叉", lambda arr: arr[-1])
def is_macd_golden_cross(close, macdconfig: Optional[MacdConfig] = None, onTheZeroAxis: bool = False) -> np.ndarray:
    """返回每日是否发生MACD金叉"""
    if macdconfig is None:
        macdconfig = MacdConfig(value={"fastperiod": 8, "slowperiod": 16, "signalperiod": 6})
    macd, signal, _ = talib.MACD(close, **macdconfig.value)
    golden = np.zeros(len(close), dtype=bool)
    golden[1:] = (macd[:-1] <= signal[:-1]) & (macd[1:] > signal[1:])
    if onTheZeroAxis:
        golden &= (macd > 0)
    return golden


@manager_boolean("均线向上", lambda arr: arr[-1])
def is_moving_average_up(close, ma_periods: Optional[MaPeriodsConfig] = None) -> np.ndarray:
    """返回每日是否所有指定均线均向上"""
    if ma_periods is None:
        ma_periods = MaPeriodsConfig()
    periods = ma_periods.value
    n = len(close)
    if not periods:
        return np.zeros(n, dtype=bool)
    masks = []
    for p in periods:
        if n < p + 1:
            return np.zeros(n, dtype=bool)
        ma = talib.SMA(close, timeperiod=p)
        up = np.full(n, False)
        up[p:] = (ma[p:] > ma[p-1:-1])
        masks.append(up)
    return np.all(masks, axis=0)


@manager_boolean("均线金叉", lambda arr: arr[-1])
def ma_golden_cross(close, ma_pairs: Optional[MaPairsConfig] = None, afewDays: int = 0) -> np.ndarray:
    """返回每日是否满足均线金叉条件（afewDays功能保留，但实现略简化）"""
    if ma_pairs is None:
        ma_pairs = MaPairsConfig()
    pairs = ma_pairs.value
    n = len(close)
    if n < max(max(pair) for pair in pairs) + 1:
        return np.zeros(n, dtype=bool)

    # 预先计算所有需要的均线
    cache = {}
    def get_ma(period):
        if period not in cache:
            cache[period] = talib.SMA(close, timeperiod=period)
        return cache[period]

    res = np.zeros(n, dtype=bool)
    for i in range(n):
        cross_ok = True
        for short_p, long_p in pairs:
            s = get_ma(short_p); l = get_ma(long_p)
            if i < long_p:
                cross_ok = False; break
            if afewDays == 0:
                # 当前交叉或者已经处于金叉状态
                if s[i] <= l[i]:
                    cross_ok = False; break
            else:
                # 在最近 afewDays 内发生过金叉且当前仍保持
                found = False
                for d in range(0, afewDays + 1):
                    idx = i - d
                    if idx < 1: continue
                    if s[idx-1] <= l[idx-1] and s[idx] > l[idx]:
                        found = True; break
                if not found or s[i] <= l[i]:
                    cross_ok = False; break
        res[i] = cross_ok
    return res


@manager_boolean("早晨之星", lambda arr: arr[-1] == 100)
def is_morning_star(open_, high, low, close) -> np.ndarray:
    return talib.CDLMORNINGSTAR(open_, high, low, close, penetration=0)


@manager_boolean("曙光初现", lambda arr: arr[-1] == 100)
def is_piercing(open_, high, low, close) -> np.ndarray:
    return talib.CDLPIERCING(open_, high, low, close)


@manager_boolean("红三兵", lambda arr: arr[-1])
def is_three_white_soldiers(open_, high, low, close, ma_periods: Optional[MaPeriodsConfig] = None) -> np.ndarray:
    base = talib.CDL3WHITESOLDIERS(open_, high, low, close) == 100
    if ma_periods is not None:
        periods = ma_periods.value
        for p in periods:
            ma = talib.SMA(close, timeperiod=p)
            base &= (close > ma)
    return base


@manager_boolean("多方炮", lambda arr: arr[-1])
def is_two_crows_one_white(open_, high, low, close, ma_periods: Optional[MaPeriodsConfig] = None) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if close[i-2] <= open_[i-2]: continue
        if close[i-1] >= open_[i-1]: continue
        if close[i] <= open_[i]: continue
        if close[i] <= close[i-2]: continue
        if ma_periods is not None:
            periods = ma_periods.value
            ma_ok = True
            for p in periods:
                if i < p: ma_ok = False; break
                ma = talib.SMA(close[:i+1], timeperiod=p)
                if close[i] <= ma[-1]: ma_ok = False; break
            if not ma_ok: continue
        res[i] = True
    return res


@manager_boolean("阳包阴加强", lambda arr: arr[-1])
def is_bullish_engulfing_enhanced(open_, high, low, close, lookback: int = 4) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        if close[i] <= open_[i]: continue
        prev_h = high[i-lookback:i]
        prev_l = low[i-lookback:i]
        prev_c = close[i-lookback:i]
        prev_o = open_[i-lookback:i]
        bear_cnt = np.sum(prev_c <= prev_o)
        if bear_cnt < lookback * 0.6: continue
        if high[i] < np.max(prev_h) or low[i] > np.min(prev_l): continue
        res[i] = True
    return res


@manager_boolean("跳空上扬", lambda arr: arr[-1])
def is_gap_up(open_, high, low, close) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if low[i] > high[i-1] and close[i] > open_[i]:
            res[i] = True
    return res


@manager_boolean("均线多头排列", lambda arr: arr[-1])
def is_bullish_ma_arrangement(close, ma_periods: Optional[MaPeriodsConfig] = None) -> np.ndarray:
    if ma_periods is None:
        ma_periods = MaPeriodsConfig()
    periods = ma_periods.value
    n = len(close)
    if n < max(periods):
        return np.zeros(n, dtype=bool)
    mas = [talib.SMA(close, timeperiod=p) for p in periods]
    # 多头排列：mas[0] > mas[1] > ...，且每根均线都向上
    res = np.ones(n, dtype=bool)
    for i in range(1, len(periods)):
        res &= (mas[i-1] > mas[i])
    for ma in mas:
        res &= np.concatenate([[False], ma[1:] > ma[:-1]])
    return res


@manager_boolean("MACD零上二次金叉", lambda arr: arr[-1])
def is_macd_second_golden_cross(close, macdconfig: Optional[MacdConfig] = None) -> np.ndarray:
    if macdconfig is None:
        macdconfig = MacdConfig(value={"fastperiod": 8, "slowperiod": 16, "signalperiod": 6})
    diff, dea, _ = talib.MACD(close, **macdconfig.value)
    n = len(diff)
    res = np.zeros(n, dtype=bool)
    for i in range(4, n):
        if diff[i] <= dea[i] or diff[i] <= 0: continue
        if not (diff[i-1] <= dea[i-1] and diff[i] > dea[i]): continue
        # 寻找首次金叉和之后的死叉
        first_golden = False
        dead = False
        for j in range(i-1, 0, -1):
            if diff[j-1] >= dea[j-1] and diff[j] < dea[j]:
                dead = True
                continue
            if diff[j-1] <= dea[j-1] and diff[j] > dea[j] and not dead:
                if diff[j] > 0:
                    first_golden = True
                    break
        if first_golden and dead:
            res[i] = True
    return res


@manager_boolean("N周期阳线占优", lambda arr: arr[-1])
def is_bullish_candle_dominant(open_, close, N: int = 12, require_price_rise: bool = True) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(N-1, n):
        segment_o = open_[i-N+1:i+1]
        segment_c = close[i-N+1:i+1]
        bullish = np.sum(segment_c > segment_o)
        bearish = N - bullish
        if bullish <= bearish: continue
        if require_price_rise and close[i] <= close[i-N]: continue
        res[i] = True
    return res


@manager_boolean("N周期量价同步走强", lambda arr: arr[-1])
def is_volume_price_sync(open_, close, volume, N: int = 12, M: int = 20) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(max(N, M)-1, n):
        seg_o = open_[i-N+1:i+1]; seg_c = close[i-N+1:i+1]; seg_v = volume[i-N+1:i+1]
        bull_v = seg_v[seg_c > seg_o]; bear_v = seg_v[seg_c <= seg_o]
        if len(bull_v) == 0 or len(bear_v) == 0: continue
        if len(bull_v) <= len(bear_v): continue
        if np.mean(bull_v) <= np.mean(bear_v): continue
        recent_vol = np.mean(volume[i-N+1:i+1])
        if i - M - N + 1 >= 0:
            prev_vol = np.mean(volume[i-M-N+1:i-N+1])
        else:
            prev_vol = np.mean(volume[:M])
        if recent_vol <= prev_vol: continue
        res[i] = True
    return res


@manager_boolean("回调缩量上涨放量", lambda arr: arr[-1])
def is_healthy_pullback(open_, close, volume, lookback: int = 20, pullback_ratio: float = 0.5) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        # 寻找最近高点
        high_idx = -1
        for j in range(i-lookback, i-1):
            if close[j] > close[j-1] and close[j] > close[j+1]:
                high_idx = j
                break
        if high_idx == -1 or high_idx >= i: continue
        pullback_vol = volume[high_idx+1:i+1]
        prev_vol = volume[max(0, high_idx-5):high_idx]
        if len(pullback_vol) == 0 or len(prev_vol) == 0: continue
        if np.mean(pullback_vol) >= np.mean(prev_vol): continue
        if volume[i] <= volume[i-1] or close[i] <= close[i-1]: continue
        if close[i] < close[high_idx] * (1 - pullback_ratio): continue
        res[i] = True
    return res


@manager_boolean("底部连续小阳推升", lambda arr: arr[-1])
def is_small_bullish_steps(open_, close, volume=None, N: int = 8, max_gain_pct: float = 2.5, max_loss_pct: float = 2.0) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(N-1, n):
        seg_o = open_[i-N+1:i+1]; seg_c = close[i-N+1:i+1]
        bullish = 0
        ok = True
        for j in range(N):
            change = (seg_c[j] - seg_o[j]) / seg_o[j] * 100
            if seg_c[j] > seg_o[j]:
                bullish += 1
                if change > max_gain_pct: ok = False; break
            else:
                if abs(change) > max_loss_pct: ok = False; break
        if not ok: continue
        if bullish / N < 0.7: continue
        if close[i] <= close[i-N]: continue
        res[i] = True
    return res


@manager_boolean("RSI超卖", lambda arr: arr[-1])
def is_rsi_oversold(close, rsi_periods: Optional[RsiConfig] = None, threshold: float = 30.0) -> np.ndarray:
    if rsi_periods is None:
        rsi_periods = RsiConfig()
    periods = rsi_periods.value
    n = len(close)
    res = np.zeros(n, dtype=bool)
    rsi_cache = {}
    for p in periods:
        rsi_cache[p] = talib.RSI(close, timeperiod=p)
    for i in range(n):
        for p in periods:
            rsi_val = rsi_cache[p][i]
            if not np.isnan(rsi_val) and rsi_val < threshold:
                res[i] = True
                break
    return res


# 如果本文件作为主模块运行，可进行简单测试
if __name__ == '__main__':
    print("buy signals vectorized version loaded.")