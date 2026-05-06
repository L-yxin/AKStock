from typing import Optional

import numpy as np
import talib

from KLineForm.managerTool import MacdConfig, MaPeriodsConfig, MaPairsConfig, RsiConfig, manager_boolean

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
def is_rsi_oversold(
    close: np.ndarray,
    rsi_periods: Optional[RsiConfig] = None,
    threshold: float = 30.0,
    lookback: int = 20,
    accel_factor: float = 1.8
) -> np.ndarray:
    """
    动态 RSI 超卖判定：结合当前趋势和价格加速度，捕获加速赶底。

    - 在下降趋势中，只有价格下跌加速（短期跌幅显著大于中期均速）且 RSI 进入极端低位才视为超卖；
    - 在上升趋势中，价格快速回调（短期跌幅异常）且 RSI 超卖也会触发信号；
    - 横盘震荡时，沿用传统 RSI 阈值过滤。

    Parameters
    ----------
    close : 收盘价序列
    rsi_periods : RSI周期配置（默认 [6,12,24]）
    threshold : RSI 超卖阈值（默认 30.0）
    lookback : 趋势计算窗口（默认 20）
    accel_factor : 加速因子：短期跌幅 / 中期平均跌幅 > accel_factor 才认定加速
    """
    if rsi_periods is None:
        rsi_periods = RsiConfig()
    periods = rsi_periods.value
    n = len(close)

    # ---------- 计算 RSI ----------
    rsi_cache = {}
    for p in periods:
        rsi_cache[p] = talib.RSI(close, timeperiod=p)

    # ---------- 计算价格变动 ----------
    # 短期跌幅（5日）
    ret5 = np.full(n, np.nan, dtype=np.float64)
    ret5[5:] = close[5:] / close[:-5] - 1.0

    # 中期平均每日跌幅（用 lookback 窗口线性回归斜率近似）
    # 斜率表示每日涨跌额的中期趋势（正为上升，负为下降）
    slopes = np.zeros(n, dtype=np.float64)
    for i in range(lookback, n):
        y = close[i - lookback : i]
        x = np.arange(lookback)
        # 线性回归斜率
        slope = np.polyfit(x, y, 1)[0]
        slopes[i] = slope

    # 中期平均涨跌幅（近似日均涨跌幅）
    avg_daily_ret = slopes / close  # 化为百分比

    # ---------- 输出信号 ----------
    res = np.zeros(n, dtype=bool)

    for i in range(max(lookback, max(periods)) + 1, n):
        # 必须至少有一个 RSI 低于阈值
        rsi_low = any(
            (not np.isnan(rsi_cache[p][i])) and rsi_cache[p][i] < threshold
            for p in periods
        )
        if not rsi_low:
            continue

        # 趋势判断
        trend_up = slopes[i] > 0   # 上升趋势
        trend_down = slopes[i] < 0 # 下降趋势

        # 加速条件
        accel = False
        if trend_down:
            # 下降趋势：短期跌幅要明显大于中期平均跌幅
            avg_down = abs(avg_daily_ret[i]) * 5  # 5日预期跌幅
            if ret5[i] < -avg_down * accel_factor:
                accel = True
        elif trend_up:
            # 上升趋势：出现异常回调（短期跌幅大）视为超卖
            if ret5[i] < -0.05:    # 5日跌超5%，显著回调
                accel = True
        else:
            # 横盘：直接用 RSI 阈值，相当于传统超卖
            accel = True

        if accel:
            res[i] = True

    return res


@manager_boolean("n日低开率高", lambda arr: arr[-1])
def is_high_low_open_ratio(
        open_: np.ndarray,
        close: np.ndarray,
        n_days: int = 5,
        ratio: float = 0.5,
        require_price_above_n: bool = True    # 新增：要求当前价高于n日前收盘价
) -> np.ndarray:
    """
    判断最近 n_days 个交易日内，低开天数占总天数的比例是否 >= ratio。

    参数：
        n_days    : 回顾窗口天数，默认 5
        ratio     : 低开天数占比阈值，默认 0.5（即 50%）
        require_price_above_n : 为 True 时，要求当前收盘价 > n_days 前的收盘价
    返回与输入等长的 bool 数组。
    """
    n = len(close)
    # 每日是否低开（开盘价 < 前收）
    is_low_open_day = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if open_[i] < close[i - 1]:
            is_low_open_day[i] = True

    result = np.zeros(n, dtype=bool)
    for i in range(n_days - 1, n):
        # 低开率条件
        window = is_low_open_day[i - n_days + 1 : i + 1]
        if np.sum(window) / n_days < ratio:
            continue

        # 附加条件：当前价高于 n 日前价格
        if require_price_above_n:
            idx_before = i - n_days
            if idx_before < 0 or close[i] <= close[idx_before]:
                continue

        result[i] = True

    return result



@manager_boolean("看涨孕线", lambda arr: arr[-1] == 100)
def is_bullish_harami(open_, high, low, close) -> np.ndarray:
    return talib.CDLHARAMI(open_, high, low, close)


# 如果本文件作为主模块运行，可进行简单测试
if __name__ == '__main__':
    print("buy signals vectorized version loaded.")