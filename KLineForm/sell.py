"""
看跌形态识别模块

包含 K 线形态、技术指标及中长期趋势的卖出信号识别
所有函数返回与输入等长的 numpy 数组，装饰器自动提取最新布尔值。
"""
import numpy as np
import talib
from typing import List, Tuple, Dict, Optional, Union
from KLineForm.managerTool import manager_boolean, MacdConfig, MaPeriodsConfig, MaPairsConfig
from KLineForm.neutral import has_recent_surge


# ----------------------------------------------------------------------
# 1. 基础看跌形态
# ----------------------------------------------------------------------

@manager_boolean("看跌吞没", lambda arr: arr[-1] == -100)
def is_bearish_engulfing(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """看跌吞没：返回完整吞噬信号数组（100/-100/0）"""
    return talib.CDLENGULFING(open_, high, low, close)


@manager_boolean("一阴穿三线", lambda arr: arr[-1])
def is_one_black_cross_three(
    open_: np.ndarray,
    close: np.ndarray,
    ma_periods: Optional[MaPeriodsConfig] = None
) -> np.ndarray:
    """返回每日是否一阴穿三线"""
    if ma_periods is None:
        ma_periods = MaPeriodsConfig()
    periods = ma_periods.value
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(n):
        if i < max(periods) + 1:
            continue
        if close[i] >= open_[i]:
            continue
        ok = True
        for p in periods:
            if i < p:
                ok = False
                break
            ma = talib.MA(close[:i+1], p)
            if not (close[i] < ma[-1] < open_[i]):
                ok = False
                break
        res[i] = ok
    return res


@manager_boolean("MACD死叉", lambda arr: arr[-1])
def is_macd_death_cross(
    close: np.ndarray,
    macdConfig: Optional[MacdConfig] = None,
    onTheZeroAxis: bool = False
) -> np.ndarray:
    """返回每日是否发生MACD死叉"""
    if macdConfig is None:
        macdConfig = MacdConfig(value={"fastperiod": 8, "slowperiod": 16, "signalperiod": 6})
    macd, signal, _ = talib.MACD(close, **macdConfig.value)
    death = np.zeros(len(close), dtype=bool)
    death[1:] = (macd[:-1] >= signal[:-1]) & (macd[1:] < signal[1:])
    if onTheZeroAxis:
        death &= (macd < 0)
    return death


@manager_boolean("均线向下", lambda arr: arr[-1])
def is_moving_average_down(
    close: np.ndarray,
    ma_periods: Optional[MaPeriodsConfig] = None
) -> np.ndarray:
    """返回每日是否所有指定均线均向下"""
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
        ma = talib.MA(close, timeperiod=p)
        down = np.full(n, False)
        down[p:] = (ma[p:] < ma[p-1:-1])
        masks.append(down)
    return np.all(masks, axis=0)


@manager_boolean("均线死叉", lambda arr: arr[-1])
def ma_death_cross(
    close: np.ndarray,
    ma_pairs: Optional[MaPairsConfig] = None,
    afewDays: int = 0
) -> np.ndarray:
    """返回每日是否满足均线死叉条件"""
    if ma_pairs is None:
        ma_pairs = MaPairsConfig()
    pairs = ma_pairs.value
    n = len(close)
    if n < max(max(pair) for pair in pairs) + 1:
        return np.zeros(n, dtype=bool)

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
                if s[i] >= l[i]:
                    cross_ok = False; break
            else:
                found = False
                for d in range(0, afewDays + 1):
                    idx = i - d
                    if idx < 1: continue
                    if s[idx-1] >= l[idx-1] and s[idx] < l[idx]:
                        found = True; break
                if not found or s[i] >= l[i]:
                    cross_ok = False; break
        res[i] = cross_ok
    return res


# ----------------------------------------------------------------------
# 2. 成交量相关看跌形态
# ----------------------------------------------------------------------

@manager_boolean("平开低走阴线 + 巨额成交量", lambda arr: arr[-1])
def flat_open_low_close_big_volume(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    volume_mult: float = 1.5,
    body_ratio: float = 0.5,
    flat_tolerance: float = 0.01
) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(20, n):  # 需要至少20日历史平均成交量
        # 必须有近期快速上涨
        if not has_recent_surge(close[:i+1]):
            continue
        if close[i] >= open_[i]:
            continue
        if abs(open_[i] - close[i-1]) / close[i-1] > flat_tolerance:
            continue
        body = abs(close[i] - open_[i])
        range_ = high[i] - low[i]
        if range_ == 0 or body < range_ * body_ratio:
            continue
        avg_vol = np.mean(volume[i-20:i])
        if volume[i] < avg_vol * volume_mult:
            continue
        res[i] = True
    return res


@manager_boolean("放量滞涨", lambda arr: arr[-1])
def volume_stagnation(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    volume_mult: float = 1.3,
    price_range_ratio: float = 0.03
) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(20, n):
        if abs(open_[i] - close[i]) < 1e-5 and abs(open_[i] - high[i]) < open_[i] * 0.01:
            continue
        if not has_recent_surge(close[:i+1]):
            continue
        avg_vol = np.mean(volume[i-20:i])
        if volume[i] < avg_vol * volume_mult:
            continue
        if (high[i] - low[i]) / close[i-1] > price_range_ratio:
            continue
        res[i] = True
    return res


@manager_boolean("长上影线 + 巨额成交量", lambda arr: arr[-1])
def long_upper_shadow_big_volume(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    volume_mult: float = 1.5,
    shadow_ratio: float = 0.6
) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(20, n):
        if not has_recent_surge(close[:i+1]):
            continue
        range_ = high[i] - low[i]
        if range_ == 0:
            continue
        upper_shadow = high[i] - max(open_[i], close[i])
        if upper_shadow < range_ * shadow_ratio:
            continue
        avg_vol = np.mean(volume[i-20:i])
        if volume[i] < avg_vol * volume_mult:
            continue
        res[i] = True
    return res


# ----------------------------------------------------------------------
# 3. 经典看跌蜡烛线形态
# ----------------------------------------------------------------------

@manager_boolean("黄昏星", lambda arr: arr[-1] == -100)
def is_evening_star(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    penetration: float = 0.3
) -> np.ndarray:
    star = talib.CDLEVENINGSTAR(open_, high, low, close, penetration=penetration)
    # 要求前期有冲高，在向量化版本中用循环判断每个点是否有近期 surge
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(n):
        if star[i] == -100 and i >= 5 and has_recent_surge(close[:i+1], lookback=5):
            res[i] = True
    return res


@manager_boolean("黄昏十字星", lambda arr: arr[-1] == -100)
def is_evening_doji_star(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    penetration: float = 0.3
) -> np.ndarray:
    star = talib.CDLEVENINGDOJISTAR(open_, high, low, close, penetration=penetration)
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(n):
        if star[i] == -100 and i >= 5 and has_recent_surge(close[:i+1], lookback=5):
            res[i] = True
    return res


@manager_boolean("三只乌鸦", lambda arr: arr[-1] == -100)
def is_three_black_crows(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    return talib.CDL3BLACKCROWS(open_, high, low, close)


@manager_boolean("下跌三部曲", lambda arr: arr[-1])
def is_falling_three_methods(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(4, n):
        idx1 = i - 4
        if close[idx1] >= open_[idx1]:
            continue
        body1 = abs(close[idx1] - open_[idx1])
        prev_close = close[idx1 - 1] if idx1 - 1 >= 0 else close[idx1]
        if body1 / prev_close < 0.02:
            continue
        ok = True
        for j in range(idx1+1, idx1+4):
            if abs(close[j] - open_[j]) > body1 * 0.5:
                ok = False; break
            if high[j] > high[idx1] or low[j] < low[idx1]:
                ok = False; break
        if not ok: continue
        if close[i] >= open_[i] or close[i] >= low[idx1]:
            continue
        body5 = abs(close[i] - open_[i])
        if body5 / close[i-1] < 0.02:
            continue
        res[i] = True
    return res


@manager_boolean("倾盆大雨", lambda arr: arr[-1])
def is_pouring_rain(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if close[i-1] <= open_[i-1]:
            continue
        if close[i] >= open_[i] or open_[i] >= close[i-1]:
            continue
        if close[i] >= open_[i-1]:
            continue
        res[i] = True
    return res


@manager_boolean("高位十字星", lambda arr: arr[-1])
def is_high_cross(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    body_ratio: float = 0.1
) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(n):
        if not has_recent_surge(close[:i+1]):
            continue
        body = abs(close[i] - open_[i])
        k_range = high[i] - low[i]
        if k_range == 0:
            continue
        res[i] = body / k_range <= body_ratio
    return res


@manager_boolean("高位墓碑十字", lambda arr: arr[-1] == -100)
def is_gravestone_doji(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    grave = talib.CDLGRAVESTONEDOJI(open_, high, low, close)
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(n):
        if grave[i] == -100 and has_recent_surge(close[:i+1]):
            res[i] = True
    return res


@manager_boolean("乌云盖顶", lambda arr: arr[-1] == -100)
def is_dark_cloud_cover(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    penetration: float = 0.5
) -> np.ndarray:
    return talib.CDLDARKCLOUDCOVER(open_, high, low, close, penetration=penetration)


@manager_boolean("双飞乌鸦", lambda arr: arr[-1] == -100)
def is_two_crows(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    return talib.CDLUPSIDEGAP2CROWS(open_, high, low, close)


@manager_boolean("高档五连阴", lambda arr: arr[-1])
def is_five_consecutive_bears(
    close: np.ndarray,
    open_: Optional[np.ndarray] = None,
    lookback: int = 5
) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(lookback-1, n):
        if not has_recent_surge(close[:i+1]):
            continue
        ok = True
        for j in range(0, lookback):
            idx = i - j
            if open_ is not None:
                if close[idx] >= open_[idx]:
                    ok = False; break
            else:
                if idx > 0 and close[idx] >= close[idx-1]:
                    ok = False; break
        res[i] = ok
    return res


@manager_boolean("空方尖兵", lambda arr: arr[-1])
def is_bearish_soldier(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int = 5
) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if close[i-1] <= open_[i-1]:
            continue
        if close[i] >= open_[i] or close[i] >= open_[i-1]:
            continue
        res[i] = True
    return res


@manager_boolean("高位吊颈线", lambda arr: arr[-1] == -100)
def is_hanging_man(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: Optional[np.ndarray] = None,
    volume_ratio_to_prev: Optional[float] = None,
) -> np.ndarray:
    hang = talib.CDLHANGINGMAN(open_, high, low, close)
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(n):
        if hang[i] == -100 and has_recent_surge(close[:i+1]):
            if volume_ratio_to_prev is not None:
                if volume is None or i < 1:
                    continue
                if volume[i] < volume[i-1] * volume_ratio_to_prev:
                    continue
            res[i] = True
    return res


# ----------------------------------------------------------------------
# 4. 顶背离形态
# ----------------------------------------------------------------------

@manager_boolean("MACD顶背离", lambda arr: arr[-1])
def macd_top_divergence(
    close: np.ndarray,
    macdconfig: Optional[MacdConfig] = None,
    lookback: int = 20
) -> np.ndarray:
    if macdconfig is None:
        macdconfig = MacdConfig(value={"fastperiod": 8, "slowperiod": 16, "signalperiod": 6})
    macd, _, _ = talib.MACD(close, **macdconfig.value)
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        seg = close[i-lookback:i+1]
        macd_seg = macd[i-lookback:i+1]
        price_high_idx = np.argmax(seg)
        price_high = seg[price_high_idx]
        macd_at_high = macd_seg[price_high_idx]
        for j in range(i-lookback, i+1):
            if close[j] > price_high and macd[j] < macd_at_high:
                res[i] = True
                break
    return res


@manager_boolean("KDJ顶背离", lambda arr: arr[-1])
def kdj_top_divergence(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    kdjconfig: Optional[Dict[str, int]] = None,
    lookback: int = 20
) -> np.ndarray:
    if kdjconfig is None:
        kdjconfig = {"n": 9, "m1": 3, "m2": 3}
    n_period = kdjconfig.get("n", 9)
    low_n = talib.MIN(low, timeperiod=n_period)
    high_n = talib.MAX(high, timeperiod=n_period)
    rsv = (close - low_n) / (high_n - low_n + 1e-9) * 100
    rsv = np.nan_to_num(rsv)
    k = talib.EMA(rsv, timeperiod=kdjconfig.get("m1", 3))
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        seg_price = close[i-lookback:i+1]
        seg_k = k[i-lookback:i+1]
        price_high_idx = np.argmax(seg_price)
        price_high = seg_price[price_high_idx]
        k_at_high = seg_k[price_high_idx]
        for j in range(i-lookback, i+1):
            if close[j] > price_high and k[j] < k_at_high:
                res[i] = True
                break
    return res


@manager_boolean("RSI顶背离", lambda arr: arr[-1])
def rsi_top_divergence(
    close: np.ndarray,
    rsi_period: int = 14,
    lookback: int = 20
) -> np.ndarray:
    rsi = talib.RSI(close, timeperiod=rsi_period)
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        seg_price = close[i-lookback:i+1]
        seg_rsi = rsi[i-lookback:i+1]
        price_high_idx = np.argmax(seg_price)
        price_high = seg_price[price_high_idx]
        rsi_at_high = seg_rsi[price_high_idx]
        for j in range(i-lookback, i+1):
            if close[j] > price_high and rsi[j] < rsi_at_high:
                res[i] = True
                break
    return res


# ----------------------------------------------------------------------
# 5. 中长期看跌形态
# ----------------------------------------------------------------------

@manager_boolean("N周期阴线占优", lambda arr: arr[-1])
def bears_dominant_in_n_days(
    close: np.ndarray,
    open_: np.ndarray,
    lookback: int = 20,
    ratio: float = 0.6
) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(lookback-1, n):
        seg_c = close[i-lookback+1:i+1]
        seg_o = open_[i-lookback+1:i+1]
        bear_count = np.sum(seg_c < seg_o)
        if bear_count / lookback >= ratio:
            res[i] = True
    return res


@manager_boolean("中期量价同步走弱", lambda arr: arr[-1])
def medium_term_volume_price_weak(
    close: np.ndarray,
    open_: np.ndarray,
    volume: np.ndarray,
    lookback: int = 20
) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(lookback-1, n):
        seg_c = close[i-lookback+1:i+1]
        seg_o = open_[i-lookback+1:i+1]
        seg_v = volume[i-lookback+1:i+1]
        bear_v = seg_v[seg_c < seg_o]
        bull_v = seg_v[seg_c >= seg_o]
        if len(bear_v) == 0 or len(bull_v) == 0:
            continue
        if len(bear_v) > len(bull_v) and np.mean(bear_v) > np.mean(bull_v):
            res[i] = True
    return res


@manager_boolean("均线空头排列", lambda arr: arr[-1])
def moving_average_bearish_arrangement(
    close: np.ndarray,
    ma_periods: Optional[MaPeriodsConfig] = None
) -> np.ndarray:
    if ma_periods is None:
        ma_periods = MaPeriodsConfig(value=[5, 10, 20, 60])
    periods = ma_periods.value
    n = len(close)
    if n < max(periods):
        return np.zeros(n, dtype=bool)
    mas = [talib.SMA(close, timeperiod=p) for p in periods]
    res = np.ones(n, dtype=bool)
    for i in range(1, len(periods)):
        res &= (mas[i-1] < mas[i])
    for ma in mas:
        res &= np.concatenate([[True], ma[1:] < ma[:-1]])
    return res


@manager_boolean("N周期高点低点逐步降低", lambda arr: arr[-1])
def descending_highs_lows(
    high: np.ndarray,
    low: np.ndarray,
    lookback: int = 10
) -> np.ndarray:
    n = len(high)
    res = np.zeros(n, dtype=bool)
    for i in range(lookback-1, n):
        seg_h = high[i-lookback+1:i+1]
        seg_l = low[i-lookback+1:i+1]
        if np.all(np.diff(seg_h) < 0) and np.all(np.diff(seg_l) < 0):
            res[i] = True
    return res


@manager_boolean("反弹缩量下跌放量", lambda arr: arr[-1])
def rally_shrink_drop_swell(
    open_: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    lookback: int = 20
) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(lookback-1, n):
        seg_c = close[i-lookback+1:i+1]
        seg_o = open_[i-lookback+1:i+1]
        seg_v = volume[i-lookback+1:i+1]
        up_v = seg_v[seg_c > seg_o]
        down_v = seg_v[seg_c <= seg_o]
        if len(up_v) == 0 or len(down_v) == 0:
            continue
        if np.mean(up_v) < np.mean(down_v):
            res[i] = True
    return res


@manager_boolean("高位连续小阴派发", lambda arr: arr[-1])
def high_small_bears_distribution(
    close: np.ndarray,
    open_: np.ndarray,
    lookback: int = 10,
    body_ratio: float = 0.02
) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(lookback-1, n):
        if not has_recent_surge(close[:i+1]):
            continue
        seg_c = close[i-lookback+1:i+1]
        seg_o = open_[i-lookback+1:i+1]
        bear_count = 0
        ok = True
        for j in range(lookback):
            if seg_c[j] < seg_o[j]:
                body = abs(seg_c[j] - seg_o[j])
                if body / seg_c[i-lookback+j] > body_ratio:
                    ok = False
                    break
                bear_count += 1
        if ok and bear_count / lookback >= 0.7:
            res[i] = True
    return res


@manager_boolean("中长期量能递减但阴量占优", lambda arr: arr[-1])
def volume_decline_bear_dominant(
    volume: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    lookback: int = 30
) -> np.ndarray:
    n = len(volume)
    res = np.zeros(n, dtype=bool)
    for i in range(lookback-1, n):
        seg_v = volume[i-lookback+1:i+1]
        x = np.arange(lookback)
        slope = np.polyfit(x, seg_v, 1)[0]
        if slope >= 0:
            continue
        seg_c = close[i-lookback+1:i+1]
        seg_o = open_[i-lookback+1:i+1]
        bear_v = seg_v[seg_c < seg_o]
        bull_v = seg_v[seg_c >= seg_o]
        if len(bear_v) == 0 or len(bull_v) == 0:
            continue
        if np.mean(bear_v) > np.mean(bull_v):
            res[i] = True
    return res


@manager_boolean("N周期内跌幅远大于涨幅", lambda arr: arr[-1])
def net_decline_dominant(
    close: np.ndarray,
    lookback: int = 20,
    ratio: float = 2.0
) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(lookback-1, n):
        seg = close[i-lookback+1:i+1]
        total_down = 0.0
        total_up = 0.0
        for j in range(1, len(seg)):
            change = (seg[j] - seg[j-1]) / seg[j-1]
            if change < 0:
                total_down += abs(change)
            else:
                total_up += change
        if total_down > total_up * ratio:
            res[i] = True
    return res


@manager_boolean("高位放量滞涨转空头", lambda arr: arr[-1])
def high_volume_stagnation_to_bear(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    lookback: int = 10,
    vol_mult: float = 1.5
) -> np.ndarray:
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(20, n):  # 需要20日均量
        if not has_recent_surge(close[:i+1]):
            continue
        avg_vol = np.mean(volume[i-20:i])
        stagnation = False
        for j in range(i-lookback, i):
            if volume[j] > avg_vol * vol_mult:
                if (high[j] - low[j]) / close[j-1] < 0.03:
                    stagnation = True
                    break
        if not stagnation:
            continue
        recent_bears = np.sum(close[i-4:i+1] < open_[i-4:i+1])
        if recent_bears >= 3:
            res[i] = True
    return res


@manager_boolean("一阴穿多中长期均线", lambda arr: arr[-1])
def one_black_cross_multiple_ma(
    close: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    ma_periods: Optional[MaPeriodsConfig] = None
) -> np.ndarray:
    if ma_periods is None:
        ma_periods = MaPeriodsConfig(value=[10, 20, 60])
    periods = ma_periods.value
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(n):
        if close[i] >= open_[i]:
            continue
        ok = True
        for p in periods:
            if i < p:
                ok = False; break
            ma = talib.SMA(close, timeperiod=p)
            if close[i] >= ma[i]:
                ok = False; break
        res[i] = ok
    return res


@manager_boolean("中长期均线死叉共振", lambda arr: arr[-1])
def multiple_ma_death_cross(
    close: np.ndarray,
    ma_pairs: Optional[MaPairsConfig] = None
) -> np.ndarray:
    if ma_pairs is None:
        ma_pairs = MaPairsConfig(value=[(5, 10), (10, 20), (20, 60)])
    pairs = ma_pairs.value
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(n):
        ok = True
        for short_p, long_p in pairs:
            if i < long_p:
                ok = False; break
            short_ma = talib.SMA(close, timeperiod=short_p)
            long_ma = talib.SMA(close, timeperiod=long_p)
            if not (short_ma[i-1] >= long_ma[i-1] and short_ma[i] < long_ma[i]):
                ok = False; break
        res[i] = ok
    return res


@manager_boolean("MACD零轴下多次死叉", lambda arr: arr[-1])
def macd_multiple_death_below_zero(
    close: np.ndarray,
    macdconfig: Optional[MacdConfig] = None,
    times: int = 2,
    lookback: int = 50
) -> np.ndarray:
    if macdconfig is None:
        macdconfig = MacdConfig()
    macd, signal, _ = talib.MACD(close, **macdconfig.value)
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        count = 0
        for j in range(i-lookback+1, i+1):
            if macd[j-1] >= signal[j-1] and macd[j] < signal[j] and macd[j] < 0:
                count += 1
        if count >= times:
            res[i] = True
    return res


@manager_boolean("高位乌云盖顶+中期阴量占优", lambda arr: arr[-1])
def high_dark_cloud_with_bear_volume(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    lookback: int = 20
) -> np.ndarray:
    dark = talib.CDLDARKCLOUDCOVER(open_, high, low, close, penetration=0.5)
    n = len(close)
    res = np.zeros(n, dtype=bool)
    for i in range(lookback-1, n):
        if dark[i] != -100:
            continue
        seg_c = close[i-lookback+1:i+1]
        seg_o = open_[i-lookback+1:i+1]
        seg_v = volume[i-lookback+1:i+1]
        bear_v = seg_v[seg_c < seg_o]
        bull_v = seg_v[seg_c >= seg_o]
        if len(bear_v) == 0 or len(bull_v) == 0:
            continue
        if np.mean(bear_v) > np.mean(bull_v):
            res[i] = True
    return res


# 文件结束