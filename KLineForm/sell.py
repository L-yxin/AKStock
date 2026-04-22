"""
看跌形态识别模块

包含 K 线形态、技术指标及中长期趋势的卖出信号识别
"""
import numpy as np
import talib
from typing import List, Tuple, Dict, Optional, Union
from KLineForm.managerTool import manager_boolean, MacdConfig, MaPeriodsConfig,MaPairsConfig
from KLineForm.neutral import has_recent_surge


@manager_boolean("看跌吞没")
def is_bearish_engulfing(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> bool:
    """看跌吞没：当前阴线实体完全覆盖前一根阳线实体。"""
    return talib.CDLENGULFING(open_, high, low, close)[-1] == -100


@manager_boolean("一阴穿三线")
def is_one_black_cross_three(
    open_: np.ndarray,
    close: np.ndarray,
    ma_periods: Optional[MaPeriodsConfig]= None
) -> bool:
    """
    一阴穿三线：当前阴线同时下穿多条均线。

    参数:
        open_ : np.ndarray 开盘价序列
        close : np.ndarray 收盘价序列
        ma_periods : list of int 需要穿过的均线

    返回:
        bool 是否满足条件
    """
    if ma_periods is None:
        ma_periods = MaPeriodsConfig()
    if len(open_) < 2 or len(close) < 2:
        return False

    if close[-1] >= open_[-1]:
        return False

    for period in ma_periods.value:
        ma = talib.MA(close, period)
        if len(ma) < 2:
            return False
        if not (close[-1] < ma[-1] < open_[-1]):
            return False
    return True


@manager_boolean("MACD死叉")
def is_macd_death_cross(
    close: np.ndarray,
    macdConfig: Optional[MacdConfig] = None,
    onTheZeroAxis: bool = False
) -> bool:
    """
    检测 MACD 死叉（快线下穿慢线）

    Parameters
    ----------
    close : np.ndarray
        收盘价序列
    macdConfig : MacdConfig, optional
        MACD 参数，默认 {"fastperiod":8, "slowperiod":16, "signalperiod":6}
    onTheZeroAxis : bool, default False
        是否要求死叉发生在零轴之下
    """
    if macdConfig is None:
        macdConfig = MacdConfig(value={"fastperiod": 8, "slowperiod": 16, "signalperiod": 6})

    macd, signal, _ = talib.MACD(close, **macdConfig.value)

    if len(macd) < 2 or len(signal) < 2:
        return False

    death = macd[-2] >= signal[-2] and macd[-1] < signal[-1]

    if not death:
        return False

    if onTheZeroAxis and macd[-1] >= 0:
        return False

    return True

@manager_boolean("均线向下")
def is_moving_average_down(
    close: np.ndarray,
    ma_periods: Optional[MaPeriodsConfig] = None
) -> bool:
    """
    判断是否为均线向下移动
    """
    if ma_periods is None:
        ma_periods = MaPeriodsConfig()
    periods = ma_periods.value

    if not periods:
        return False

    for period in periods:
        if len(close) < period + 1:
            return False
        ma = talib.MA(close, period)
        if ma[-1] > ma[-2]:
            return False
    return True


@manager_boolean("均线死叉")
def ma_death_cross(
    close: np.ndarray,
    ma_pairs: Optional[MaPairsConfig] = None,
    afewDays: int = 0
) -> bool:
    """
    均线死叉检测
    """
    if ma_pairs is None:
        ma_pairs = MaPairsConfig()  # 默认 [(5,10), (10,20)]
    pairs = ma_pairs.value

    if len(close) < max(max(pair) for pair in pairs) + 1:
        return False

    cross_signals: List[bool] = []
    for short_period, long_period in pairs:
        sma_short = talib.SMA(close, timeperiod=short_period)
        sma_long = talib.SMA(close, timeperiod=long_period)
        if np.isnan(sma_short[-1]) or np.isnan(sma_long[-1]):
            continue

        current_death = sma_short[-1] < sma_long[-1]

        prev_short = sma_short[-2] if len(sma_short) > 1 else np.nan
        prev_long = sma_long[-2] if len(sma_long) > 1 else np.nan
        cross_event = False
        if not np.isnan(prev_short) and not np.isnan(prev_long):
            cross_event = (prev_short >= prev_long) and (sma_short[-1] < sma_long[-1])

        if afewDays == 0:
            cross_signals.append(current_death or cross_event)
        else:
            if len(close) < afewDays + max(short_period, long_period):
                cross_signals.append(False)
                continue
            found = False
            for i in range(1, afewDays + 1):
                idx = -i
                if idx < -len(sma_short) + 1:
                    break
                curr_short = sma_short[idx]
                curr_long = sma_long[idx]
                prev_short_idx = idx - 1
                if prev_short_idx < -len(sma_short):
                    continue
                prev_short_val = sma_short[prev_short_idx]
                prev_long_val = sma_long[prev_short_idx]
                if not np.isnan(prev_short_val) and not np.isnan(prev_long_val):
                    if (prev_short_val >= prev_long_val) and (curr_short < curr_long):
                        found = True
                        break
            cross_signals.append(found and current_death)

    if afewDays == 0:
        return any(cross_signals)
    else:
        return all(cross_signals)

# ---------- 形态2：平开低走阴线 + 巨额成交量 ----------
@manager_boolean("平开低走阴线 + 巨额成交量")
def flat_open_low_close_big_volume(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    volume_mult: float = 1.5,
    body_ratio: float = 0.5,
    flat_tolerance: float = 0.01
) -> bool:
    """
    平开低走阴线 + 巨额成交量
    """
    if not has_recent_surge(close):
        return False
    if len(close) < 21:
        return False

    if close[-1] >= open_[-1]:
        return False

    if abs(open_[-1] - close[-2]) / close[-2] > flat_tolerance:
        return False

    body = abs(close[-1] - open_[-1])
    range_ = high[-1] - low[-1]
    if range_ == 0 or body < range_ * body_ratio:
        return False

    avg_vol = np.mean(volume[-20:-1]) if len(volume) >= 21 else np.mean(volume)
    if volume[-1] < avg_vol * volume_mult:
        return False

    return True


# ---------- 形态3：放量滞涨 ----------
@manager_boolean("放量滞涨")
def volume_stagnation(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    volume_mult: float = 1.3,
    price_range_ratio: float = 0.03
) -> bool:
    """
    放量滞涨：成交量显著放大，但价格波动很小
    """
    if len(open_) > 0 and len(close) > 0 and len(high) > 0:
        if abs(open_[-1] - close[-1]) < 1e-5 and abs(open_[-1] - high[-1]) < open_[-1] * 0.01:
            return False

    if not has_recent_surge(close):
        return False
    if len(close) < 21:
        return False

    avg_vol = np.mean(volume[-20:-1]) if len(volume) >= 21 else np.mean(volume)
    if volume[-1] < avg_vol * volume_mult:
        return False

    prev_close = close[-2]
    if (high[-1] - low[-1]) / prev_close > price_range_ratio:
        return False

    return True


# ---------- 形态4：长上影线 + 巨额成交量 ----------
@manager_boolean("长上影线 + 巨额成交量")
def long_upper_shadow_big_volume(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    volume_mult: float = 1.5,
    shadow_ratio: float = 0.6
) -> bool:
    """
    长上影线 + 巨额成交量
    """
    if not has_recent_surge(close):
        return False
    if len(close) < 21:
        return False

    range_ = high[-1] - low[-1]
    if range_ == 0:
        return False

    upper_shadow = high[-1] - max(open_[-1], close[-1])
    if upper_shadow < range_ * shadow_ratio:
        return False

    avg_vol = np.mean(volume[-20:-1]) if len(volume) >= 21 else np.mean(volume)
    if volume[-1] < avg_vol * volume_mult:
        return False

    return True


# ---------- 常见看跌 K 线形态 ----------
@manager_boolean("黄昏星")
def is_evening_star(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    penetration: float = 0.3
) -> bool:
    """黄昏星形态"""
    if not has_recent_surge(close, lookback=5):
        return False
    star = talib.CDLEVENINGSTAR(open_, high, low, close, penetration=penetration)
    return star[-1] == -100


@manager_boolean("黄昏十字星")
def is_evening_doji_star(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    penetration: float = 0.3
) -> bool:
    """黄昏十字星"""
    star = talib.CDLEVENINGDOJISTAR(open_, high, low, close, penetration=penetration)
    return star[-1] == -100


@manager_boolean("三只乌鸦")
def is_three_black_crows(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> bool:
    """三只乌鸦"""
    crows = talib.CDL3BLACKCROWS(open_, high, low, close)
    return crows[-1] == -100


@manager_boolean("下跌三部曲")
def is_falling_three_methods(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> bool:
    """下跌三部曲（空方炮）"""
    if len(close) < 5:
        return False

    idx1 = -5
    if close[idx1] >= open_[idx1]:
        return False
    body1 = abs(close[idx1] - open_[idx1])
    prev_close = close[idx1 - 1] if idx1 - 1 >= 0 else close[idx1]
    if body1 / prev_close < 0.02:
        return False

    for i in range(-4, -1):
        body = abs(close[i] - open_[i])
        if body > body1 * 0.5:
            return False
        if high[i] > high[idx1] or low[i] < low[idx1]:
            return False

    if close[-1] >= open_[-1]:
        return False
    if close[-1] >= low[idx1]:
        return False
    body5 = abs(close[-1] - open_[-1])
    if body5 / close[-2] < 0.02:
        return False

    return True


@manager_boolean("倾盆大雨")
def is_pouring_rain(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> bool:
    """倾盆大雨"""
    if len(close) < 2:
        return False
    if close[-2] <= open_[-2]:
        return False
    if close[-1] >= open_[-1] or open_[-1] >= close[-2]:
        return False
    if close[-1] >= open_[-2]:
        return False
    return True


@manager_boolean("高位十字星")
def is_high_cross(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    body_ratio: float = 0.1
) -> bool:
    """高位十字星"""
    if not has_recent_surge(close):
        return False
    body = abs(close[-1] - open_[-1])
    k_range = high[-1] - low[-1]
    if k_range == 0:
        return False
    return body / k_range <= body_ratio


@manager_boolean("高位墓碑十字")
def is_gravestone_doji(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> bool:
    """高位墓碑十字"""
    grave = talib.CDLGRAVESTONEDOJI(open_, high, low, close)
    return grave[-1] == -100


@manager_boolean("乌云盖顶")
def is_dark_cloud_cover(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    penetration: float = 0.5
) -> bool:
    """乌云盖顶"""
    cloud = talib.CDLDARKCLOUDCOVER(open_, high, low, close, penetration=penetration)
    return cloud[-1] == -100


@manager_boolean("双飞乌鸦")
def is_two_crows(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> bool:
    """双飞乌鸦"""
    crows = talib.CDLUPSIDEGAP2CROWS(open_, high, low, close)
    return crows[-1] == -100


@manager_boolean("高档五连阴")
def is_five_consecutive_bears(
    close: np.ndarray,
    open_: Optional[np.ndarray] = None,
    lookback: int = 5
) -> bool:
    """高档五连阴"""
    if len(close) < lookback:
        return False
    if not has_recent_surge(close):
        return False
    for i in range(1, lookback + 1):
        if open_ is not None:
            if close[-i] >= open_[-i]:
                return False
        else:
            if close[-i] >= close[-i - 1]:
                return False
    return True


@manager_boolean("空方尖兵")
def is_bearish_soldier(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int = 5
) -> bool:
    """空方尖兵"""
    if len(close) < 2:
        return False
    if close[-2] <= open_[-2]:
        return False
    if close[-1] >= open_[-1] or close[-1] >= open_[-2]:
        return False
    return True


@manager_boolean("高位吊颈线")
def is_hanging_man(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: Optional[np.ndarray] = None,
    volume_ratio_to_prev: Optional[float] = None,
) -> bool:
    """高位吊颈线"""
    if not has_recent_surge(close):
        return False

    hang = talib.CDLHANGINGMAN(open_, high, low, close)
    if hang[-1] != -100:
        return False

    if volume_ratio_to_prev is not None:
        if volume is None or len(volume) < 2:
            return False
        if volume[-1] < volume[-2] * volume_ratio_to_prev:
            return False

    return True


# ---------- 顶背离形态 ----------
@manager_boolean("MACD顶背离")
def macd_top_divergence(
    close: np.ndarray,
    macdconfig: Optional[MacdConfig] = None,
    lookback: int = 20
) -> bool:
    """MACD顶背离"""
    if macdconfig is None:
        macdconfig = MacdConfig(value={"fastperiod": 8, "slowperiod": 16, "signalperiod": 6})
    macd, _, _ = talib.MACD(close, **macdconfig.value)
    if len(macd) < lookback:
        return False
    price_high_idx = np.argmax(close[-lookback:])
    price_high = close[-lookback:][price_high_idx]
    macd_at_price_high = macd[-lookback:][price_high_idx]
    for i in range(-lookback, 0):
        if close[i] > price_high and macd[i] < macd_at_price_high:
            return True
    return False


@manager_boolean("KDJ顶背离")
def kdj_top_divergence(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    kdjconfig: Optional[Dict[str, int]] = None,
    lookback: int = 20
) -> bool:
    """KDJ顶背离
    参数:
    kdjconfig:{"n": 9, "m1": 3, "m2": 3}
    """
    if kdjconfig is None:
        kdjconfig = {"n": 9, "m1": 3, "m2": 3}
    n = kdjconfig.get("n", 9)
    low_n = talib.MIN(low, timeperiod=n)
    high_n = talib.MAX(high, timeperiod=n)
    rsv = (close - low_n) / (high_n - low_n) * 100
    rsv = np.nan_to_num(rsv)
    k = talib.EMA(rsv, timeperiod=kdjconfig.get("m1", 3))
    d = talib.EMA(k, timeperiod=kdjconfig.get("m2", 3))
    if len(k) < lookback:
        return False
    price_high_idx = np.argmax(close[-lookback:])
    price_high = close[-lookback:][price_high_idx]
    k_at_high = k[-lookback:][price_high_idx]
    for i in range(-lookback, 0):
        if close[i] > price_high and k[i] < k_at_high:
            return True
    return False


@manager_boolean("RSI顶背离")
def rsi_top_divergence(
    close: np.ndarray,
    rsi_period: int = 14,
    lookback: int = 20
) -> bool:
    """RSI顶背离"""
    rsi = talib.RSI(close, timeperiod=rsi_period)
    if len(rsi) < lookback:
        return False
    price_high_idx = np.argmax(close[-lookback:])
    price_high = close[-lookback:][price_high_idx]
    rsi_at_high = rsi[-lookback:][price_high_idx]
    for i in range(-lookback, 0):
        if close[i] > price_high and rsi[i] < rsi_at_high:
            return True
    return False


# ---------- 中长期看跌形态 ----------
@manager_boolean("N周期阴线占优")
def bears_dominant_in_n_days(
    close: np.ndarray,
    open_: np.ndarray,
    lookback: int = 20,
    ratio: float = 0.6
) -> bool:
    """N周期内阴线数量占比大于 ratio"""
    if len(close) < lookback:
        return False
    bear_count = sum(1 for i in range(-lookback, 0) if close[i] < open_[i])
    return bear_count / lookback >= ratio


@manager_boolean("中期量价同步走弱")
def medium_term_volume_price_weak(
    close: np.ndarray,
    open_: np.ndarray,
    volume: np.ndarray,
    lookback: int = 20
) -> bool:
    """中期量价同步走弱"""
    if len(close) < lookback:
        return False
    bear_vol: List[float] = []
    bull_vol: List[float] = []
    for i in range(-lookback, 0):
        if close[i] < open_[i]:
            bear_vol.append(volume[i])
        else:
            bull_vol.append(volume[i])
    if not bear_vol or not bull_vol:
        return False
    return (len(bear_vol) > len(bull_vol)) and (np.mean(bear_vol) > np.mean(bull_vol))


@manager_boolean("均线空头排列")
def moving_average_bearish_arrangement(
    close: np.ndarray,
    ma_periods: Optional[MaPeriodsConfig] = None
) -> bool:
    """均线空头排列"""
    if ma_periods is None:
        ma_periods = MaPeriodsConfig(value=[5, 10, 20, 60])  # 默认使用指定周期
    periods = ma_periods.value

    if len(close) < max(periods):
        return False
    mas = [talib.SMA(close, timeperiod=p) for p in periods]
    for i in range(len(mas) - 1):
        if mas[i][-1] >= mas[i + 1][-1]:
            return False
    for ma in mas:
        if ma[-1] >= ma[-2]:
            return False
    return True


@manager_boolean("N周期高点低点逐步降低")
def descending_highs_lows(
    high: np.ndarray,
    low: np.ndarray,
    lookback: int = 10
) -> bool:
    """N周期内高点依次降低，低点依次降低"""
    if len(high) < lookback or len(low) < lookback:
        return False
    recent_highs = high[-lookback:]
    recent_lows = low[-lookback:]
    high_desc = all(recent_highs[i] > recent_highs[i + 1] for i in range(len(recent_highs) - 1))
    low_desc = all(recent_lows[i] > recent_lows[i + 1] for i in range(len(recent_lows) - 1))
    return high_desc and low_desc


@manager_boolean("反弹缩量下跌放量")
def rally_shrink_drop_swell(
    open_: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    lookback: int = 20
) -> bool:
    """反弹缩量下跌放量"""
    if len(close) < lookback:
        return False
    up_vol: List[float] = []
    down_vol: List[float] = []
    for i in range(-lookback, 0):
        if close[i] > open_[i]:
            up_vol.append(volume[i])
        else:
            down_vol.append(volume[i])
    if not up_vol or not down_vol:
        return False
    return np.mean(up_vol) < np.mean(down_vol)


@manager_boolean("高位连续小阴派发")
def high_small_bears_distribution(
    close: np.ndarray,
    open_: np.ndarray,
    lookback: int = 10,
    body_ratio: float = 0.02
) -> bool:
    """高位连续小阴线派发"""
    if not has_recent_surge(close):
        return False
    if len(close) < lookback:
        return False
    bear_count = 0
    for i in range(-lookback, 0):
        if close[i] < open_[i]:
            body = abs(close[i] - open_[i])
            if body / close[i - 1] > body_ratio:
                return False
            bear_count += 1
    return bear_count / lookback >= 0.7


@manager_boolean("中长期量能递减但阴量占优")
def volume_decline_bear_dominant(
    volume: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    lookback: int = 30
) -> bool:
    """成交量趋势下降，但阴线均量大于阳线均量"""
    if len(volume) < lookback:
        return False
    x = np.arange(lookback)
    y = volume[-lookback:]
    slope = np.polyfit(x, y, 1)[0]
    if slope >= 0:
        return False
    bear_vol = [volume[i] for i in range(-lookback, 0) if close[i] < open_[i]]
    bull_vol = [volume[i] for i in range(-lookback, 0) if close[i] >= open_[i]]
    if not bear_vol or not bull_vol:
        return False
    return np.mean(bear_vol) > np.mean(bull_vol)


@manager_boolean("N周期内跌幅远大于涨幅")
def net_decline_dominant(
    close: np.ndarray,
    lookback: int = 20,
    ratio: float = 2.0
) -> bool:
    """总跌幅 > 总涨幅 * ratio"""
    if len(close) < lookback:
        return False
    total_down = 0.0
    total_up = 0.0
    for i in range(-lookback, -1):
        change = (close[i + 1] - close[i]) / close[i]
        if change < 0:
            total_down += abs(change)
        else:
            total_up += change
    return total_down > total_up * ratio


@manager_boolean("高位放量滞涨转空头")
def high_volume_stagnation_to_bear(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    lookback: int = 10,
    vol_mult: float = 1.5
) -> bool:
    """高位放量滞涨后阴线占优"""
    if not has_recent_surge(close):
        return False
    if len(volume) < 20:
        return False
    avg_vol = np.mean(volume[-20:-1])
    stagnation = False
    for i in range(-lookback, 0):
        if volume[i] > avg_vol * vol_mult:
            range_ = (high[i] - low[i]) / close[i - 1]
            if range_ < 0.03:
                stagnation = True
                break
    if not stagnation:
        return False
    recent_bears = sum(1 for i in range(-5, 0) if close[i] < open_[i])
    return recent_bears >= 3


@manager_boolean("一阴穿多中长期均线")
def one_black_cross_multiple_ma(
    close: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    ma_periods: Optional[MaPeriodsConfig] = None
) -> bool:
    """单根阴线同时跌破多条中长期均线"""
    if ma_periods is None:
        ma_periods = MaPeriodsConfig(value=[10, 20, 60])  # 默认周期
    periods = ma_periods.value

    if close[-1] >= open_[-1]:
        return False
    for p in periods:
        if len(close) < p:
            return False
        ma = talib.SMA(close, timeperiod=p)
        if close[-1] >= ma[-1]:
            return False
    return True


@manager_boolean("中长期均线死叉共振")
def multiple_ma_death_cross(
    close: np.ndarray,
    ma_pairs: Optional[MaPairsConfig] = None
) -> bool:
    """多组均线先后死叉且当前空头排列"""
    if ma_pairs is None:
        ma_pairs = MaPairsConfig(value=[(5, 10), (10, 20), (20, 60)])
    pairs = ma_pairs.value

    for short_p, long_p in pairs:
        if len(close) < long_p:
            return False
        short_ma = talib.SMA(close, timeperiod=short_p)
        long_ma = talib.SMA(close, timeperiod=long_p)
        if not (short_ma[-2] >= long_ma[-2] and short_ma[-1] < long_ma[-1]):
            return False
    return True


@manager_boolean("MACD零轴下多次死叉")
def macd_multiple_death_below_zero(
    close: np.ndarray,
    macdconfig: Optional[MacdConfig] = None,
    times: int = 2,
    lookback: int = 50
) -> bool:
    """MACD零轴下出现多次死叉"""
    if macdconfig is None:
        macdconfig = MacdConfig()  # 使用默认的 (12,26,9)
    macd, signal, _ = talib.MACD(close, **macdconfig.value)
    if len(macd) < lookback:
        return False
    death_count = 0
    for i in range(-lookback + 1, 0):
        if macd[i - 1] >= signal[i - 1] and macd[i] < signal[i] and macd[i] < 0:
            death_count += 1
    return death_count >= times

@manager_boolean("高位乌云盖顶+中期阴量占优")
def high_dark_cloud_with_bear_volume(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    lookback: int = 20
) -> bool:
    """乌云盖顶后中期阴量占优"""
    if not is_dark_cloud_cover(open_, high, low, close):
        return False
    bear_vol = [volume[i] for i in range(-lookback, 0) if close[i] < open_[i]]
    bull_vol = [volume[i] for i in range(-lookback, 0) if close[i] >= open_[i]]
    if not bear_vol or not bull_vol:
        return False
    return np.mean(bear_vol) > np.mean(bull_vol)