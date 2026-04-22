from typing import Optional, List, Dict, Tuple
import numpy as np
import talib
from KLineForm.managerTool import manager_boolean, MacdConfig, MaPeriodsConfig,MaPairsConfig


@manager_boolean("看涨吞没")
def is_bullish_engulfing(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> bool:
    """
    看涨吞没：当前阳线实体完全覆盖前一根阴线实体。
    """
    return talib.CDLENGULFING(open_, high, low, close)[-1] == 100


@manager_boolean("一阳穿三线")
def is_one_barrier_three_lines(
    open_: np.ndarray,
    close: np.ndarray,
    ma_periods: Optional[MaPeriodsConfig] = None
) -> bool:
    """
    一阳穿三线：当前阳线同时上穿多条均线。
    """
    if ma_periods is None:
        ma_periods = MaPeriodsConfig()  # 默认 [5,10,20]
    periods = ma_periods.value

    if len(open_) < 2 or len(close) < 2:
        return False

    if close[-1] <= open_[-1]:
        return False

    for period in periods:
        if len(close) < period + 1:
            return False
        ma = talib.MA(close, period)
        if not (open_[-1] < ma[-1] < close[-1]):
            return False
    return True

@manager_boolean("MACD金叉")
def is_macd_golden_cross(
    close: np.ndarray,
    macdconfig: Optional[MacdConfig] = None,
    onTheZeroAxis: bool = False
) -> bool:
    """
    检测 MACD 金叉（快线上穿慢线）

    Parameters
    ----------
    close : np.ndarray
        收盘价序列
    macdconfig : MacdConfig, optional
        MACD 参数，默认 {"fastperiod":8, "slowperiod":16, "signalperiod":6}
    onTheZeroAxis : bool, default False
        是否要求金叉发生在零轴之上
    """
    if macdconfig is None:
        macdconfig = MacdConfig(value={"fastperiod": 8, "slowperiod": 16, "signalperiod": 6})

    macd, signal, _ = talib.MACD(close, **macdconfig.value)

    if len(macd) < 2 or len(signal) < 2:
        return False

    golden = macd[-2] <= signal[-2] and macd[-1] > signal[-1]
    if not golden:
        return False

    if onTheZeroAxis and macd[-1] <= 0:
        return False

    return True


@manager_boolean("均线向上")
def is_moving_average_up(
    close: np.ndarray,
    ma_periods: Optional[MaPeriodsConfig] = None
) -> bool:
    """判断是否为均线向上移动"""
    if ma_periods is None:
        ma_periods = MaPeriodsConfig()
    periods = ma_periods.value

    if not periods:
        return False

    for period in periods:
        if len(close) < period + 1:
            return False
        ma = talib.SMA(close, timeperiod=period)
        if ma[-1] < ma[-2]:
            return False
    return True


@manager_boolean("均线金叉")
def ma_golden_cross(
    close: np.ndarray,
    ma_pairs: Optional[MaPairsConfig] = None,
    afewDays: int = 0
) -> bool:
    """
    均线金叉检测

    Parameters:
        close: 收盘价序列
        ma_pairs: 均线对配置，支持字符串 "5,10;10,20" 或列表 [[5,10],[10,20]] 或 MaPairsConfig 实例
        afewDays: 0表示任意一对出现金叉即可；>0表示所有均线对必须在最近afewDays天内发生过金叉
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

        current_golden = sma_short[-1] > sma_long[-1]

        prev_short = sma_short[-2] if len(sma_short) > 1 else np.nan
        prev_long = sma_long[-2] if len(sma_long) > 1 else np.nan
        cross_event = False
        if not np.isnan(prev_short) and not np.isnan(prev_long):
            cross_event = (prev_short <= prev_long) and (sma_short[-1] > sma_long[-1])

        if afewDays == 0:
            cross_signals.append(current_golden or cross_event)
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
                    if (prev_short_val <= prev_long_val) and (curr_short > curr_long):
                        found = True
                        break
            cross_signals.append(found and current_golden)

    if afewDays == 0:
        return any(cross_signals)
    else:
        return all(cross_signals)


# ---------------------------- 精选高胜率形态 ----------------------------
@manager_boolean("早晨之星")
def is_morning_star(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> bool:
    """早晨之星 / 早晨十字星"""
    return talib.CDLMORNINGSTAR(open_, high, low, close, penetration=0)[-1] == 100


@manager_boolean("曙光初现")
def is_piercing(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> bool:
    """曙光初现"""
    return talib.CDLPIERCING(open_, high, low, close)[-1] == 100


@manager_boolean("红三兵")
def is_three_white_soldiers(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ma_periods: Optional[MaPeriodsConfig] = None
) -> bool:
    """红三兵"""
    if ma_periods is None:
        ma_periods = MaPeriodsConfig()
    periods = ma_periods.value

    if talib.CDL3WHITESOLDIERS(open_, high, low, close)[-1] != 100:
        return False
    for period in periods:
        if len(close) < period:
            return False
        ma = talib.SMA(close, timeperiod=period)
        if close[-1] <= ma[-1]:
            return False
    return True

@manager_boolean("多方炮")
def is_two_crows_one_white(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ma_periods: Optional[MaPeriodsConfig] = None
) -> bool:
    """多方炮（两阳夹一阴）"""
    if len(close) < 3:
        return False
    if close[-3] <= open_[-3]:
        return False
    if close[-2] >= open_[-2]:
        return False
    if close[-1] <= open_[-1]:
        return False
    if close[-1] <= close[-3]:
        return False
    if ma_periods is not None:
        periods = ma_periods.value
        for period in periods:
            if len(close) < period:
                return False
            ma = talib.SMA(close, timeperiod=period)
            if close[-1] <= ma[-1]:
                return False
    return True



@manager_boolean("阳包阴加强")
def is_bullish_engulfing_enhanced(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int = 4
) -> bool:
    """阳包阴加强版"""
    if len(close) < lookback + 1:
        return False
    if close[-1] <= open_[-1]:
        return False

    prev_highs = high[-lookback-1:-1]
    prev_lows = low[-lookback-1:-1]
    prev_closes = close[-lookback-1:-1]
    prev_opens = open_[-lookback-1:-1]

    bearish_count = sum(1 for i in range(lookback) if prev_closes[i] <= prev_opens[i])
    if bearish_count < lookback * 0.6:
        return False
    if high[-1] < np.max(prev_highs) or low[-1] > np.min(prev_lows):
        return False
    return True


@manager_boolean("跳空上扬")
def is_gap_up(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> bool:
    """跳空上扬"""
    if len(close) < 2:
        return False
    return low[-1] > high[-2] and close[-1] > open_[-1]


@manager_boolean("均线多头排列")
def is_bullish_ma_arrangement(
    close: np.ndarray,
    ma_periods: Optional[MaPeriodsConfig] = None
) -> bool:
    """均线多头排列"""
    if ma_periods is None:
        ma_periods = MaPeriodsConfig()
    periods = ma_periods.value

    if len(close) < max(periods) + 1:
        return False

    ma_values: List[float] = []
    for p in periods:
        ma = talib.SMA(close, timeperiod=p)
        if np.isnan(ma[-1]):
            return False
        ma_values.append(ma[-1])
        if len(ma) > 1 and ma[-1] <= ma[-2]:
            return False

    for i in range(len(ma_values) - 1):
        if ma_values[i] <= ma_values[i + 1]:
            return False
    return True

@manager_boolean("MACD零上二次金叉")
def is_macd_second_golden_cross(
    close: np.ndarray,
    macdconfig: Optional[MacdConfig] = None
) -> bool:
    """MACD零上二次金叉"""
    if macdconfig is None:
        macdconfig = MacdConfig(value={"fastperiod": 8, "slowperiod": 16, "signalperiod": 6})

    params = macdconfig.value  # 获取参数字典
    diff, dea, _ = talib.MACD(
        close,
        fastperiod=params['fastperiod'],
        slowperiod=params['slowperiod'],
        signalperiod=params['signalperiod']
    )
    if len(diff) < 5:
        return False

    curr_golden = diff[-2] <= dea[-2] and diff[-1] > dea[-1]
    if not curr_golden:
        return False
    if diff[-1] <= 0:
        return False

    found_first_golden = False
    found_dead = False
    for i in range(len(diff) - 3, 1, -1):
        dead_cross = diff[i-1] >= dea[i-1] and diff[i] < dea[i]
        if dead_cross and not found_dead:
            found_dead = True
            continue
        golden = diff[i-1] <= dea[i-1] and diff[i] > dea[i]
        if golden and not found_first_golden and not found_dead:
            if diff[i] > 0:
                found_first_golden = True
                break
    return found_first_golden and found_dead

# ---------------------------- 中期趋势类 ----------------------------
@manager_boolean("N周期阳线占优")
def is_bullish_candle_dominant(
    open_: np.ndarray,
    close: np.ndarray,
    N: int = 12,
    require_price_rise: bool = True
) -> bool:
    """最近N根K线中，阳线数量 > 阴线数量"""
    if len(close) < N:
        return False

    bullish = sum(1 for i in range(-N, 0) if close[i] > open_[i])
    bearish = N - bullish
    if bullish <= bearish:
        return False
    if require_price_rise and close[-1] <= close[-N]:
        return False
    return True


@manager_boolean("N周期量价同步走强")
def is_volume_price_sync(
    open_: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    N: int = 12,
    M: int = 20
) -> bool:
    """最近N根K线：阳线占优，阳线均量 > 阴线均量，近期均量 > 前期均量"""
    if len(close) < max(N, M) or len(volume) < max(N, M):
        return False

    vol_bullish: List[float] = []
    vol_bearish: List[float] = []
    for i in range(-N, 0):
        if close[i] > open_[i]:
            vol_bullish.append(volume[i])
        else:
            vol_bearish.append(volume[i])

    if not vol_bullish or not vol_bearish:
        return False
    if len(vol_bullish) <= len(vol_bearish):
        return False
    if np.mean(vol_bullish) <= np.mean(vol_bearish):
        return False

    recent_vol_mean = np.mean(volume[-N:])
    prev_vol_mean = np.mean(volume[-M-N:-N]) if len(volume) >= M + N else np.mean(volume[:M])
    if recent_vol_mean <= prev_vol_mean:
        return False
    return True


@manager_boolean("回调缩量上涨放量")
def is_healthy_pullback(
    open_: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    lookback: int = 20,
    pullback_ratio: float = 0.5
) -> bool:
    """回调缩量上涨放量"""
    if len(close) < lookback:
        return False

    high_idx: int = -1
    for i in range(-lookback, -2):
        if close[i] > close[i-1] and close[i] > close[i+1]:
            high_idx = i
            break
    if high_idx == -1:
        return False

    pullback_start = high_idx + 1
    if pullback_start >= 0:
        return False

    pullback_volumes = volume[pullback_start:]
    prev_start = high_idx - 5
    prev_volumes = volume[prev_start:high_idx] if prev_start >= 0 else volume[:high_idx]

    if len(pullback_volumes) == 0 or len(prev_volumes) == 0:
        return False
    if np.mean(pullback_volumes) >= np.mean(prev_volumes):
        return False
    if volume[-1] <= volume[-2] or close[-1] <= close[-2]:
        return False
    if close[-1] < close[high_idx] * (1 - pullback_ratio):
        return False
    return True


@manager_boolean("底部连续小阳推升")
def is_small_bullish_steps(
    open_: np.ndarray,
    close: np.ndarray,
    volume: Optional[np.ndarray] = None,
    N: int = 8,
    max_gain_pct: float = 2.5,
    max_loss_pct: float = 2.0
) -> bool:
    """底部连续小阳推升"""
    if len(close) < N:
        return False

    bullish_count = 0
    for i in range(-N, 0):
        change = (close[i] - open_[i]) / open_[i] * 100
        if close[i] > open_[i]:
            bullish_count += 1
            if change > max_gain_pct:
                return False
        else:
            if abs(change) > max_loss_pct:
                return False

    if bullish_count / N < 0.7:
        return False
    if close[-1] <= close[-N]:
        return False
    return True

if __name__ == '__main__':

    print(is_moving_average_up.__oldFunc__.__doc__)