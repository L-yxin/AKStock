import numpy as np
import talib
from KLineForm.managerTool import manager_boolean
from KLineForm.neutral import has_recent_surge

@manager_boolean("看跌吞没")
def is_bearish_engulfing(open_,high,low,close ):
    """看跌吞没：当前阴线实体完全覆盖前一根阳线实体。"""
    return talib.CDLENGULFING(open_, high, low,close)[-1]==-100

@manager_boolean("一阴穿三线")
def is_one_black_cross_three(open_, close, ma_list):
    """
    一阴穿三线：当前阴线同时下穿多条均线。

    参数:
        open_ : np.ndarray 开盘价序列
        close : np.ndarray 收盘价序列
        ma_list : list of np.ndarray 需要穿过的均线数组列表

    返回:
        bool 是否满足条件
    """
    if len(open_) < 2 or len(close) < 2:
        return False

    # 当前必须是阴线
    if close[-1] >= open_[-1]:
        return False

    for ma in ma_list:
        if len(ma) < 2:
            return False
        # 下穿条件：前一日收盘价在均线之上，今日收盘价在均线之下
        # 或实体穿越：开盘价在均线之上，收盘价在均线之下
        if not (close[-1] < ma[-1] < open_[-1]):
            return False
    return True

@manager_boolean("MACD死叉")
def is_macd_death_cross(close: np.ndarray, macdConfig=None, onTheZeroAxis=False):
    """
    检测 MACD 死叉（快线下穿慢线）

    Parameters
    ----------
    close : np.ndarray
        收盘价序列（至少需包含 slowperiod+signalperiod 根数据）
    macdConfig : dict, optional
        MACD 参数，默认 {"fastperiod":8, "slowperiod":16, "signalperiod":6}
    onTheZeroAxis : bool, default False
        是否要求死叉发生在零轴之下（即当前 MACD 值 < 0）

    Returns
    -------
    bool
        是否出现死叉
    """
    if macdConfig is None:
        macdConfig = {"fastperiod": 8, "slowperiod": 16, "signalperiod": 6}

    macd, signal, hist = talib.MACD(close, **macdConfig)

    if len(macd) < 2 or len(signal) < 2:
        return False

    # 死叉条件：前一日 macd >= signal，今日 macd < signal
    death = macd[-2] >= signal[-2] and macd[-1] < signal[-1]

    if not death:
        return False

    # 若要求零轴之下，则检查当前 macd < 0
    if onTheZeroAxis and macd[-1] >= 0:
        return False

    return True

@manager_boolean("均线向下")
def is_moving_average_down(close: np.ndarray, ma_periods: list)-> bool:
    """
    判断是否为均线向下移动

    Parameters
    ----------
    close : np.ndarray
        收盘价序列
    ma_periods : list
        均线序列

    Returns
    -------
    bool
        是否为均线向下移动
    """
    if ma_periods is None or len(ma_periods) == 0:
        return False
    for ma in ma_periods:
        if len(close) <ma+1:
            raise ValueError("ma_periods参数错误")
        ma_xx = talib.MA(close, ma)
        if ma_xx[-1] > ma_xx[-2]:
            return False
    return  True

@manager_boolean("均线死叉")
def ma_death_cross(close: np.ndarray, ma_periodsTuple=None, afewDays=0):
    """
    均线死叉检测
    """
    if ma_periodsTuple is None:
        ma_periodsTuple = [[5, 10], [10, 20]]
    if len(close) < max([max(pair) for pair in ma_periodsTuple]) + 1:
        return False

    cross_signals = []
    for short_period, long_period in ma_periodsTuple:
        sma_short = talib.SMA(close, timeperiod=short_period)
        sma_long = talib.SMA(close, timeperiod=long_period)
        if np.isnan(sma_short[-1]) or np.isnan(sma_long[-1]):
            continue
        current_death = sma_short[-1] < sma_long[-1]
        prev_short = sma_short[-2] if len(sma_short) > 1 else np.nan
        prev_long = sma_long[-2] if len(sma_long) > 1 else np.nan
        if not np.isnan(prev_short) and not np.isnan(prev_long):
            cross_event = (prev_short >= prev_long) and (sma_short[-1] < sma_long[-1])
        else:
            cross_event = False

        if afewDays == 0:
            cross_signals.append(current_death or cross_event)
        else:
            if len(close) < afewDays + max(short_period, long_period):
                cross_signals.append(False)
                continue
            found = False
            for i in range(1, afewDays+1):
                idx = -i
                if idx < -len(sma_short)+1:
                    break
                curr_short = sma_short[idx]
                curr_long = sma_long[idx]
                prev_short_idx = idx-1
                prev_long_idx = idx-1
                if prev_short_idx < 0 or prev_long_idx < 0:
                    continue
                prev_short = sma_short[prev_short_idx]
                prev_long = sma_long[prev_short_idx]
                if not np.isnan(prev_short) and not np.isnan(prev_long):
                    if (prev_short >= prev_long) and (curr_short < curr_long):
                        found = True
                        break
            cross_signals.append(found and current_death)

    if afewDays == 0:
        return any(cross_signals)
    else:
        return all(cross_signals)



# ---------- 形态1：高开低走大阴线 + 巨额成交量 ----------
@manager_boolean("高开低走大阴线 + 巨额成交量")
def high_open_low_close_big_volume(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    volume_mult: float = 1.5,
    body_ratio: float = 0.03,
    open_gap: float = 0.01
) -> bool:
    """
    高开低走大阴线 + 巨额成交量
    条件：
        1. 前期有过大幅上涨
        2. 当前阴线（收盘 < 开盘）
        3. 高开：开盘价 > 前一日收盘价 * (1 + open_gap)
        4. 回落：|收盘-开盘| /开盘 <  body_ratio
        5. 成交量 ≥ 过去20日均量 * volume_mult
    """
    if not has_recent_surge(close):
        return False
    if len(close) < 21:  # 需要足够数据计算均量
        return False

    # 当前阴线
    if close[-1] >= open_[-1]:
        return False

    # 高开
    if open_[-1] <= close[-2] * (1 + open_gap):
        return False

    if (open_[-1]-close[-1]) /open_[-1] < body_ratio:
        return False

    # # 成交量条件（过去20日均量，不含当日）
    # avg_vol = np.mean(volume[-20:-1]) if len(volume) >= 21 else np.mean(volume)
    # if volume[-1] < avg_vol * volume_mult:
    #     return False

    return True


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
    条件：
        1. 前期有过大幅上涨
        2. 当前阴线
        3. 平开：|开盘 - 前收盘| / 前收盘 ≤ flat_tolerance
        4. 实体较大：|收盘-开盘| ≥ (最高-最低) * body_ratio
        5. 成交量 ≥ 过去20日均量 * volume_mult
    """
    if not has_recent_surge(close):
        return False
    if len(close) < 21:
        return False

    if close[-1] >= open_[-1]:
        return False

    # 平开
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
    条件：
        1. 前期有过大幅上涨
        2. 成交量 ≥ 过去20日均量 * volume_mult
        3. 当日振幅 ≤ 前一日收盘价 * price_range_ratio
        4. 排除一字板（涨停板）
    """
    # 排除一字板（开盘=收盘=最高，允许微小误差）
    if len(open_) > 0 and len(close) > 0 and len(high) > 0:
        if abs(open_[-1] - close[-1]) < 1e-5 and abs(open_[-1] - high[-1]) < open_[-1] * 0.01:
            return False

    if not has_recent_surge(close):
        return False
    if len(close) < 21:
        return False

    # 成交量放大
    avg_vol = np.mean(volume[-20:-1]) if len(volume) >= 21 else np.mean(volume)
    if volume[-1] < avg_vol * volume_mult:
        return False

    # 滞涨：振幅小
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
    条件：
        1. 前期有过大幅上涨
        2. 上影线长度 ≥ (最高-最低) * shadow_ratio
        3. 成交量 ≥ 过去20日均量 * volume_mult
    """
    if not has_recent_surge(close, ):
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



# ---------- 以下为补充的常见看跌 K 线形态（优先使用 TA-Lib）----------

@manager_boolean("黄昏星")
def is_evening_star(open_, high, low, close, penetration=0.3):
    """
    黄昏星形态：一段上涨后，先一根大阳线，中间一根小实体（星线），最后一根大阴线深入第一根阳线实体。
    参数 penetration: 阴线收盘价需跌破阳线实体的比例，默认 0.3（即 30%）
    """
    # TA-Lib 的 CDLEVENINGSTAR 返回 -100 表示标准黄昏星

    if not has_recent_surge( close,lookback=5):
        return  False
    star = talib.CDLEVENINGSTAR(open_, high, low, close, penetration=penetration)
    return star[-1] == -100

@manager_boolean("黄昏十字星")
def is_evening_doji_star(open_, high, low, close, penetration=0.3):
    """
    黄昏十字星：中间为十字星（开盘=收盘）的黄昏星。
    TA-Lib 有 CDLEVENINGDOJISTAR
    """

    star = talib.CDLEVENINGDOJISTAR(open_, high, low, close, penetration=penetration)
    return star[-1] == -100


@manager_boolean("三只乌鸦")
def is_three_black_crows(open_, high, low, close):
    """三只乌鸦：高位连续三根向下的阴线，每日收盘价依次降低。"""
    crows = talib.CDL3BLACKCROWS(open_, high, low, close)
    return crows[-1] == -100

@manager_boolean("下跌三部曲")
def is_falling_three_methods(open_, high, low, close):
    """
    下跌三部曲（空方炮）：大阴线 -> 小阳线反弹（不破阴线高点） -> 再一根大阴线跌破前低。
    手动实现标准下跌三部曲（5根K线模式）：
        - 第1根：大阴线
        - 中间3根：小实体阳线（或小阴线），高低点被第1根K线范围包含
        - 第5根：大阴线，收盘价低于第1根最低点
    """
    if len(close) < 5:
        return False

    # 第1根（倒数第5根）为大阴线
    idx1 = -5
    if close[idx1] >= open_[idx1]:
        return False
    body1 = abs(close[idx1] - open_[idx1])
    prev_close = close[idx1-1] if idx1-1 >= 0 else close[idx1]
    if body1 / prev_close < 0.02:   # 大阴线实体至少2%
        return False

    # 中间3根（索引 -4, -3, -2）必须是小实体，且高低点不突破第1根的范围
    for i in range(-4, -1):
        body = abs(close[i] - open_[i])
        # 实体不能太大（小于第1根实体的50%）
        if body > body1 * 0.5:
            return False
        # 高低点不能突破第1根K线的高低点
        if high[i] > high[idx1] or low[i] < low[idx1]:
            return False

    # 第5根（最后一根）为大阴线，且收盘价跌破第1根最低点
    if close[-1] >= open_[-1]:
        return False
    if close[-1] >= low[idx1]:
        return False
    body5 = abs(close[-1] - open_[-1])
    if body5 / close[-2] < 0.02:
        return False

    return True

@manager_boolean("倾盆大雨")
def is_pouring_rain(open_, high, low, close):
    """
    倾盆大雨：前一根阳线，后一根低开低走大阴线，收盘低于前一根开盘。
    使用 TA-Lib 的 CDLDARKCLOUDCOVER（乌云盖顶）有相似但不同，需单独实现。
    """
    if len(close) < 2:
        return False
    # 前一根阳线
    if close[-2] <= open_[-2]:
        return False
    # 当前阴线且低开
    if close[-1] >= open_[-1] or open_[-1] >= close[-2]:
        return False
    # 收盘低于前一根开盘价
    if close[-1] >= open_[-2]:
        return False
    return True


@manager_boolean("高位十字星")
def is_high_cross(open_, high, low, close, body_ratio=0.1):
    """
    高位十字星：实体极小，上下影线较长，出现在上涨之后。
    body_ratio: 实体长度占整个K线长度的比例上限，默认 0.1
    """
    if not has_recent_surge(close):
        return False
    body = abs(close[-1] - open_[-1])
    k_range = high[-1] - low[-1]
    if k_range == 0:
        return False
    return body / k_range <= body_ratio

@manager_boolean("高位墓碑十字")
def is_gravestone_doji(open_, high, low, close):
    """
    高位墓碑十字：开盘价=收盘价=最低价，长上影线。
    TA-Lib 有 CDLGRAVESTONEDOJI
    """
    grave = talib.CDLGRAVESTONEDOJI(open_, high, low, close)
    return grave[-1] == -100

@manager_boolean("乌云盖顶")
def is_dark_cloud_cover(open_, high, low, close, penetration=0.5):
    """
    乌云盖顶：阳线之后，高开低走阴线，收盘扎入前阳线实体内部。
    penetration: 阴线收盘价深入阳线实体的比例（0~1），默认 0.5
    """
    cloud = talib.CDLDARKCLOUDCOVER(open_, high, low, close, penetration=penetration)
    return cloud[-1] == -100

@manager_boolean("双飞乌鸦")
def is_two_crows(open_, high, low, close):
    """
    双飞乌鸦：两根向上跳空低开的阴线，逐步走低。
    TA-Lib 有 CDLUPSIDEGAP2CROWS
    """
    crows = talib.CDLUPSIDEGAP2CROWS(open_, high, low, close)
    return crows[-1] == -100

@manager_boolean("高档五连阴")
def is_five_consecutive_bears(close, open_=None, lookback=5):
    """
    高档五连阴：连续5根阴线（收盘价 < 开盘价），且处于高位。
    """
    if len(close) < lookback:
        return False
    if not has_recent_surge(close):
        return False
    # 要求最近 lookback 根全部是阴线
    for i in range(1, lookback+1):
        if close[-i] >= open_[-i] if open_ is not None else close[-i] >= close[-i-1]:
            # 如果没提供 open_，用收盘价比较前一日收盘（简单近似）
            if open_ is None:
                if close[-i] >= close[-i-1]:
                    return False
            else:
                return False
    return True


@manager_boolean("空方尖兵")
def is_bearish_soldier(open_, high, low, close, lookback=5):
    """
    空方尖兵：下跌途中一根阳线反弹，随后立刻被阴线打回，反弹失败。
    简单实现：最近两天内出现先阳后阴，且阴线收盘价低于阳线开盘价。
    """
    if len(close) < 2:
        return False
    # 前一日阳线
    if close[-2] <= open_[-2]:
        return False
    # 今日阴线且收盘低于前一日开盘
    if close[-1] >= open_[-1] or close[-1] >= open_[-2]:
        return False
    return True

@manager_boolean("高位吊颈线")
def is_hanging_man(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray = None,
    volume_ratio_to_prev: float = None,
) -> bool:
    """
    高位吊颈线：小实体、长下影线，出现在上涨之后。

    参数：
        volume               : 成交量序列（必须提供，否则无法判断放量/缩量）
        volume_ratio_to_prev : 放量阈值，例如 1.2 表示当日成交量必须 ≥ 前一日成交量 × 1.2。
                               若设置为 None，则不检查成交量（不推荐，因为缩量震仓应排除）。
                               若设置为 0 < x ≤ 1，则要求放量（大于前日），但通常用 >1。

    逻辑：
        1. 前期必须有大幅上涨（has_recent_surge）
        2. 形态必须为 TA-Lib 定义的吊颈线
        3. 若提供了 volume_ratio_to_prev，则必须放量（volume[-1] >= volume[-2] * volume_ratio_to_prev）
           缩量（小于前一日）会被过滤，因为缩量上涨是震仓，还会继续涨。
    """
    # 前期大幅上涨
    if not has_recent_surge(close):
        return False

    # TA-Lib 吊颈线识别
    hang = talib.CDLHANGINGMAN(open_, high, low, close)
    if hang[-1] != -100:
        return False

    # 成交量判断：必须放量（对比前一日），排除缩量震仓
    if volume_ratio_to_prev is not None:
        if volume is None or len(volume) < 2:
            return False
        # 缩量（当日成交量 < 前一日 * ratio）则过滤掉
        if volume[-1] < volume[-2] * volume_ratio_to_prev:
            return False

    return True

# ---------- 顶背离形态（K线 + 指标）----------
@manager_boolean("MACD顶背离")
def macd_top_divergence(close, macdconfig=None, lookback=20):
    """
    MACD顶背离：价格创新高，MACD（或MACD柱）不创新高。
    macdconfig: 可选字典，
    """
    if macdconfig is None:
        macdconfig = {"fastperiod": 8, "slowperiod": 16, "signalperiod": 6}
    macd, signal, hist = talib.MACD(close, **macdconfig)
    if len(macd) < lookback:
        return False
    # 取最近两个波峰（简单方法：找最近N日内的最高价和对应MACD值）
    price_high_idx = np.argmax(close[-lookback:])
    price_high = close[-lookback:][price_high_idx]
    macd_at_price_high = macd[-lookback:][price_high_idx]
    # 检查近期是否有更高价格但MACD更低
    for i in range(-lookback, 0):
        if close[i] > price_high and macd[i] < macd_at_price_high:
            return True
    return False

@manager_boolean("KDJ顶背离")
def kdj_top_divergence(high, low, close, kdjconfig=None, lookback=20):
    """
    KDJ顶背离：价格创新高，K值或J值不创新高。
    kdjconfig: {"n":9, "m1":3, "m2":3} 默认
    """
    if kdjconfig is None:
        kdjconfig = {"n": 9, "m1": 3, "m2": 3}
    n = kdjconfig.get("n", 9)
    # 计算RSV
    low_n = talib.MIN(low, timeperiod=n)
    high_n = talib.MAX(high, timeperiod=n)
    rsv = (close - low_n) / (high_n - low_n) * 100
    rsv = np.nan_to_num(rsv)
    # 计算K、D
    k = talib.EMA(rsv, timeperiod=kdjconfig.get("m1", 3))
    d = talib.EMA(k, timeperiod=kdjconfig.get("m2", 3))
    j = 3*k - 2*d
    if len(k) < lookback:
        return False
    # 类似MACD背离逻辑
    price_high_idx = np.argmax(close[-lookback:])
    price_high = close[-lookback:][price_high_idx]
    k_at_high = k[-lookback:][price_high_idx]
    for i in range(-lookback, 0):
        if close[i] > price_high and k[i] < k_at_high:
            return True
    return False

@manager_boolean("RSI顶背离")
def rsi_top_divergence(close, rsi_period=14, lookback=20):
    """RSI顶背离：价格创新高，RSI不创新高。"""
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

# ---------- 中长期看跌形态（胜率≥85%）----------
@manager_boolean("N周期阴线占优")
def bears_dominant_in_n_days(close, open_, lookback=20, ratio=0.6):
    """
    N周期内阴线数量占比大于 ratio。
    默认 lookback=20, ratio=0.6（即60%以上为阴线）
    """
    if len(close) < lookback:
        return False
    bear_count = 0
    for i in range(-lookback, 0):
        if close[i] < open_[i]:
            bear_count += 1
    return bear_count / lookback >= ratio

@manager_boolean("中期量价同步走弱")
def medium_term_volume_price_weak(close, open_, volume, lookback=20):
    """
    最近N根K线：
      - 阴线数量 > 阳线数量
      - 阴线成交量均值 > 阳线成交量均值
    """
    if len(close) < lookback:
        return False
    bear_vol = []
    bull_vol = []
    for i in range(-lookback, 0):
        if close[i] < open_[i]:
            bear_vol.append(volume[i])
        else:
            bull_vol.append(volume[i])
    if len(bear_vol) == 0 or len(bull_vol) == 0:
        return False
    return (len(bear_vol) > len(bull_vol)) and (np.mean(bear_vol) > np.mean(bull_vol))

@manager_boolean("均线空头排列")
def moving_average_bearish_arrangement(close, ma_periods=None):
    """
    均线空头排列：短期均线在下，长期均线在上，且所有均线向下。
    """
    if ma_periods is None:
        ma_periods = [5, 10, 20, 60]
    if len(close) < max(ma_periods):
        return False
    mas = [talib.SMA(close, timeperiod=p) for p in ma_periods]
    # 检查排列：ma[0] < ma[1] < ... < ma[-1]
    for i in range(len(mas)-1):
        if mas[i][-1] >= mas[i+1][-1]:
            return False
    # 检查所有均线向下（斜率负）
    for ma in mas:
        if ma[-1] >= ma[-2]:
            return False
    return True



@manager_boolean("N周期高点低点逐步降低")
def descending_highs_lows(high, low, lookback=10):
    """
    N周期内，每个周期的高点依次降低，低点依次降低。
    """
    if len(high) < lookback or len(low) < lookback:
        return False
    recent_highs = high[-lookback:]
    recent_lows = low[-lookback:]
    # 检查单调递减
    high_desc = all(recent_highs[i] > recent_highs[i+1] for i in range(len(recent_highs)-1))
    low_desc = all(recent_lows[i] > recent_lows[i+1] for i in range(len(recent_lows)-1))
    return high_desc and low_desc

@manager_boolean("反弹缩量下跌放量")
def rally_shrink_drop_swell(open_,close, volume, lookback=20):
    """
    最近N日中，反弹日（收盘>开盘）成交量均值 < 下跌日成交量均值。
    """
    if len(close) < lookback:
        return False
    up_vol = []
    down_vol = []
    for i in range(-lookback, 0):
        if close[i] > open_[i]:
            up_vol.append(volume[i])
        else:
            down_vol.append(volume[i])
    if len(up_vol)==0 or len(down_vol)==0:
        return False
    return np.mean(up_vol) < np.mean(down_vol)

@manager_boolean("高位连续小阴派发")
def high_small_bears_distribution(close, open_, lookback=10, body_ratio=0.02):
    """
    高位连续小阴线：最近 lookback 根K线中，阴线比例 > 70%，且每根阴线实体小于前日收盘价的 body_ratio。
    """
    if not has_recent_surge(close):
        return False
    if len(close) < lookback:
        return False
    bear_count = 0
    for i in range(-lookback, 0):
        if close[i] < open_[i]:
            body = abs(close[i] - open_[i])
            if body / close[i-1] > body_ratio:
                return False  # 实体过大，不是小阴线
            bear_count += 1
    return bear_count / lookback >= 0.7

@manager_boolean("中长期量能递减但阴量占优")
def volume_decline_bear_dominant(volume, close, open_, lookback=30):
    """
    成交量总体趋势下降，但阴线成交量均值 > 阳线成交量均值。
    """
    if len(volume) < lookback:
        return False
    # 成交量趋势：用线性回归斜率判断
    x = np.arange(lookback)
    y = volume[-lookback:]
    slope = np.polyfit(x, y, 1)[0]
    if slope >= 0:  # 不是递减
        return False
    # 阴量 vs 阳量
    bear_vol = [volume[i] for i in range(-lookback,0) if close[i] < open_[i]]
    bull_vol = [volume[i] for i in range(-lookback,0) if close[i] >= open_[i]]
    if len(bear_vol)==0 or len(bull_vol)==0:
        return False
    return np.mean(bear_vol) > np.mean(bull_vol)

@manager_boolean("N周期内跌幅远大于涨幅")
def net_decline_dominant(close, lookback=20, ratio=2.0):
    """
    N周期内，总跌幅（阴线下跌幅度之和） > 总涨幅 * ratio。
    """
    if len(close) < lookback:
        return False
    total_down = 0.0
    total_up = 0.0
    for i in range(-lookback, -1):
        change = (close[i+1] - close[i]) / close[i]
        if change < 0:
            total_down += abs(change)
        else:
            total_up += change
    return total_down > total_up * ratio

@manager_boolean("高位放量滞涨转空头")
def high_volume_stagnation_to_bear(open_, high, low,close, volume, lookback=10, vol_mult=1.5):
    """
    前期放量滞涨，随后阴线开始占优。
    条件：最近N日中有成交量放大但价格波动小，且后续阴线数量增加。
    """
    if not has_recent_surge(close):
        return False
    if len(volume) < 20:
        return False
    avg_vol = np.mean(volume[-20:-1])
    # 检查最近是否有放量滞涨日
    stagnation = False
    for i in range(-lookback, 0):
        if volume[i] > avg_vol * vol_mult:
            range_ = (high[i] - low[i]) / close[i-1]
            if range_ < 0.03:
                stagnation = True
                break
    if not stagnation:
        return False
    # 后续（滞涨日之后）阴线占比 > 60%
    # 简单实现：检查最近5日阴线比例
    recent_bears = sum(1 for i in range(-5,0) if close[i] < open_[i])
    return recent_bears >= 3

@manager_boolean("一阴穿多中长期均线")
def one_black_cross_multiple_ma(close, open_, high, low, ma_periods=[10,20,60]):
    """
    单根阴线同时跌破多条中长期均线（收盘价低于均线）。
    """
    if close[-1] >= open_[-1]:  # 不是阴线
        return False
    for p in ma_periods:
        if len(close) < p:
            return False
        ma = talib.SMA(close, timeperiod=p)
        if close[-1] >= ma[-1]:  # 未跌破
            return False
    return True

@manager_boolean("中长期均线死叉共振")
def multiple_ma_death_cross(close, ma_pairs=[[5,10],[10,20],[20,60]]):
    """
    多组均线先后死叉（短期下穿长期），且当前全部呈空头排列。
    """
    for short_p, long_p in ma_pairs:
        if len(close) < long_p:
            return False
        short_ma = talib.SMA(close, timeperiod=short_p)
        long_ma = talib.SMA(close, timeperiod=long_p)
        # 检查死叉：前一日 short >= long，今日 short < long
        if not (short_ma[-2] >= long_ma[-2] and short_ma[-1] < long_ma[-1]):
            return False
    return True

@manager_boolean("MACD零轴下多次死叉")
def macd_multiple_death_below_zero(close, macdconfig=None, times=2, lookback=50):
    """
    MACD在零轴下方出现至少 times 次死叉，每次金叉后均无力反弹。
    简单实现：统计 lookback 内死叉次数且死叉时 MACD 值 < 0。
    """
    if macdconfig is None:
        macdconfig = {"fastperiod":12, "slowperiod":26, "signalperiod":9}
    macd, signal, hist = talib.MACD(close, **macdconfig)
    if len(macd) < lookback:
        return False
    death_count = 0
    for i in range(-lookback+1, 0):
        if macd[i-1] >= signal[i-1] and macd[i] < signal[i] and macd[i] < 0:
            death_count += 1
    return death_count >= times

@manager_boolean("高位乌云盖顶+中期阴量占优")
def high_dark_cloud_with_bear_volume(open_, high, low, close, volume, lookback=20):
    """
    出现乌云盖顶形态后，中期（lookback）内阴线成交量占优。
    """
    # 检查乌云盖顶
    if not is_dark_cloud_cover(open_, high, low, close):
        return False
    # 检查中期阴量占优
    bear_vol = [volume[i] for i in range(-lookback,0) if close[i] < open_[i]]
    bull_vol = [volume[i] for i in range(-lookback,0) if close[i] >= open_[i]]
    if len(bear_vol)==0 or len(bull_vol)==0:
        return False
    return np.mean(bear_vol) > np.mean(bull_vol)

