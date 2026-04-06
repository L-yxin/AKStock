import numpy as np
import talib

from KLineForm.managerTool import manager_boolean


@manager_boolean("看涨吞没")
def is_bullish_engulfing(open_,high,low,close):
    """
    看涨吞没：当前阳线实体完全覆盖前一根阴线实体。
    """
    return talib.CDLENGULFING(open_, high, low, close)[-1] == 100

@manager_boolean("一阳穿三线")
def is_one_barrier_three_lines(open_, close, ma_list):
    """
    一阳穿三线：当前阳线同时上穿多条均线。

    参数:
        open_ : np.ndarray 开盘价序列
        close : np.ndarray 收盘价序列
        ma_list : list of np.ndarray 需要穿过的均线数组列表（每条均线长度与K线数据一致）

    返回:
        bool 是否满足条件
    """
    if len(open_) < 2 or len(close) < 2:
        return False

    # 当前必须是阳线
    if close[-1] <= open_[-1]:
        return False

    # 遍历每条均线，检查是否同时上穿
    for ma in ma_list:
        if len(ma) < 2:
            return False
        # 上穿条件：前一日收盘价在均线之下，今日收盘价在均线之上
        # 这里采用更严格的条件：实体穿越（开盘价在均线之下，收盘价在均线之上）
        # 可根据需要调整为收盘价穿越
        if not (open_[-1] < ma[-1] < close[-1]):
            return False
    return True

@manager_boolean("MACD金叉")
def is_macd_golden_cross(close: np.ndarray, macdConfig=None, onTheZeroAxis=False):
    """
    检测 MACD 金叉（快线上穿慢线）

    Parameters
    ----------
    close : np.ndarray
        收盘价序列（至少需包含 slowperiod+signalperiod 根数据）
    macdConfig : dict, optional
        MACD 参数，默认 {"fastperiod":8, "slowperiod":16, "signalperiod":6}
    onTheZeroAxis : bool, default False
        是否要求金叉发生在零轴之上（即当前 MACD 值 > 0）

    Returns
    -------
    bool
        是否出现金叉
    """
    if macdConfig is None:
        macdConfig = {"fastperiod": 8, "slowperiod": 16, "signalperiod": 6}

    # 计算 MACD
    macd, signal, hist = talib.MACD(close, **macdConfig)

    # 确保有足够数据（至少 2 个值用于判断）
    if len(macd) < 2 or len(signal) < 2:
        return False

    # 金叉条件：前一日 macd <= signal，今日 macd > signal
    golden = macd[-2] <= signal[-2] and macd[-1] > signal[-1]

    if not golden:
        return False

    # 若要求零轴之上，则检查当前 macd > 0
    if onTheZeroAxis and macd[-1] <= 0:
        return False

    return True

@manager_boolean("均线向上")
def is_moving_average_up(close: np.ndarray, ma_periods: list):
    """
    判断是否为均线向上移动

    Parameters
    ----------
    close : np.ndarray
        收盘价序列
    ma_periods : list
        均线周期列表

    Returns
    -------
    bool
        是否为均线向上移动
    """
    if ma_periods is None or len(ma_periods) == 0:
        return False

    for ma_period in ma_periods:
        if len(close) < ma_period+1:
            return False
        ma = talib.SMA(close, timeperiod=ma_period)
        if ma[-1] < ma[-2]:
            return False
    return  True

@manager_boolean("均线金叉")
def ma_golden_cross(close: np.ndarray, ma_periodsTuple=None, afewDays=0):
    """
    均线金叉检测

    Parameters:
        close: 收盘价序列
        ma_periodsTuple: 均线对列表，如 [[5,10],[10,20]]
        afewDays: 0表示任意一对出现金叉即可；>0表示所有均线对必须在最近afewDays天内发生过金叉

    Returns:
        bool
    """
    if ma_periodsTuple is None:
        ma_periodsTuple = [[5, 10], [10, 20]]
    if len(close) < max([max(pair) for pair in ma_periodsTuple]) + 1:
        return False

    # 存储每个均线对的金叉信号（True表示当前满足金叉状态或历史金叉）
    cross_signals = []
    for short_period, long_period in ma_periodsTuple:
        # 计算短期和长期均线
        sma_short = talib.SMA(close, timeperiod=short_period)
        sma_long = talib.SMA(close, timeperiod=long_period)
        # 取有效值
        if np.isnan(sma_short[-1]) or np.isnan(sma_long[-1]):
            continue
        # 判断当前是否金叉状态（短期 > 长期）
        current_golden = sma_short[-1] > sma_long[-1]
        # 判断是否上穿事件（前一日短期 <= 长期，今日 >）
        prev_short = sma_short[-2] if len(sma_short) > 1 else np.nan
        prev_long = sma_long[-2] if len(sma_long) > 1 else np.nan
        if not np.isnan(prev_short) and not np.isnan(prev_long):
            cross_event = (prev_short <= prev_long) and (sma_short[-1] > sma_long[-1])
        else:
            cross_event = False

        if afewDays == 0:
            # 只要当前是金叉状态或者发生了上穿事件
            cross_signals.append(current_golden or cross_event)
        else:
            # 需要检查最近afewDays天内是否存在上穿事件
            # 获取最近afewDays+1天的数据
            if len(close) < afewDays + max(short_period, long_period):
                cross_signals.append(False)
                continue
            # 检查从 -afewDays-1 到 -1 范围内是否有上穿
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
                    if (prev_short <= prev_long) and (curr_short > curr_long):
                        found = True
                        break
            # 另外，当前也必须处于金叉状态（短期>长期）
            cross_signals.append(found and current_golden)

    if afewDays == 0:
        return any(cross_signals)
    else:
        return all(cross_signals)



"""
超高胜率看涨 K 线形态（成功率 ≥85%）
1. 早晨之星 / 早晨十字星
下跌末期出现：小阴线 → 十字星 / 小 K 线 → 大阳线。强烈见底反转信号，胜率极高。
2. 曙光初现
连续下跌后，一根大阳线开盘跳空低开，收盘深入前一根阴线实体 1/2 以上。空头力量耗尽，多头开始反攻，短期胜率极高。
3. 红三兵
连续三根逐步走高的小阳线，收盘价不断创新高。稳健上涨信号，中长期胜率极高。
4. 多方炮（两阳夹一阴）
两根阳线中间夹一根阴线，第二根阳线收盘价高于第一根。典型强势整理后继续上涨，短期胜率 85%+。
5. 跳空上扬
上涨途中出现向上跳空缺口，不回补。多头强势延续信号，趋势继续概率极高。
6. 低位孕线
前一根大阴线，后一根小 K 线完全被前一根实体包裹。下跌动能枯竭，即将反弹，胜率极高。
7. 反击线（捉腰带线）
开盘价为当日最低价，一路上涨收阳线。多头强势反击，短期胜率极高。
8. 上升三部曲
大阳线 → 小阴线回调 → 再大阳线创新高。强势上涨中继，几乎必涨。
9. 平底（平脚底）
连续多根 K 线最低价在同一水平，获得强支撑。下跌结束，即将反弹，中长期稳定。
10. 阳包阴（看涨吞没加强版）
一根大阳线完全包裹前面多根阴线。多头全面接管，短期暴涨概率大。
11. 低位长下影线（探底针）
大跌后出现长下影 K 线，下影线长度≥实体 2 倍。资金抄底，止跌信号，胜率极高。
12. 均线多头排列（全周期看涨）
短期、中期、长期均线从上到下依次排列。最稳定、最可靠的趋势上涨信号，长期胜率 90%+。
13. MACD 零上二次金叉
MACD 在零轴上方再次金叉，主升浪信号。趋势最强形态，胜率 90%+。
14. 放量突破前高
成交量放大，K 线收盘价突破前期高点。真突破，加速上涨信号。
15. 回踩支撑不破
回踩均线 / 前高 / 趋势线后立刻收阳线。趋势延续信号，成功率极高。
我帮你精选 最值得写代码、胜率最高的 10 个
早晨之星
曙光初现
红三兵
多方炮
低位长下影线
阳包阴（吞没加强）
跳空上扬
上升三部曲
均线多头排列
MACD 零上二次金叉

"""

"""
一、中期趋势类（5～20 周期，最稳、胜率最高）
1. N 周期阳线占优（中期强势）
最近 N 根 K 线中，阳线数量明显大于阴线数量，且收盘价重心持续上移。代表中期买盘远大于卖盘，趋势健康。
2. N 周期量价同步走强（中期最稳信号）
最近 N 根 K 线：
阳线数量 > 阴线数量
阳线成交量均值 > 阴线成交量均值
近期成交量 > 前期成交量
量价齐升，中期上涨概率 90%+。
3. 中期均线多头排列（5/10/20/60）
短期均线在上，长期均线在下，且所有均线向上发散。标准上升趋势，中期必涨形态。
4. 回调缩量、上涨放量（中期健康结构）
上涨时放量，下跌时缩量。说明主力锁仓、抛压小，中期继续上涨。
5. 底部连续小阳推升（红肥绿瘦）
连续小阳线推升，阴线少、跌幅小。典型主力吸筹，中期爆发概率极高。

"""


# ---------------------------- 精选高胜率形态 ----------------------------
@manager_boolean("早晨之星")
def is_morning_star(open_, high, low, close):
    """
    早晨之星 / 早晨十字星（下跌末期反转信号）
    使用 TA-Lib 的 CDLMORNINGSTAR，穿透率设为0（标准形态）
    """
    # 注意：CDLMORNINGSTAR 返回值 100 表示看涨早晨之星
    return talib.CDLMORNINGSTAR(open_, high, low, close, penetration=0)[-1] == 100

@manager_boolean("曙光初现")
def is_piercing(open_, high, low, close):
    """曙光初现：大阳线深入前一根阴线实体1/2以上"""
    return talib.CDLPIERCING(open_, high, low, close)[-1] == 100

@manager_boolean("红三兵")
def is_three_white_soldiers(open_, high, low, close, ma_periods:list=None):
    """红三兵：连续三根逐步走高的小阳线，且最后一根收盘价站上指定均线"""
    if ma_periods is None:
        ma_periods = [5, 10, 20]
    # 红三兵形态
    if talib.CDL3WHITESOLDIERS(open_, high, low, close)[-1] != 100:
        return False
    # 最后一根站上所有均线
    for period in ma_periods:
        if len(close) < period:
            return False
        ma = talib.SMA(close, timeperiod=period)
        if close[-1] <= ma[-1]:
            return False
    return True

@manager_boolean("多方炮")
def is_two_crows_one_white(open_, high, low, close, ma_periods:list=None):
    """
    多方炮（两阳夹一阴）：
    最近3根K线：阳-阴-阳，且第三根阳线收盘价高于第一根阳线收盘价

    tip:上升趋势高度有效 推荐30ma以上
    """
    if len(close) < 3:
        return False
    # 第一根阳线
    if close[-3] <= open_[-3]:
        return False
    # 第二根阴线
    if close[-2] >= open_[-2]:
        return False
    # 第三根阳线
    if close[-1] <= open_[-1]:
        return False
    # 第三根收盘价高于第一根收盘价
    if close[-1] <= close[-3]:
        return False
    # 可选：第二根阴线实体在第一根阳线实体内（更强的多方炮）
    # 此处不做严格限制，可根据需要注释
    if ma_periods is not None:
        for period in ma_periods:
            if len(close) < period:
                return False
            ma = talib.SMA(close, timeperiod=period)
            if close[-1] <= ma[-1]:
                return False


    return True


@manager_boolean("阳包阴加强")
def is_bullish_engulfing_enhanced(open_, high, low, close, lookback=4):
    """
    阳包阴加强版：当前大阳线实体完全包裹前面多根阴线的实体范围
    （前面连续阴线或主要阴线，且今日阳线覆盖它们的最高最低）
    tip：中长期有效，短期效果差
    """
    if len(close) < lookback + 1:
        return False
    # 当前必须是阳线
    if close[-1] <= open_[-1]:
        return False
    # 检查前lookback根K线：是否多数为阴线，且其最高最低被今日覆盖
    prev_highs = high[-lookback-1:-1]
    prev_lows = low[-lookback-1:-1]
    prev_closes = close[-lookback-1:-1]
    prev_opens = open_[-lookback-1:-1]
    # 前lookback根K线中阴线比例
    bearish_count = sum(1 for i in range(lookback) if prev_closes[i] <= prev_opens[i])
    if bearish_count < lookback * 0.6:   # 至少60%阴线
        return False
    # 今日阳线覆盖前lookback根K线的价格范围
    if high[-1] < np.max(prev_highs) or low[-1] > np.min(prev_lows):
        return False
    # 今日开盘价低于前一根阴线收盘价（向下跳空后拉起更强势，可选）
    return True

@manager_boolean("跳空上扬")
def is_gap_up(open_, high, low, close):
    """
    跳空上扬：向上跳空缺口且今日收阳线（或缺口后继续上涨）
    简化：今日最低价 > 昨日最高价，且今日收盘 > 今日开盘
    tip：高度强势有效，一旦失败，注意止损
    """
    if len(close) < 2:
        return False
    if low[-1] > high[-2] and close[-1] > open_[-1]:
        return True
    return False



@manager_boolean("均线多头排列")
def is_bullish_ma_arrangement(close, ma_periods:list=None):
    """
    均线多头排列：短期均线 > 中期均线 > 长期均线，且所有均线向上（斜率正）
    tip：注意打上回撤；表示上涨区间有效
    """
    if ma_periods is None:
        ma_periods = [5, 10, 20]
    if len(close) < max(ma_periods) + 1:
        return False
    ma_values = []
    for p in ma_periods:
        ma = talib.SMA(close, timeperiod=p)
        if np.isnan(ma[-1]):
            return False
        ma_values.append(ma[-1])
        # 均线方向：当前值大于前一天值
        if len(ma) > 1 and ma[-1] <= ma[-2]:
            return False
    # 检查顺序：ma_values应严格递减（即短>中>长）
    for i in range(len(ma_values)-1):
        if ma_values[i] <= ma_values[i+1]:
            return False
    return True

@manager_boolean("MACD零上二次金叉")
def is_macd_second_golden_cross(close, macdconfig=None):
    """
    MACD零上二次金叉：
    1. DIFF在零轴上方
    2. 之前发生过一次金叉，之后发生死叉，现在再次金叉
    """
    if macdconfig is None:
        macdconfig = {"fastperiod": 8, "slowperiod": 16, "signalperiod": 6}
    diff, dea, _ = talib.MACD(close,
                                 fastperiod=macdconfig.get('fastperiod', 8),
                                 slowperiod=macdconfig.get('slowperiod', 16),
                                 signalperiod=macdconfig.get('signalperiod', 6))
    if len(diff) < 5:
        return False

    # 当前金叉：diff上穿dea
    curr_golden = diff[-2] <= dea[-2] and diff[-1] > dea[-1]
    if not curr_golden:
        return False
    # 当前diff > 0（零轴之上）
    if diff[-1] <= 0:
        return False

    # 寻找之前的第一次金叉和死叉
    # 从后往前扫描，找到第一次死叉前的金叉
    found_first_golden = False
    found_dead = False
    for i in range(len(diff)-3, 1, -1):
        # 死叉：diff上穿dea的反向
        dead_cross = diff[i-1] >= dea[i-1] and diff[i] < dea[i]
        if dead_cross and not found_dead:
            found_dead = True
            continue
        # 金叉
        golden = diff[i-1] <= dea[i-1] and diff[i] > dea[i]
        if golden and not found_first_golden and found_dead is False:
            # 记录第一次金叉，要求当时diff也在零轴之上（可选）
            if diff[i] > 0:
                found_first_golden = True
                break
    return found_first_golden and found_dead

# ---------------------------- 中期趋势类 ----------------------------
@manager_boolean("N周期阳线占优")
def is_bullish_candle_dominant(open_, close, N=12, require_price_rise=True):
    """
    最近N根K线中，阳线数量 > 阴线数量，且收盘价重心上移（可选）
    tip：注意止损；中长期优秀
    """
    if len(close) < N:
        return False
    # 阳线计数
    bullish = sum(1 for i in range(-N, 0) if close[i] > open_[i])
    bearish = N - bullish
    if bullish <= bearish:
        return False
    if require_price_rise:
        # 当前收盘价高于N天前收盘价
        if close[-1] <= close[-N]:
            return False
    return True

@manager_boolean("N周期量价同步走强")
def is_volume_price_sync(open_, close, volume, N=12, M=20):
    """
    最近N根K线：
    1. 阳线数量 > 阴线数量
    2. 阳线成交量均值 > 阴线成交量均值
    3. 近期成交量均值 > 前期成交量均值（前期M天）
    tip：注意下降时的主力伪平台
    """
    if len(close) < max(N, M) or len(volume) < max(N, M):
        return False
    # 阳线和阴线成交量列表
    vol_bullish = []
    vol_bearish = []
    for i in range(-N, 0):
        if close[i] > open_[i]:
            vol_bullish.append(volume[i])
        else:
            vol_bearish.append(volume[i])
    if not vol_bullish or not vol_bearish:
        return False
    # 条件1：阳线数量 > 阴线数量
    if len(vol_bullish) <= len(vol_bearish):
        return False
    # 条件2：阳线均量 > 阴线均量
    if np.mean(vol_bullish) <= np.mean(vol_bearish):
        return False
    # 条件3：近期均量 > 前期均量
    recent_vol_mean = np.mean(volume[-N:])
    prev_vol_mean = np.mean(volume[-M-N:-N]) if len(volume) >= M+N else np.mean(volume[:M])
    if recent_vol_mean <= prev_vol_mean:
        return False
    return True



@manager_boolean("回调缩量上涨放量")
def is_healthy_pullback(open_, close, volume, lookback=20, pullback_ratio=0.5):
    """
    简化版回调缩量上涨放量：
    在最近lookback天内，找到最近的一个高点，然后回调，回调期间成交量萎缩，
    最近一天放量上涨突破前高（或接近前高）

    tip：对压力位和支撑位，箱体  要求过高；阴跌时务必注意
    """
    if len(close) < lookback:
        return False
    # 找最近一个局部高点（从后往前，至少高于前后两天）
    high_idx = -1
    for i in range(-lookback, -2):
        if close[i] > close[i-1] and close[i] > close[i+1]:
            high_idx = i
            break
    if high_idx == -1:
        return False
    # 回调区间：high_idx+1 到 -1
    pullback_start = high_idx + 1
    if pullback_start >= 0:
        return False
    # 回调期间成交量是否缩量（回调期成交量均值 < 高点前5日均量）
    pullback_volumes = volume[pullback_start:]
    prev_volumes = volume[high_idx-5:high_idx] if high_idx-5 >= 0 else volume[:high_idx]
    if len(pullback_volumes) == 0 or len(prev_volumes) == 0:
        return False
    if np.mean(pullback_volumes) >= np.mean(prev_volumes):
        return False
    # 最近一天放量上涨：今日成交量 > 昨日成交量，且今日收盘 > 昨日收盘
    if volume[-1] <= volume[-2] or close[-1] <= close[-2]:
        return False
    # 可选：今日收盘超过前高（或回撤幅度小于pullback_ratio）
    if close[-1] < close[high_idx] * (1 - pullback_ratio):
        return False
    return True

@manager_boolean("底部连续小阳推升")
def is_small_bullish_steps(open_, close, volume=None, N=8, max_gain_pct=2.5, max_loss_pct=2.0):
    """
    底部连续小阳推升（红肥绿瘦）：
    最近N天中，阳线占比 > 70%，每根阳线涨幅 ≤ max_gain_pct，阴线跌幅 ≤ max_loss_pct，
    且整体重心逐步上移（当前收盘价高于N天前收盘价）
    tip：注意下降小平台
    """
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
    # 重心上移
    if close[-1] <= close[-N]:
        return False
    return True

