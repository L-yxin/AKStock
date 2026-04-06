import numpy as np
import talib


def has_recent_surge(close: np.ndarray, lookback: int = 3, surge_threshold: float = 0.07, method='cumulative') -> bool:
    """
       判断最近 lookback 根 K 线是否有价格 surge_threshold 倍以上的涨幅

       参数:
           close (np.ndarray): 收盘价数组
           lookback (int): 回看周期，默认 3 根 K 线
           surge_threshold (float): 涨幅阈值，
           method (str): 计算方法，可选 'cumulative'(累计涨幅) 或 'max'(最大涨幅)

       返回:
           bool: 是否存在超过阈值的涨幅

       说明:
           - cumulative 模式：计算从第一根到最后一根的累计涨幅
           - max 模式：计算看跌周期内的最高价相对最低价的涨幅
       """
    if len(close) < lookback + 1:
        return False
    recent_close = close[-lookback - 1:]
    if method == 'cumulative':
        start = recent_close[0]
        end = recent_close[-1]
        return (end - start) / start > surge_threshold
    elif method == 'max':
        min_price = np.min(recent_close)
        max_price = np.max(recent_close)
        return (max_price - min_price) / min_price > surge_threshold
    else:
        raise ValueError("method must be 'cumulative' or 'max'")

def is_doji(open_, close, high, low, threshold=0.1):
    """
    十字星：实体长度 ≤ (high-low)*threshold，默认10%。
    通常实体很小，上下影线存在。
    """
    if len(open_) < 1 or len(close) < 1:
        return False
    body = abs(close[-1] - open_[-1])
    range_ = high[-1] - low[-1]
    if range_ == 0:
        return False
    return body / range_ <= threshold





