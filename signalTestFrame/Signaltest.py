import importlib
import logging
from datetime import datetime
from typing import Union, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from tool.inspectFuncArgsAndInfo import inspect_func_args_and_info
from zszqDataLoader import ZSZQDataLoader  # 假设可用

logging.basicConfig(level=logging.INFO)


class Signaltest:
    def __init__(self, code, period='1d', adjust_type='', start_date=None, end_date=None, signals=None,
                 modelType="history"):
        # 参数校验 （同原代码）
        if start_date is None:
            raise ValueError("请传入开始时间")
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date is not None and isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        if signals is None:
            raise ValueError("请传入待测试的信号")
        if modelType not in {"history", "realtime"}:
            raise ValueError("请传入正确的模型类型")
        self.code = code
        self.period = period
        self.adjust_type = adjust_type
        self.start_date = start_date
        self.end_date = end_date if modelType == "history" else datetime.now()
        self.signals = signals
        self.modelType = modelType
        self.results: List[Dict[str, Any]] = []
        self._indicator_funcs = self._load_indicator_functions(signals)

    # _load_indicator_functions 保持原样
    def _load_indicator_functions(self, signals):
        loaded = []
        for sig in signals:
            method_name = sig.get("method")
            sig_type = sig.get("type", "buy").lower()
            user_params = sig.get("params", {})
            if sig_type == "buy":
                module_name = "KLineForm.buy"
            elif sig_type == "sell":
                module_name = "KLineForm.sell"
            else:
                logging.warning(f"未知信号类型 {sig_type}，跳过 {method_name}")
                continue
            try:
                module = importlib.import_module(module_name)
                func = getattr(module, method_name)
                raw_func = func
                func_info = inspect_func_args_and_info(raw_func.__oldFunc__)["params"]
                param_defaults = {}
                for p in func_info:
                    if not p["required"] and p["default"] is not None:
                        param_defaults[p["name"]] = p["default"]
                loaded.append({
                    "method": method_name,
                    "type": sig_type,
                    "info": raw_func.__oldFunc__.__doc__,
                    "message": raw_func.__message__,
                    "user_params": user_params,
                    "func": raw_func,
                    "param_defaults": param_defaults,
                    "func_info": func_info,
                })
            except (ModuleNotFoundError, AttributeError) as e:
                logging.error(f"加载指标 {method_name} 失败: {e}")
                continue
        return loaded

    def _prepare_kwargs(self, func_info, base_data, user_params):
        """构建调用指标函数所需的参数字典"""
        kwargs = {}
        for param_meta in func_info:
            name = param_meta["name"]
            required = param_meta["required"]
            if name in base_data:
                kwargs[name] = base_data[name]
            elif name in user_params:
                kwargs[name] = user_params[name]
            elif not required:
                # 非必填且用户未提供，不传，让函数用默认值
                continue
            else:
                # 必填参数缺失
                return None
        return kwargs

    def start_history_vectorized(self):
        """使用向量化方式计算历史信号"""
        # 加载数据
        loader = ZSZQDataLoader()
        start_str = self.start_date.strftime('%Y-%m-%d')
        end_str = self.end_date.strftime('%Y-%m-%d')
        df = loader.select(self.code, self.period, self.adjust_type, start_str, end_str)
        if df.empty:
            logging.warning("没有获取到数据")
            return self.results
        # 标准化列名
        if 'datetime' in df.columns:
            df['date'] = pd.to_datetime(df['datetime'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            raise ValueError("数据中缺少日期列")
        df = df.set_index('date')
        # 确保所需的OHLCV列存在
        needed = ['open', 'high', 'low', 'close', 'volume']
        for col in needed:
            if col not in df.columns:
                raise ValueError(f"数据缺少列 {col}")

        # 准备基础数据数组
        base_data = {
            "open_": df['open'].values,
            "open": df['open'].values,
            "high": df['high'].values,
            "low": df['low'].values,
            "close": df['close'].values,
            "volume": df['volume'].values,
            "date": df.index.values,  # 暂不需要
        }

        dates = df.index
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values

        # 对每个指标计算信号
        for ind in self._indicator_funcs:
            func = ind["func"]
            user_params = ind["user_params"]
            func_info = ind["func_info"]

            kwargs = self._prepare_kwargs(func_info, base_data, user_params)
            if kwargs is None:
                continue

            try:
                # 调用指标函数，获得带装饰器的对象
                result_obj = func(**kwargs)
                # 获取原始返回值数组
                raw_arr = result_obj.__original_return_value__
                # 将原始数组转换为布尔信号掩码
                if np.issubdtype(raw_arr.dtype, np.bool_):
                    signal_mask = raw_arr
                else:
                    # 数值类型，非零值为信号（或大于0，根据信号类型）
                    # buy 信号通常为正，sell 信号可能为负，但这里我们用非零判断
                    signal_mask = raw_arr >0 if ind["type"] == "buy" else raw_arr <0

                # 遍历信号发生的索引
                true_idx = np.where(signal_mask)[0]
                for idx in true_idx:
                    self.results.append({
                        "datetime": dates[idx].strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": self.code,
                        "period": self.period,
                        "adjust_type": self.adjust_type,
                        "method": ind["method"],
                        "type": ind["type"],
                        "info": ind["info"],
                        "message": ind["message"],
                        "open": float(opens[idx]),
                        "high": float(highs[idx]),
                        "low": float(lows[idx]),
                        "close": float(closes[idx]),
                        "volume": int(volumes[idx])
                    })
            except Exception as e:
                logging.warning(f"指标 {ind['method']} 计算失败: {e}")

        return self.results

    def start(self):
        if self.modelType == "history":
            return self.start_history_vectorized()
        else:
            # 实时模式保留原 pybroker 实现，或者提示不支持
            logging.error("当前版本不支持实时模式的向量化加速，请使用 history 模式")
            return []

if __name__ == "__main__":


    signals = [
        {
            'method': 'is_bullish_candle_dominant',
            'type': 'buy',
            'params': {'N': 14, 'require_price_rise': True}
        },
        {
            'method': 'is_moving_average_up',
            'type': 'buy',
            'params': {'ma_periods': "5,10"}   # 直接传字符串，自动转换
        }
    ]
    tester = Signaltest("sh000001", "1d", "", "2022-01-01", "2026-03-20", signals=signals)
    results = tester.start()
    print(results)