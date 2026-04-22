import importlib
import logging
from datetime import datetime
from typing import Union, List, Dict, Any, Optional

import numpy as np
import pybroker

from strategy.dataTransfer.DataTransfer import ZszqDataSource
from tool.inspectFuncArgsAndInfo import inspect_func_args_and_info

logging.basicConfig(level=logging.INFO)


class Signaltest:
    def __init__(
        self,
        code: str,
        period: str = '1d',
        adjust_type: str = '',
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None,
        signals: Optional[List[Dict[str, Any]]] = None,
        modelType: str = "history"
    ):
        # 参数校验（同原代码，省略...）
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

    def _load_indicator_functions(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        loaded = []
        for sig in signals:
            method_name = sig.get("method")
            sig_type = sig.get("type", "buy").lower()
            # info = sig.get("info", method_name)
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
                    "message":raw_func.__message__,
                    "user_params": user_params,
                    "func": raw_func,
                    "param_defaults": param_defaults,
                    "func_info": func_info,
                })
            except (ModuleNotFoundError, AttributeError) as e:
                logging.error(f"加载指标 {method_name} 失败: {e}")
                continue
        return loaded

    def _run(self, ctx: pybroker.context.ExecContext):
        current_time: datetime = ctx.dt

        base_data = {
            "open_": np.array(ctx.open),
            "open": np.array(ctx.open),
            "high": np.array(ctx.high),
            "low": np.array(ctx.low),
            "close": np.array(ctx.close),
            "volume": np.array(ctx.volume),
            "date": np.array(ctx.date),
        }

        for ind in self._indicator_funcs:
            func = ind["func"]
            user_params = ind["user_params"]
            func_info = ind["func_info"]

            kwargs = {}
            skip = False

            for param_meta in func_info:
                name = param_meta["name"]
                required = param_meta["required"]

                # 1. 基础数据（必需）
                if name in base_data:
                    kwargs[name] = base_data[name]
                    continue

                # 2. 用户显式传入的参数（直接传递，不做任何特殊生成）
                if name in user_params:
                    kwargs[name] = user_params[name]
                    continue

                # 3. 非必填参数且用户未提供 → 不传递，让函数使用默认值
                if not required:
                    continue

                # 4. 必填参数缺失 → 跳过该指标
                if required:
                    logging.debug(f"指标 {ind['method']} 缺少必填参数 '{name}'，跳过")
                    skip = True
                    break

            if skip:
                continue

            try:
                result = func(**kwargs)
                if result:
                    self.results.append({
                        "datetime": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": self.code,
                        "period": self.period,
                        "adjust_type": self.adjust_type,
                        "method": ind["method"],
                        "type": ind["type"],
                        "info": ind["info"],
                        "message":ind["message"]
                    })
            except Exception as e:
                logging.warning(f"指标 {ind['method']} 计算失败 at {current_time}: {e}")

    def start(self) -> List[Dict[str, Any]]:
        config = pybroker.StrategyConfig(initial_cash=1_000_000, fee_amount=0.0001)
        strategy = pybroker.Strategy(ZszqDataSource(), self.start_date, self.end_date, config)
        strategy.add_execution(self._run, [self.code])

        if self.modelType == "history":
            strategy.backtest(timeframe=self.period, adjust=self.adjust_type)
            return self.results
        else:
            logging.info("启动实时信号监控...")
            try:
                strategy.run(timeframe=self.period, adjust=self.adjust_type)
            except KeyboardInterrupt:
                logging.info("实时监控已停止")
            return self.results

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