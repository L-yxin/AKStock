"""
K 线形态识别模块

按看涨/看跌分类的 K 线形态识别工具
"""
import numpy as np

from KLineForm import sell, buy, neutral, managerTool
import pandas as pd
from tool.inspectFuncArgsAndInfo import inspect_func_args_and_info
__version__ = '0.1.0'


class methodInfo:
    @staticmethod
    def buy() -> pd.DataFrame:
        data = []
        for i in dir(buy):
            # 跳过特殊方法
            if i.startswith('__') and i.endswith('__'):
                continue
            func = getattr(buy, i)
            if callable(func):
                message = getattr(func, '__message__', None)
                if message is not None:
                    data.append({"method": i, "info": message, "type": "buy", 'enabled': False, 'params': inspect_func_args_and_info(func.__oldFunc__)})
        df = pd.DataFrame(data,
                          columns=["method", "info", "type", "enabled", "params"])
        df = df.astype({
            "method": "string",
            "info": "string",
            "type": "string",
            "enabled": "boolean",
            "params": "object"
        })
        return df

    @staticmethod
    def sell() -> pd.DataFrame:
        data = []
        for i in dir(sell):
            # 跳过特殊方法
            if i.startswith('__') and i.endswith('__'):
                continue
            func = getattr(sell, i)
            if callable(func):
                message = getattr(func, '__message__', None)
                if message is not None:
                    data.append({"method": i, "info": message, "type": "sell", 'enabled': False, 'params': inspect_func_args_and_info(func.__oldFunc__)})
        df = pd.DataFrame(data, columns=["method", 'info', 'type', 'enabled','params'])
        df = df.astype({
            "method": "string",
            "info": "string",
            "type": "string",
            "enabled": "boolean",
            "params": "object"
        })
        return df


__all__ = [
    __version__,
    sell, buy, neutral, managerTool, methodInfo
]

if __name__ == '__main__':
    print(methodInfo.sell())