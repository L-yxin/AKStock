import inspect
from typing import Callable


def inspect_func_args_and_info(func: Callable):
    """
    分析函数传参信息：参数名、类型注解、默认值、是否必填,文档信息
    """
    sig = inspect.signature(func)
    params = {"params": [], "doc": func.__doc__}

    for name, param in sig.parameters.items():
        info = {
            "name": name,
            "annotation": param.annotation if param.annotation != inspect._empty else None,
            "required": param.default is inspect._empty,
            "default": param.default if param.default != inspect._empty else None,
        }
        if info["annotation"]not in [None, inspect.Parameter.empty]:
            info["annotation"] = str(info["annotation"])
        params["params"].append(info)

    return params

__all__ = ["inspect_func_args_and_info"]