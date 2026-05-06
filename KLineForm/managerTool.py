from typing import Any, List, Tuple

import numpy as np
from pydantic import BaseModel, Field, GetCoreSchemaHandler, field_validator
from pydantic import validate_call
from pydantic_core import CoreSchema, core_schema
from pydantic_core.core_schema import ValidationInfo


class ManagerBoolean:
    __oldFunc__= None
    __message__ =None
    __original_return_value__= None
    def __init__(self, value: str, result: bool):
        self.value = value
        self.result = result          # 最新日的 bool 值

    def __bool__(self):
        return self.result

    def __str__(self):
        return f"{self.value}:{self.result}"


def manager_boolean(value: str, boolFunc=None):
    """
    装饰器：
    - 被装饰函数返回 numpy 数组 (与输入等长)
    - boolFunc: 如何从原始数组提取最新 bool 值，若不提供则使用整个数组的最后布尔值
    - 附加 __original_return_value__ 属性保存原始数组
    """
    def wrapper(func):
        # 参数校验包装
        func_validated = validate_call(func, config={'arbitrary_types_allowed': True})

        def inner(*args, **kwargs):
            # 调用原始函数，获取完整的数组结果
            raw_array = func_validated(*args, **kwargs)
            # 处理可能返回整型数组的情况，转换为 bool
            if np.issubdtype(raw_array.dtype, np.integer):
                raw_array = raw_array.astype(bool)

            # 保存原始返回值供外部使用
            inner.__original_return_value__ = raw_array

            # 使用 boolFunc 计算最新 bool，如果未提供则取最后一个元素
            if boolFunc is not None:
                latest_bool = boolFunc(raw_array)
            else:
                latest_bool = raw_array[-1] if len(raw_array) > 0 else False

            res = ManagerBoolean(value, bool(latest_bool))
            res.__oldFunc__ = func
            res.__message__ = value
            res.__original_return_value__ = inner.__original_return_value__
            return res

        inner.__message__ = value
        inner.__oldFunc__ = func
        inner.__original_return_value__ = None   # 初始占位
        return inner
    return wrapper




class MacdConfig(BaseModel):
    value: dict ={"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}

    @field_validator('value', mode='before')
    @classmethod
    def parse_and_validate(cls, v: Any, info: ValidationInfo) -> dict:
        if v is None:
            return {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}

        # 处理字符串 "12,26,9"
        if isinstance(v, str):
            parts = v.split(',')
            if len(parts) != 3:
                raise ValueError(f"value must be 'fastperiod,slowperiod,signalperiod', got {v}")
            try:
                v = {
                    "fastperiod": int(parts[0]),
                    "slowperiod": int(parts[1]),
                    "signalperiod": int(parts[2])
                }
            except ValueError:
                raise ValueError(f"All parts must be integers, got {v}")

        # 以下是对字典的校验（无论原始输入是 dict 还是转换后的 dict）
        if not isinstance(v, dict):
            raise ValueError(f"value must be dict or comma-separated string, got {type(v)}")

        required_keys = {"fastperiod", "slowperiod", "signalperiod"}
        if set(v.keys()) != required_keys:
            raise ValueError(f"keys must be exactly {required_keys}, got {set(v.keys())}")

        for key, val in v.items():
            if not isinstance(val, int):
                raise ValueError(f"value of '{key}' must be int, got {type(val)}")
            if val <= 0:
                raise ValueError(f"value of '{key}' must be > 0, got {val}")

        return v

    @classmethod
    def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        # 获取模型自身的模式（无约束，因为没使用Field）
        inner_schema = handler(cls)

        def ensure_over_zero(v: Any) -> Any:
            if isinstance(v, str):
                return cls(value=v)
            return v

        return core_schema.no_info_before_validator_function(
            ensure_over_zero,
            inner_schema,
        )



class MaPeriodsConfig(BaseModel):
    value: List[int] = Field(default=[5, 10, 20], description="均线周期列表，必须为正整数")

    @field_validator('value', mode='before')
    @classmethod
    def parse_and_validate(cls, v: Any, info: ValidationInfo) -> List[int]:
        if v is None:
            return [5, 10, 20]
        # 处理字符串 "5,10,20"
        if isinstance(v, str):
            parts = v.split(',')
            if not parts:
                raise ValueError("Empty string, expected comma-separated integers")
            try:
                periods = [int(p.strip()) for p in parts]
            except ValueError:
                raise ValueError(f"All parts must be integers, got '{v}'")
            v = periods


        # 校验列表
        if not isinstance(v, list):
            raise ValueError(f"value must be list of ints or comma-separated string, got {type(v)}")
        if not v:
            raise ValueError("List cannot be empty")
        for idx, p in enumerate(v):
            if not isinstance(p, int):
                raise ValueError(f"Element {idx} must be int, got {type(p)}")
            if p <= 0:
                raise ValueError(f"Element {idx} ({p}) must be > 0")
        return v

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        inner_schema = handler(cls)
        def ensure_list(v: Any) -> Any:
            if isinstance(v, str) or isinstance(v, list):
                return cls(value=v)
            return v
        return core_schema.no_info_before_validator_function(
            ensure_list,
            inner_schema,
        )


class MaPairsConfig(BaseModel):
    value: List[Tuple[int, int]] = Field(
        default=[(5, 10), (10, 20)],
        description="均线对列表，每个元素为 (短期周期, 长期周期)，周期必须为正整数"
    )

    @field_validator('value', mode='before')
    @classmethod
    def parse_and_validate(cls, v: Any, info: ValidationInfo) -> List[Tuple[int, int]]:
        # 处理 None -> 默认值
        if v is None:
            return [(5, 10), (10, 20)]

        # 处理字符串，格式如 "5,10;10,20;20,60"
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Empty string, expected pairs separated by ';' and numbers by ','")
            pairs_str = v.split(';')
            result = []
            for idx, pair_str in enumerate(pairs_str):
                parts = pair_str.split(',')
                if len(parts) != 2:
                    raise ValueError(f"Each pair must contain exactly 2 integers, got '{pair_str}'")
                try:
                    short = int(parts[0].strip())
                    long = int(parts[1].strip())
                except ValueError:
                    raise ValueError(f"All parts must be integers, got '{pair_str}'")
                if short <= 0 or long <= 0:
                    raise ValueError(f"Periods must be > 0, got ({short}, {long})")
                result.append((short, long))
            v = result

        # 处理列表形式（可能是列表的列表或列表的元组）
        if isinstance(v, list):
            result = []
            for idx, item in enumerate(v):
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    try:
                        short = int(item[0])
                        long = int(item[1])
                    except (TypeError, ValueError):
                        raise ValueError(f"Item {idx} must contain two integers, got {item}")
                    if short <= 0 or long <= 0:
                        raise ValueError(f"Periods must be > 0, got ({short}, {long})")
                    result.append((short, long))
                else:
                    raise ValueError(f"Each element must be a pair (list/tuple of length 2), got {item}")
            v = result
        else:
            raise ValueError(f"value must be string, list of pairs, or None, got {type(v)}")

        # 最终校验非空
        if not v:
            raise ValueError("Pairs list cannot be empty")
        return v

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        inner_schema = handler(cls)

        def ensure_pairs(v: Any) -> Any:
            # 如果是字符串或列表，自动包装为 MaPairsConfig 实例
            if isinstance(v, str) or isinstance(v, list):
                return cls(value=v)
            return v

        return core_schema.no_info_before_validator_function(
            ensure_pairs,
            inner_schema,
        )




class RsiConfig(BaseModel):
    value: List[int] = Field(default=[6, 12, 24], description="RSI 周期列表，必须为正整数")

    @field_validator('value', mode='before')
    @classmethod
    def parse_and_validate(cls, v: Any) -> List[int]:
        if v is None:
            return [6, 12, 24]
        # 处理字符串 "6,12,24"
        if isinstance(v, str):
            parts = v.split(',')
            if not parts:
                raise ValueError("Empty string, expected comma-separated integers")
            try:
                periods = [int(p.strip()) for p in parts]
            except ValueError:
                raise ValueError(f"All parts must be integers, got '{v}'")
            v = periods

        # 校验列表
        if not isinstance(v, list):
            raise ValueError(f"value must be list of ints or comma-separated string, got {type(v)}")
        if not v:
            raise ValueError("List cannot be empty")
        for idx, p in enumerate(v):
            if not isinstance(p, int):
                raise ValueError(f"Element {idx} must be int, got {type(p)}")
            if p <= 0:
                raise ValueError(f"Element {idx} ({p}) must be > 0")
        return v

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        inner_schema = handler(cls)
        def ensure_list(v: Any) -> Any:
            if isinstance(v, str) or isinstance(v, list):
                return cls(value=v)
            return v
        return core_schema.no_info_before_validator_function(
            ensure_list,
            inner_schema,
        )


# 测试
@validate_call
def func(a: MaPairsConfig=None):
    print("✅ passed:", a.value)

if __name__ == "__main__":
    # 字典输入
    func("5,10;10,20;20,60")

    func([[5,10], [10,20]])
    func([(5,10), (10,20)])
