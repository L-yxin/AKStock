from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
from typing import Any, Self
from pydantic import validate_call


class OverZero(BaseModel):
    value: int

    def __init__(self, **data):
        # 自定义初始化校验
        if 'value' in data:
            self._validate_value(data['value'])
        super().__init__(**data)

    @classmethod
    def _validate_value(cls, v: any) -> None:
        if v <= 0:
            raise ValueError(f"Value must be > 0, got {v}")

    @classmethod
    def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        # 获取模型自身的模式（无约束，因为没使用Field）
        inner_schema = handler(cls)

        def ensure_over_zero(v: Any) -> Any:
            if isinstance(v, int):
                return cls(value=v)
            return v

        return core_schema.no_info_before_validator_function(
            ensure_over_zero,
            inner_schema,
        )




@validate_call
def func(a: OverZero):
    print("✅ passed:", a.value)

func(5)               # ✅ passed: 5
func(OverZero(value=10))  # ✅ passed: 10
func(0)