from enum import Enum
from typing import Dict, Any


class clickhouseConfig(Enum):
    """ClickHouse数据库配置枚举类"""
    HOST = "192.168.100.128"
    PORT = 8123
    USER = "zszq"
    PASSWORD = "zszq"
    DATABASE = "zszq"
    url = f"http://{HOST}:{PORT}/"

    @classmethod
    def get_client_config(cls) -> Dict[str, Any]:
        """获取clickhouse-connect客户端配置"""
        return {
            'host': cls.HOST.value,
            'port': cls.PORT.value,
            'username': cls.USER.value,
            'password': cls.PASSWORD.value,
            'database': cls.DATABASE.value,
            'compress': True,
            'autogenerate_session_id': False,
            'settings':{
                'max_partitions_per_insert_block': 1000,
                'async_insert': 0,  # 启用异步插入
                'wait_for_async_insert': 0,  # 不等待异步插入完成
                        }
        }