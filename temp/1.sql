SELECT datetime  FROM KLineData order by datetime DESC LIMIT 1;
ALTER TABLE KLineData
MODIFY COLUMN datetime DateTime('Asia/Shanghai');


create table KLineData
(
    code        String comment '股票代码（如600000.SH、000001.SZ）',
    period      String comment 'k线级别（如1min/5min/15min/day）',
    adjust_type String comment '复权级别（如前复权/后复权/不复权）',
    datetime    DateTime('Asia/Shanghai') comment '时间（精确到分钟，格式：YYYY-MM-DD HH:MM:00）',
    open        Decimal(12, 2) comment '开盘价',
    high        Decimal(12, 2) comment '最高价',
    low         Decimal(12, 2) comment '最低价',
    close       Decimal(12, 2) comment '收盘价',
    volume      UInt64 comment '成交量（股）',
    amount      Decimal(16, 2) comment '成交额（元）',
    _version    DateTime64(3) default now64(3) comment '版本号（毫秒级，用于去重取最新）'
)
    engine = ReplacingMergeTree(_version)
        PARTITION BY toYYYYMM(datetime)
        ORDER BY (code, period, adjust_type, datetime)
        SETTINGS index_granularity = 8192, merge_with_ttl_timeout = 3600, storage_policy = 'default';

delete
from KLineData
where true;
