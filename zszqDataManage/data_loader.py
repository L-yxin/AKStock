import concurrent
import datetime
import os
import re
import zipfile
import struct

import clickhouse_connect
import pandas as pd

from zszqConfig import clickhouse
import logging


class ZSZQDataLoader:
    """
    招商证券数据导入类（从单个通达信VIP数据ZIP包导入，使用生成器减少内存）
    """
    def __init__(self):
        self._config = clickhouse.clickhouseConfig.get_client_config()
        self.client = clickhouse_connect.get_client(**self._config)

    def parse_file_to_df(self, zip_path: str):
        """
        生成器：解析ZIP文件中的每个.day文件，每次yield一个股票的DataFrame
        :param zip_path: ZIP文件路径
        :yield: pd.DataFrame 包含单只股票的K线数据（列：code, period, adjust_type, datetime,
                open, high, low, close, volume, amount）
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                day_files = [f for f in zf.namelist() if f.endswith('.day')]
                if not day_files:
                    logging.warning(f"ZIP文件 {zip_path} 中未找到.day文件")
                    return

                for day_file in day_files:
                    base_name = os.path.basename(day_file)
                    code_match = re.search(r'(sh|sz|bj)(\d{6})', base_name.lower())
                    if not code_match:
                        logging.warning(f"无法从文件名 {base_name} 提取股票代码，跳过")
                        continue
                    prefix = code_match.group(1)
                    code_num = code_match.group(2)
                    code = f"{prefix}{code_num}"

                    with zf.open(day_file) as f:
                        buffer = f.read()

                    num_bars = len(buffer) // 32
                    if num_bars == 0:
                        continue

                    # ETF 价格精度
                    if (code_num.startswith(('56', '58')) and prefix == 'sh') or \
                       (code_num.startswith('159') and prefix == 'sz'):
                        price_scale = 1000.0
                    else:
                        price_scale = 100.0

                    # 构建该股票的DataFrame
                    rows = []
                    for i in range(num_bars):
                        bar = buffer[i*32 : (i+1)*32]
                        date, open_, high, low, close, amount, vol, _ = struct.unpack("IIIIIfII", bar)
                        dt = datetime.datetime(date // 10000, (date % 10000) // 100, date % 100)
                        rows.append({
                            'code': code,
                            'period': '1d',
                            'adjust_type': '',
                            'datetime': dt,
                            'open': float(open_ / price_scale),
                            'high': float(high / price_scale),
                            'low': float(low / price_scale),
                            'close': float(close / price_scale),
                            'volume': int(vol),
                            'amount': float(amount)
                        })

                    if rows:
                        df = pd.DataFrame(rows)
                        df['open'] = df['open'].astype('float32')
                        df['high'] = df['high'].astype('float32')
                        df['low'] = df['low'].astype('float32')
                        df['close'] = df['close'].astype('float32')
                        df['volume'] = df['volume'].astype('int64')
                        df['amount'] = df['amount'].astype('float32')
                        yield df  # 每次yield一只股票的全部数据

        except Exception as e:
            logging.error(f"解析ZIP文件 {zip_path} 失败: {e}")
            return

    def load_data_from_zip(self, zip_path: str, start_date=None, end_date=None):
        """
        从单个ZIP文件导入数据到ClickHouse（使用生成器逐股票处理）
        :param zip_path: ZIP文件路径
        :param start_date: 起始日期，格式 'YYYY-MM-DD' 或 datetime 对象
        :param end_date: 结束日期，格式 'YYYY-MM-DD' 或 datetime 对象
        """
        if not os.path.isfile(zip_path) or not zip_path.lower().endswith('.zip'):
            print(f"无效的ZIP文件路径: {zip_path}")
            return

        # 日期范围预处理
        if start_date is not None:
            start_date = pd.Timestamp(start_date) if isinstance(start_date, str) else pd.Timestamp(start_date)
        if end_date is not None:
            end_date = (pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                        if isinstance(end_date, str) else pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))

        batch_size = 100000
        buffer = []  # 存储待插入的DataFrame（每个元素是一个股票的DataFrame）
        total_rows = 0

        for stock_df in self.parse_file_to_df(zip_path):
            if stock_df.empty:
                continue

            # 日期过滤
            if start_date is not None:
                stock_df = stock_df[stock_df['datetime'] >= start_date]
            if end_date is not None:
                stock_df = stock_df[stock_df['datetime'] <= end_date]
            if stock_df.empty:
                continue

            buffer.append(stock_df)
            total_rows += len(stock_df)

            # 当累积行数达到batch_size时执行插入
            if total_rows >= batch_size:
                combined = pd.concat(buffer, ignore_index=True)
                try:
                    self.client.insert_df('KLineData', combined)
                    print(f"成功插入 {len(combined)} 条记录")
                except Exception as e:
                    print(f"批量插入失败: {e}")
                buffer.clear()
                total_rows = 0

        # 处理剩余数据
        if buffer:
            combined = pd.concat(buffer, ignore_index=True)
            try:
                self.client.insert_df('KLineData', combined)
                print(f"成功插入 {len(combined)} 条记录")
            except Exception as e:
                print(f"批量插入失败: {e}")

        print("ZIP文件处理完成")

    def select(self, code: str, period: str, adjust_type: str, start_date, end_date):
        """查询时需传入带前缀的code，例如 'sh600000'"""
        if isinstance(start_date, datetime.datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime.datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        query = """
                WITH ranked AS (SELECT *, row_number() OVER (PARTITION BY code, period, adjust_type, datetime ORDER BY _version DESC) AS rn
                                FROM KLineData
                                WHERE code = %(code)s
                                  AND period = %(period)s
                                  AND adjust_type = %(adjust_type)s
                                  AND datetime >= %(start_date)s
                                  AND datetime <= %(end_date)s)
                SELECT code, period, adjust_type, datetime, open, high, low, close, volume, amount
                FROM ranked WHERE rn = 1 ORDER BY datetime
                """
        try:
            df = self.client.query_df(query, parameters={
                'code': code,
                'period': period,
                'adjust_type': adjust_type,
                'start_date': start_date,
                'end_date': end_date
            })
            for col in ['open', 'high', 'low', 'close', 'amount']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            return df
        except Exception as e:
            logging.error(f"查询数据失败: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    loader = ZSZQDataLoader()
    loader.load_data_from_zip(r"D:\Users\lyx\Desktop\hsjday.zip", '2026-01-04', '2026-04-08')
    # 查询示例
    # df = loader.select('sh000001', '1d', '', '2026-04-04', '2026-04-04')
    # print(df)