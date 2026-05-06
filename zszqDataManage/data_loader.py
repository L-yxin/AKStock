import datetime
import logging
import os
import re
import struct
import threading
import time
import zipfile
from functools import wraps

import clickhouse_connect
import pandas as pd
import requests
from tqdm import tqdm

from zszqConfig import clickhouse

def sync_once(func):
    """
    线程安全装饰器：
    1. 函数执行中被重复调用 → 直接返回 "处理中"
    2. 执行完毕后恢复状态
    3. 适配类实例方法
    """
    # 线程锁，保证多线程安全
    lock = threading.Lock()
    is_running = False

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal is_running
        with lock:
            # 正在执行，直接返回
            if is_running:
                return "处理中"
            # 标记为执行中
            is_running = True

        try:
            # 执行原业务逻辑
            return func(*args, **kwargs)
        finally:
            # 无论成功/异常，重置状态
            with lock:
                is_running = False

    return wrapper

class ZSZQDataLoader:
    """
    招商证券数据导入类（从单个通达信VIP数据ZIP包导入，使用生成器减少内存）
    """
    def __init__(self):
        self._config = clickhouse.clickhouseConfig.get_client_config()
        try:
            self.client = clickhouse_connect.get_client(**self._config)
        except Exception as e:
            logging.error(f"无法连接到ClickHouse: {e}")
            raise e

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
                        dt = datetime.datetime(
                            date // 10000,
                            (date % 10000) // 100,
                            date % 100 # 加东八区
                        )
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
            logging.error(f"无效的ZIP文件路径: {zip_path}")
            return
        COLUMN_TYPES = [
            "String",  # code
            "String",  # period
            "String",  # adjust_type
            "DateTime('Asia/Shanghai')",  # datetime 强制东八区
            "Float32",  # open
            "Float32",  # high
            "Float32",  # low
            "Float32",  # close
            "Int64",  # volume
            "Float32"  # amount
        ]
        # 日期范围预处理
        if start_date is not None:
            start_date = pd.Timestamp(start_date) if isinstance(start_date, str) else pd.Timestamp(start_date)
        if end_date is not None:
            end_date = (pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                        if isinstance(end_date, str) else pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))

        batch_size = 50000
        buffer = []  # 存储待插入的DataFrame（每个元素是一个股票的DataFrame）
        total_rows = 0

        for stock_df in self.parse_file_to_df(zip_path):
            if stock_df.empty:
                continue
            stock_df["datetime"] = pd.to_datetime(stock_df["datetime"]) - pd.Timedelta(hours=8)
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
                    logging.info(f"成功插入 {len(combined)} 条记录")
                except Exception as e:
                    logging.error(f"批量插入失败({total_rows}条): {e}")
                buffer.clear()
                total_rows = 0

        # 处理剩余数据
        if buffer:
            combined = pd.concat(buffer, ignore_index=True)
            try:
                self.client.insert_df('KLineData', combined)
                logging.info(f"成功插入 {len(combined)} 条记录")
            except Exception as e:
                logging.error(f"批量插入失败({total_rows}条): {e}")

        logging.info("ZIP文件处理完成")

    def select(self, code: str, period: str, adjust_type: str, start_date, end_date):
        """查询时需传入带前缀的code，例如 'sh600000'"""
        if isinstance(start_date, datetime.datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime.datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        query = """
               SELECT
                code,
                period,
                adjust_type,
                datetime,
                argMax(open, _version) AS open,
                argMax(high, _version) AS high,
                argMax(low, _version) AS low,
                argMax(close, _version) AS close,
                argMax(volume, _version) AS volume,
                argMax(amount, _version) AS amount
            FROM KLineData
            WHERE code = %(code)s
              AND period = %(period)s
              AND adjust_type = %(adjust_type)s
              AND datetime >= %(start_date)s
              AND datetime <= %(end_date)s
            GROUP BY code, period, adjust_type, datetime
            ORDER BY datetime
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

    def downloadRawData(self)->(bool,str):
        # 工具函数：把 "500.26MB" 转成 字节数
        def str_to_bytes(size_str):
            size_str = size_str.strip().upper()
            if "MB" in size_str:
                num = float(re.sub(r'[^\d.]', '', size_str))
                return int(num * 1024 * 1024)  # 转成字节
            elif "GB" in size_str:
                num = float(re.sub(r'[^\d.]', '', size_str))
                return int(num * 1024 * 1024 * 1024)
            else:
                return False, "无效的格式"

        os.makedirs("../data", exist_ok=True)

        with open(os.path.abspath(f"{__file__}\\..\\..\\data\\dataUpdate.txt"), "a+", encoding="utf-8") as f:
            f.seek(0)
            t = f.read()

            # 判断今天是否已经更新过
            if t.strip() != "":
                try:
                    oldTime = datetime.datetime.strptime(t.strip(), "%Y-%m-%d %H:%M:%S")
                    if datetime.datetime.now().date() == oldTime.date():
                        logging.warning("✅ 数据已经是最新的，无需更新")
                        return False, "数据已经是最新的，无需更新"
                except:
                    pass

            # 开始更新逻辑

            logging.info("🚀 开始下载数据...")

            # 获取信息接口
            infoJSURL = (
                    'https://data.tdx.com.cn/vipdoc/_hsjdayinfo.js?t='
                    + str(time.time_ns() // 1_000_000 // 1000)
                    + '&_=' + str(time.time_ns() // 1_000_000)
            )

            try:
                infoJS = requests.get(infoJSURL, timeout=20).text
            except Exception as e:
                logging.error(f"❌ 获取信息失败：{e}")
                return False, "获取数据失败"

            info = re.findall(r'"[^"]+"', infoJS)
            if len(info) != 2:
                logging.error("❌ 获取数据失败，格式不正确")
                return False, "获取数据失败"

            size_str = info[0].strip('"')
            update_time = info[1].strip('"')
            res = {"size": size_str, "time": update_time}

            # 如果时间相同，不需要更新
            if res["time"].strip() == t.strip():
                logging.warning("✅ 数据已是最新，无需更新")
                return False, "数据已是最新，无需更新"

            # ==========================
            # 🔥 下载 + 正确进度条
            # ==========================
            zipUrl = "https://data.tdx.com.cn/vipdoc/hsjday.zip"
            save_path = os.path.abspath(f"{__file__}\\..\\..\\data\\hsjday.zip")
            total_size = str_to_bytes(res["size"])

            logging.info(f"📦 准备下载：{res['size']}")
            response = requests.get(zipUrl, stream=True, timeout=600)

            progress_bar = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc='⏬ 下载 hsjday.zip'
            )

            with open(save_path, "wb") as f1:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f1.write(chunk)
                        progress_bar.update(len(chunk))

            progress_bar.close()
            logging.info("✅ 数据下载完成！")
            f.seek(0)
            f.truncate(0)
            # 记录更新时间
            f.seek(0)
            f.truncate(0)
            f.write(res["time"])
            return True, "数据更新成功"

    def getAllSymbols(self)-> pd.DataFrame:
        """获取所有股票代码"""
        return self.client.query_df("SELECT DISTINCT code FROM KLineData")
    @sync_once
    def syncData(self):
        t, mes =self.downloadRawData()
        if t:
            newDate = self.client.query_df("SELECT datetime  FROM KLineData order by datetime DESC LIMIT 1")
            newDate = newDate["datetime"][0].to_pydatetime()
            self.load_data_from_zip(os.path.abspath(f"{__file__}\\..\\..\\data\\hsjday.zip"), (newDate-datetime.timedelta(days=30)).strftime('%Y-%m-%d'), datetime.datetime.now().strftime('%Y-%m-%d'))
            return "更新完毕"
        return mes

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    loader = ZSZQDataLoader()
    loader.downloadRawData()
    loader.load_data_from_zip("../data/hsjday.zip", (datetime.datetime.now()-datetime.timedelta(days=20)).strftime('%Y-%m-%d'), datetime.datetime.now().strftime('%Y-%m-%d'))
    # loader.load_data_from_zip("../data/hsjday.zip")
    # 查询示例
    df = loader.select('sh000001', '1d', '', '2026-04-12', '2026-04-14')
    print(df["datetime"])

    # symbols = loader.syncData()
    # print(symbols["code"].tolist())
