from datetime import datetime
from typing import Optional, Any

import pandas as pd

from zszqDataManage import data_loader
import pybroker
from pybroker.data import DataSource


class ZszqDataSource(DataSource):
    def __init__(self):
        self.loader = data_loader.ZSZQDataLoader()
        super().__init__()

    def _fetch_data(self, symbols: frozenset[str], start_date: datetime, end_date: datetime, timeframe: Optional[str],
                    adjust: Optional[Any]) -> pd.DataFrame:

        timeframe = timeframe[0:2]

        dfs = []
        for symbol in symbols:
            df = self.loader.select(
                code=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust_type=adjust,
                period=timeframe
            )
            if not df.empty:
                df.drop(columns=["period", "adjust_type"], inplace=True)
                df.rename(columns={
                    "code": "symbol",
                    "datetime": "date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                    "amount": "amount"
                }, inplace=True)
                dfs.append(df)
        if not dfs:
            return pd.DataFrame(columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount'])
        return pd.concat(dfs, ignore_index=True)

