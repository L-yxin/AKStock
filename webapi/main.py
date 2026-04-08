import json
import logging
from datetime import datetime, timedelta

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from zszqDataManage.data_loader import ZSZQDataLoader

zszq = ZSZQDataLoader()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/kline")
async def websocket_kline(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_text()
        params = json.loads(data)

        action = params.get("action", "history")
        code = params.get("code", "sh000001")
        period = params.get("period", "1d")
        adjust_type = params.get("adjust_type", "")
        start_date = params.get("start_date", "2022-01-01")
        end_date = params.get("end_date", "2026-03-20")

        if action == "history":
            # 自动加 8 小时（将日期字符串转为 datetime 并加上 8 小时）
            # 注意：如果数据库字段是 datetime 类型，需要传递完整的日期时间字符串
            start_dt = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(hours=8)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(hours=8)

            # 格式化回字符串，可根据 select 方法的实际要求调整格式
            # 如果 select 只接受 "YYYY-MM-DD"，则只需提取日期部分；如果接受日期时间，则保留完整
            start_date_adj = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            end_date_adj = end_dt.strftime("%Y-%m-%d %H:%M:%S")

            # 如果 zszq.select 只接受日期格式（不带时间），则改为：
            # start_date_adj = start_dt.strftime("%Y-%m-%d")
            # end_date_adj = end_dt.strftime("%Y-%m-%d")
            # 根据实际需求选择

            df = zszq.select(code, period, adjust_type, start_date_adj, end_date_adj)
            kline_data = []
            if not df.empty:
                for _, row in df.iterrows():
                    timestamp = int(pd.to_datetime(row["datetime"]).timestamp() * 1000)
                    kline_data.append({
                        "timestamp": timestamp,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    })
            await websocket.send_json({
                "type": "history",
                "code": code,
                "period": period,
                "data": kline_data
            })
            await websocket.close()
        else:
            await websocket.close()

    except WebSocketDisconnect:
        logging.info("客户端断开连接")
    except Exception as e:
        logging.error(f"WebSocket 异常: {e}")
        await websocket.close()