import asyncio
import json
import logging
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from signalTestFrame.Signaltest import Signaltest
from zszqDataManage.data_loader import ZSZQDataLoader
import KLineForm as klf
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
        print(params)
        action = params.get("action", "history")
        code = params.get("code", "")
        period = params.get("period", "")
        adjust_type = params.get("adjust_type", "")
        start_date = params.get("start_date", "2022-01-01")
        end_date = params.get("end_date", "2026-03-20")

        if action == "history":
            # 自动加 8 小时（将日期字符串转为 datetime 并加上 8 小时）
            # 注意：如果数据库字段是 datetime 类型，需要传递完整的日期时间字符串
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

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



@app.websocket("/ws/allSymbols")
async def websocket_allSymbols(websocket: WebSocket):
    await websocket.accept()
    try:
        all_symbols = zszq.getAllSymbols()
        await websocket.send_json(all_symbols["code"].tolist())
    except WebSocketDisconnect:
        logging.info("客户端断开连接")
    finally:
        await websocket.close()

@app.websocket("/ws/syncData")
async def websocket_syncData(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await asyncio.to_thread(zszq.syncData)
        await websocket.send_json(data)
    except WebSocketDisconnect:
        logging.info("客户端断开连接")
    finally:
        await websocket.close()

@app.websocket("/ws/buyingAndSellingIndicator")
async def websocket_buyingAndSellingIndicator(websocket: WebSocket):
    await websocket.accept()
    try:
        buyIndicator = klf.methodInfo.buy()
        sellIndicator = klf.methodInfo.sell()
        await websocket.send_json({
            "buy": buyIndicator.to_json(),
            "sell": sellIndicator.to_json()
        })
    except WebSocketDisconnect:
        logging.info("客户端断开连接")
    finally:
        await websocket.close()


@app.websocket("/ws/getLongShortSignal")
async def websocket_getLongShortSignal(websocket: WebSocket):
    await websocket.accept()
    try :
        params = await websocket.receive_json()
        modelType = params.get("modelType", "history") # "history" or "realtime"
        code = params.get("symbol", "")
        period = params.get("period", "1d")
        adjust_type = params.get("adjust_type", "")
        startTime = params.get("startTime", "")
        endTime = params.get("endTime", "")
        indicators = params.get("indicators", [])
        signal = Signaltest(code, period, adjust_type, startTime, endTime, indicators, modelType)
        res = await asyncio.to_thread(signal.start)
        await websocket.send_json(res)

    except WebSocketDisconnect:
        logging.info("客户端断开连接")
    except Exception as e:
        logging.error(f"WebSocket 错误: {e}")
    finally:
        await websocket.close()

from datetime import datetime, timezone, timedelta

def parse_csv_to_markers(csv_path: str):
    """读取 csv，返回符合前端 addMarkers 格式的 configs 列表"""
    df = pd.read_csv(csv_path)
    # 确保日期列正确解析
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date']  = pd.to_datetime(df['exit_date'])

    configs = []
    for _, row in df.iterrows():
        entry_dt = row['entry_date']
        exit_dt  = row['exit_date']
        entry_price = float(row['entry'])
        exit_price = float(row['exit'])
        shares = int(row['shares'])
        pnl = float(row['pnl'])

        # 买入信号
        configs.append({
            'datetime': entry_dt.strftime('%Y-%m-%d'),   # 日期字符串 "2020-10-20"
            'value': entry_price,
            'type': 'B',
            'mes': f'买入 {shares} 股 @ {entry_price:.2f}\n预期盈亏 {pnl:.2f}'
        })
        # 卖出信号
        configs.append({
            'datetime': exit_dt.strftime('%Y-%m-%d'),
            'value': exit_price,
            'type': 'S',
            'mes': f'卖出 {shares} 股 @ {exit_price:.2f}\n实际盈亏 {pnl:.2f}'
        })

    # 按日期和时间顺序排序（同一天买入在前）
    configs.sort(key=lambda x: (x['datetime'], 0 if x['type'] == 'B' else 1))
    return configs


@app.websocket("/ws/getTradingSignals")
async def websocket_getTradingSignals(websocket: WebSocket):
    await websocket.accept()
    try:
        csv_path = "D:/Users/lyx/Desktop/量化/AKStock/strategy/strategy10/ml_trades_20260502_114246.csv"
        configs = parse_csv_to_markers(csv_path)

        payload = {
            'market': 'stock',
            'configs': configs
        }
        await websocket.send_json(payload)
        print("交易信号已发送，共", len(configs), "个标记")
        await asyncio.sleep(5)
    except Exception as e:
        await websocket.send_json({'error': str(e)})
    finally:
        await websocket.close()