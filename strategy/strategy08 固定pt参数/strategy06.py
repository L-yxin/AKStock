import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import talib
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import pybroker
from pybroker import Strategy, StrategyConfig
from strategy.dataTransfer.DataTransfer import ZszqDataSource
from zszqDataLoader import ZSZQDataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    'ma_5_dist', 'ma_10_dist', 'ma_20_dist', 'ma_60_dist',
    'volatility_10', 'atr_14',
    'volume_ratio', 'volume_trend',
    'macd', 'macd_signal', 'macd_hist', 'macd_diff',
    'macd8', 'macd8_signal', 'macd8_hist', 'macd8_diff',
    'rsi_6', 'rsi_14', 'rsi_24',
    'rsi_oversold_6', 'rsi_oversold_12', 'rsi_oversold_24',
    'k', 'd', 'j',
    'bb_width', 'bb_position',
    'bullish_engulfing', 'bearish_engulfing',
    'morning_star', 'evening_star',
    'hammer', 'hanging_man',
    'three_white_soldiers', 'three_black_crows',
    'ma5_cross_up_ma10', 'ma5_cross_down_ma10'
]

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=2, dropout=0.2359):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out).squeeze(-1)

class FeatureEngineer:
    @staticmethod
    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(np.float64)
        o, h, l, c, v = df['open'].values, df['high'].values, df['low'].values, df['close'].values, df['volume'].values

        for p in [5, 10, 20, 60]:
            ma = talib.SMA(c, timeperiod=p)
            df[f'ma_{p}'] = ma
            df[f'ma_{p}_dist'] = c / ma - 1.0
        df['volatility_10'] = df['close'].pct_change().rolling(10).std()
        df['atr_14'] = talib.ATR(h, l, c, timeperiod=14) / c
        df['volume_ma_5'] = talib.SMA(v, timeperiod=5)
        df['volume_ratio'] = v / df['volume_ma_5']
        df['volume_ma_20'] = talib.SMA(v, timeperiod=20)
        df['volume_trend'] = v / df['volume_ma_20'] - 1

        macd, signal, hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'], df['macd_signal'], df['macd_hist'], df['macd_diff'] = macd, signal, hist, macd - signal
        macd8, signal8, hist8 = talib.MACD(c, fastperiod=8, slowperiod=16, signalperiod=6)
        df['macd8'], df['macd8_signal'], df['macd8_hist'], df['macd8_diff'] = macd8, signal8, hist8, macd8 - signal8

        df['rsi_6'] = talib.RSI(c, timeperiod=6)
        df['rsi_14'] = talib.RSI(c, timeperiod=14)
        df['rsi_24'] = talib.RSI(c, timeperiod=24)
        df['rsi_oversold_6'] = (df['rsi_6'] < 30).astype(int)
        df['rsi_oversold_12'] = (talib.RSI(c, timeperiod=12) < 30).astype(int)
        df['rsi_oversold_24'] = (df['rsi_24'] < 30).astype(int)

        low_n, high_n = talib.MIN(l, timeperiod=9), talib.MAX(h, timeperiod=9)
        rsv = (c - low_n) / (high_n - low_n + 1e-9) * 100
        k, d = talib.EMA(rsv, timeperiod=3), talib.EMA(k, timeperiod=3)
        df['k'], df['d'], df['j'] = k, d, 3*k - 2*d

        upper, middle, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['bb_width'] = (upper - lower) / middle
        df['bb_position'] = (c - lower) / (upper - lower + 1e-9)

        df['bullish_engulfing'] = (talib.CDLENGULFING(o, h, l, c) == 100).astype(int)
        df['bearish_engulfing'] = (talib.CDLENGULFING(o, h, l, c) == -100).astype(int)
        df['morning_star'] = (talib.CDLMORNINGSTAR(o, h, l, c, penetration=0) == 100).astype(int)
        df['evening_star'] = (talib.CDLEVENINGSTAR(o, h, l, c, penetration=0.3) == -100).astype(int)
        df['hammer'] = (talib.CDLHAMMER(o, h, l, c) == 100).astype(int)
        df['hanging_man'] = (talib.CDLHANGINGMAN(o, h, l, c) == -100).astype(int)
        df['three_white_soldiers'] = (talib.CDL3WHITESOLDIERS(o, h, l, c) == 100).astype(int)
        df['three_black_crows'] = (talib.CDL3BLACKCROWS(o, h, l, c) == -100).astype(int)

        ma5, ma10 = talib.SMA(c, timeperiod=5), talib.SMA(c, timeperiod=10)
        ma5_s, ma10_s = pd.Series(ma5, index=df.index), pd.Series(ma10, index=df.index)
        df['ma5_cross_up_ma10'] = ((ma5_s.shift(1) <= ma10_s.shift(1)) & (ma5_s > ma10_s)).astype(int)
        df['ma5_cross_down_ma10'] = ((ma5_s.shift(1) >= ma10_s.shift(1)) & (ma5_s < ma10_s)).astype(int)

        df['target'] = df['close'].shift(-5) / df['close'] - 1.0
        return df


# ---------------------- 直接加载最佳模型的策略 ----------------------
class MLStrategy:
    def __init__(self, symbol, start_date, end_date,
                 buy_thresh=0.00225, sell_thresh=-0.00115,
                 atr_stop=1.855, atr_profit=1.038,
                 weight_path='lstm_best_fixed_threshold.pt'):
        self.symbol = symbol
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.buy_thresh = buy_thresh
        self.sell_thresh = sell_thresh
        self.atr_stop = atr_stop
        self.atr_profit = atr_profit

        # 加载模型
        self.model = LSTMModel(
            input_dim=len(FEATURE_COLS),
            hidden_dim=256,  # 与权重一致
            num_layers=3,  # 与权重一致
            dropout=0.222  # 与权重一致
        ).to(device)
        self.model.load_state_dict(torch.load(weight_path, map_location=device))
        self.model.eval()
        self.trained = True
        self.seq_len = 40
        self.feature_buffer = torch.zeros((self.seq_len, len(FEATURE_COLS)), device=device)
        self.buffer_fill = 0
        self.max_price = None
        self.ret_buffer = deque(maxlen=10)

    def exec_fn(self, ctx: pybroker.context.ExecContext):
        if ctx.bars < 70:
            return
        open_ = np.asarray(ctx.open, dtype=np.float64)
        high = np.asarray(ctx.high, dtype=np.float64)
        low = np.asarray(ctx.low, dtype=np.float64)
        close = np.asarray(ctx.close, dtype=np.float64)
        volume = np.asarray(ctx.volume, dtype=np.float64)

        # 计算波动率
        if len(close) >= 2:
            daily_ret = (close[-1] - close[-2]) / close[-2]
        else:
            daily_ret = 0.0
        self.ret_buffer.append(daily_ret)
        volatility_10 = np.std(self.ret_buffer) if len(self.ret_buffer) == 10 else 0.0

        # 计算所有特征（与训练时完全一致）
        feature_vals = []
        for p in [5, 10, 20, 60]:
            ma = talib.SMA(close, timeperiod=p)
            feature_vals.append(close[-1] / ma[-1] - 1.0)
        feature_vals.append(volatility_10)
        atr = talib.ATR(high, low, close, timeperiod=14)
        feature_vals.append(atr[-1] / close[-1])
        vol_ma5 = talib.SMA(volume, timeperiod=5)
        feature_vals.append(volume[-1] / vol_ma5[-1])
        vol_ma20 = talib.SMA(volume, timeperiod=20)
        feature_vals.append(volume[-1] / vol_ma20[-1] - 1.0)
        macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        feature_vals.extend([macd[-1], signal[-1], hist[-1], macd[-1] - signal[-1]])
        macd8, signal8, hist8 = talib.MACD(close, fastperiod=8, slowperiod=16, signalperiod=6)
        feature_vals.extend([macd8[-1], signal8[-1], hist8[-1], macd8[-1] - signal8[-1]])
        rsi6 = talib.RSI(close, timeperiod=6)
        rsi14 = talib.RSI(close, timeperiod=14)
        rsi24 = talib.RSI(close, timeperiod=24)
        feature_vals.extend([rsi6[-1], rsi14[-1], rsi24[-1]])
        feature_vals.append(1 if rsi6[-1] < 30 else 0)
        rsi12 = talib.RSI(close, timeperiod=12)
        feature_vals.append(1 if rsi12[-1] < 30 else 0)
        feature_vals.append(1 if rsi24[-1] < 30 else 0)
        low_n = talib.MIN(low, timeperiod=9)
        high_n = talib.MAX(high, timeperiod=9)
        rsv = (close - low_n) / (high_n - low_n + 1e-9) * 100
        k = talib.EMA(rsv, timeperiod=3)
        d = talib.EMA(k, timeperiod=3)
        feature_vals.extend([k[-1], d[-1], 3*k[-1] - 2*d[-1]])
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        feature_vals.append((upper[-1] - lower[-1]) / middle[-1])
        feature_vals.append((close[-1] - lower[-1]) / (upper[-1] - lower[-1] + 1e-9))
        engulfing = talib.CDLENGULFING(open_, high, low, close)
        feature_vals.append(1 if engulfing[-1] == 100 else 0)
        feature_vals.append(1 if engulfing[-1] == -100 else 0)
        morning = talib.CDLMORNINGSTAR(open_, high, low, close, penetration=0)
        feature_vals.append(1 if morning[-1] == 100 else 0)
        evening = talib.CDLEVENINGSTAR(open_, high, low, close, penetration=0.3)
        feature_vals.append(1 if evening[-1] == -100 else 0)
        hammer = talib.CDLHAMMER(open_, high, low, close)
        feature_vals.append(1 if hammer[-1] == 100 else 0)
        hanging = talib.CDLHANGINGMAN(open_, high, low, close)
        feature_vals.append(1 if hanging[-1] == -100 else 0)
        three_white = talib.CDL3WHITESOLDIERS(open_, high, low, close)
        feature_vals.append(1 if three_white[-1] == 100 else 0)
        three_black = talib.CDL3BLACKCROWS(open_, high, low, close)
        feature_vals.append(1 if three_black[-1] == -100 else 0)
        ma5 = talib.SMA(close, timeperiod=5)
        ma10 = talib.SMA(close, timeperiod=10)
        cross_up = (ma5[-2] <= ma10[-2] and ma5[-1] > ma10[-1])
        cross_down = (ma5[-2] >= ma10[-2] and ma5[-1] < ma10[-1])
        feature_vals.append(1 if cross_up else 0)
        feature_vals.append(1 if cross_down else 0)

        feature_vec = np.array(feature_vals)
        if np.any(np.isnan(feature_vec)):
            return

        # GPU 缓冲区更新
        feature_tensor = torch.tensor(feature_vec, dtype=torch.float32, device=device)
        if self.buffer_fill < self.seq_len:
            self.feature_buffer[self.buffer_fill] = feature_tensor
            self.buffer_fill += 1
        else:
            self.feature_buffer = torch.roll(self.feature_buffer, -1, dims=0)
            self.feature_buffer[-1] = feature_tensor

        if self.buffer_fill < self.seq_len:
            return

        seq = self.feature_buffer.unsqueeze(0)
        with torch.no_grad():
            pred_return = self.model(seq).item()

        pos = ctx.long_pos()

        # 固定阈值买卖信号
        if pred_return > self.buy_thresh and pos is None:
            shares = ctx.calc_target_shares(1.0)
            ctx.buy_shares = (shares // 100) * 100
            self.max_price = None

        if pos is not None:
            if self.max_price is None:
                self.max_price = float(close[-1])
            else:
                self.max_price = max(self.max_price, float(close[-1]))

            total_cost = sum(e.price * e.shares for e in pos.entries)
            avg_cost = float(total_cost) / float(pos.shares)

            atr_rel = talib.ATR(high, low, close, timeperiod=14)
            if not np.isnan(atr_rel[-1]):
                atr_value = atr_rel[-1] * float(close[-1])
                if float(close[-1]) < avg_cost - self.atr_stop * atr_value:
                    ctx.sell_shares = pos.shares
                    self.max_price = None
                    return
                if float(close[-1]) < self.max_price - self.atr_profit * atr_value:
                    ctx.sell_shares = pos.shares
                    self.max_price = None
                    return

            if pred_return < self.sell_thresh:
                ctx.sell_shares = pos.shares
                self.max_price = None
        else:
            self.max_price = None


if __name__ == '__main__':
    config = StrategyConfig(initial_cash=1_000_000, fee_amount=0.0001)
    strategy = Strategy(ZszqDataSource(), '2020-01-01', '2026-03-20', config)
    ml_strat = MLStrategy('sh000001', '2020-01-01', '2026-03-20',
                          buy_thresh=0.00225, sell_thresh=-0.00115,
                          atr_stop=1.855, atr_profit=1.038,
                          weight_path='../strategy07/best_model_fixed.pt')
    strategy.add_execution(ml_strat.exec_fn, ['sh000001'])
    result = strategy.backtest(timeframe='1d', adjust='')
    print(result.metrics)