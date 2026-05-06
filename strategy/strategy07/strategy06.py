import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from collections import deque
import talib

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

import pybroker
from pybroker import Strategy, StrategyConfig
from strategy.dataTransfer.DataTransfer import ZszqDataSource
from zszqDataLoader import ZSZQDataLoader

# ---------------------- 设备 ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ---------------------- 固定特征名称 ----------------------
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

# ---------------------- LSTM 模型 ----------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=2, dropout=0.2359):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
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


# ---------------------- 特征工程 ----------------------
class FeatureEngineer:
    @staticmethod
    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(np.float64)

        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        v = df['volume'].values

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
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        df['macd_diff'] = macd - signal

        macd8, signal8, hist8 = talib.MACD(c, fastperiod=8, slowperiod=16, signalperiod=6)
        df['macd8'] = macd8
        df['macd8_signal'] = signal8
        df['macd8_hist'] = hist8
        df['macd8_diff'] = macd8 - signal8

        df['rsi_6'] = talib.RSI(c, timeperiod=6)
        df['rsi_14'] = talib.RSI(c, timeperiod=14)
        df['rsi_24'] = talib.RSI(c, timeperiod=24)

        df['rsi_oversold_6'] = (df['rsi_6'] < 30).astype(int)
        df['rsi_oversold_12'] = (talib.RSI(c, timeperiod=12) < 30).astype(int)
        df['rsi_oversold_24'] = (df['rsi_24'] < 30).astype(int)

        low_n = talib.MIN(l, timeperiod=9)
        high_n = talib.MAX(h, timeperiod=9)
        rsv = (c - low_n) / (high_n - low_n + 1e-9) * 100
        k = talib.EMA(rsv, timeperiod=3)
        d = talib.EMA(k, timeperiod=3)
        df['k'] = k
        df['d'] = d
        df['j'] = 3 * k - 2 * d

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

        ma5 = talib.SMA(c, timeperiod=5)
        ma10 = talib.SMA(c, timeperiod=10)
        ma5_s = pd.Series(ma5, index=df.index)
        ma10_s = pd.Series(ma10, index=df.index)
        df['ma5_cross_up_ma10'] = ((ma5_s.shift(1) <= ma10_s.shift(1)) & (ma5_s > ma10_s)).astype(int)
        df['ma5_cross_down_ma10'] = ((ma5_s.shift(1) >= ma10_s.shift(1)) & (ma5_s < ma10_s)).astype(int)

        df['target'] = df['close'].shift(-5) / df['close'] - 1.0
        return df


# ---------------------- LSTM 训练器（使用固定阈值） ----------------------
class LSTMTrainer:
    def __init__(self, input_dim, params, buy_thresh=0.00225, sell_thresh=-0.00115):
        self.model = LSTMModel(
            input_dim,
            hidden_dim=params.get('hidden_dim', 512),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.2359)
        ).to(device)
        self.lr = params.get('lr', 0.00267)
        self.epochs = params.get('epochs', 50)
        self.batch_size = params.get('batch_size', 64)
        # 直接使用传入的固定阈值
        self.buy_thresh = buy_thresh
        self.sell_thresh = sell_thresh

    def train(self, df_features, df_target, seq_len=40, verbose=True):
        df = df_features.copy()
        df['target'] = df_target
        df = df.dropna()
        n_samples = len(df) - seq_len
        if n_samples < 200:
            raise RuntimeError(f"数据不足，当前只有{n_samples}条")

        features = df[FEATURE_COLS].values.astype(np.float32)
        targets = df['target'].values.astype(np.float32)

        X_all = torch.tensor(features, device=device)
        y_all = torch.tensor(targets, device=device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        scaler = GradScaler(device.type)

        total_samples = len(features) - seq_len
        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(total_samples, device=device)
            epoch_loss = 0.0
            for start in range(0, total_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = perm[start:end]
                xb = torch.stack([X_all[i:i+seq_len] for i in batch_idx])
                yb = y_all[batch_idx + seq_len]

                optimizer.zero_grad()
                with autocast(device.type):
                    pred = self.model(xb)
                    loss = loss_fn(pred, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item() * len(batch_idx)

            avg_loss = epoch_loss / total_samples
            if verbose and (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")

        print(f"训练完成，使用固定阈值 => 买入: {self.buy_thresh:.5f}, 卖出: {self.sell_thresh:.5f}")

    def predict(self, feature_seq):
        if isinstance(feature_seq, np.ndarray):
            feature_seq = torch.as_tensor(feature_seq, dtype=torch.float32, device=device)
        if feature_seq.device != device:
            feature_seq = feature_seq.to(device)
        if feature_seq.dim() == 2:
            feature_seq = feature_seq.unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(feature_seq).item()
        return pred


# ---------------------- 策略 ----------------------
class MLStrategy:
    def __init__(self, symbol, start_date, end_date, params=None, seq_len=40,
                 atr_stop=1.855, atr_profit=1.038,
                 buy_thresh=0.00225, sell_thresh=-0.00115):
        self.symbol = symbol
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.params = params if params else {}
        self.seq_len = seq_len
        self.atr_stop = atr_stop
        self.atr_profit = atr_profit
        self.buy_thresh = buy_thresh
        self.sell_thresh = sell_thresh
        self.model = None
        self.trained = False
        self.feature_buffer = torch.zeros((seq_len, len(FEATURE_COLS)), device=device)
        self.buffer_fill = 0
        self.max_price = None
        self.ret_buffer = deque(maxlen=10)

    def _train_if_needed(self, verbose=True):
        if self.trained:
            return
        loader = ZSZQDataLoader()
        train_end = self.start_date - timedelta(days=1)
        train_start = train_end - timedelta(days=365 * 5)
        df = loader.select(self.symbol, '1d', '',
                           train_start.strftime('%Y-%m-%d'),
                           train_end.strftime('%Y-%m-%d'))
        if df.empty:
            print("训练数据为空")
            return
        df = df.sort_values('datetime')
        fe = FeatureEngineer()
        df_feat = fe.compute_features(df)
        target = df_feat['target']
        features_df = df_feat[FEATURE_COLS]

        input_dim = len(FEATURE_COLS)
        self.model = LSTMTrainer(input_dim, self.params,
                                 buy_thresh=self.buy_thresh,
                                 sell_thresh=self.sell_thresh)
        self.model.train(features_df, target, seq_len=self.seq_len, verbose=verbose)
        self.trained = True

    def exec_fn(self, ctx: pybroker.context.ExecContext):
        self._train_if_needed()
        if not self.trained:
            return

        if ctx.bars < 70:
            return

        open_ = np.asarray(ctx.open, dtype=np.float64)
        high = np.asarray(ctx.high, dtype=np.float64)
        low = np.asarray(ctx.low, dtype=np.float64)
        close = np.asarray(ctx.close, dtype=np.float64)
        volume = np.asarray(ctx.volume, dtype=np.float64)

        # 波动率
        if len(close) >= 2:
            daily_ret = (close[-1] - close[-2]) / close[-2]
        else:
            daily_ret = 0.0
        self.ret_buffer.append(daily_ret)
        volatility_10 = np.std(self.ret_buffer) if len(self.ret_buffer) == 10 else 0.0

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
        feature_vals.append(rsi6[-1])
        feature_vals.append(rsi14[-1])
        feature_vals.append(rsi24[-1])
        feature_vals.append(1 if rsi6[-1] < 30 else 0)
        rsi12 = talib.RSI(close, timeperiod=12)
        feature_vals.append(1 if rsi12[-1] < 30 else 0)
        feature_vals.append(1 if rsi24[-1] < 30 else 0)
        low_n = talib.MIN(low, timeperiod=9)
        high_n = talib.MAX(high, timeperiod=9)
        rsv = (close - low_n) / (high_n - low_n + 1e-9) * 100
        k = talib.EMA(rsv, timeperiod=3)
        d = talib.EMA(k, timeperiod=3)
        feature_vals.extend([k[-1], d[-1], 3 * k[-1] - 2 * d[-1]])
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

        # 更新 GPU 环形缓冲区
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
        pred_return = self.model.predict(seq)

        pos = ctx.long_pos()

        # 使用固定阈值进行买卖
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



def main():
    best_model_path = f'best_model_fixed-{datetime.now().timestamp()}.pt'
    params = {
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.222,
        'lr': 0.0094,
        'epochs': 50,
        'batch_size': 128,
    }
    seq_len = 60
    atr_stop = 1.1
    atr_profit = 2.157
    buy_thresh = 0.0020
    sell_thresh = -0.0013

    print("使用固定阈值策略进行回测...")
    config = StrategyConfig(initial_cash=1_000_000, fee_amount=0.0001)
    strategy = Strategy(ZszqDataSource(), '2020-01-01', '2026-03-20', config)

    ml_strat = MLStrategy('sh000001', '2020-01-01', '2026-03-20',
                          params=params, seq_len=seq_len,
                          atr_stop=atr_stop, atr_profit=atr_profit,
                          buy_thresh=buy_thresh, sell_thresh=sell_thresh)
    strategy.add_execution(ml_strat.exec_fn, ['sh000001'])

    result = strategy.backtest(timeframe='1d', adjust='')
    print("\n===== 固定阈值策略回测指标 =====")
    print(result.metrics)

    # ----- 保存条件判断 -----
    # 条件示例：总收益率 > 15% 且 最大回撤 < 20% 且 胜率 > 60%
    if (result.metrics.total_return_pct > 15 and
        result.metrics.max_drawdown_pct < 20 and
        result.metrics.win_rate > 60):
        torch.save(ml_strat.model.model.state_dict(), best_model_path)
        print(f"✅ 优秀模型已保存至 {best_model_path}")
    else:
        print("⚠️ 本次回测未达到保存标准，模型权重未保存")

    # 保存交易记录（可选）
    if result.trades is not None and not result.trades.empty:
        result.trades.to_csv('ml_trades_fixed_threshold.csv', index=False)
    if result.portfolio is not None and not result.portfolio.empty:
        result.portfolio.to_csv('ml_portfolio_fixed_threshold.csv')

# ============== 主程序：固定阈值直接回测 ==============
if __name__ == '__main__':
    for  i in range(10):
        print("第",i,"次回测")
        main()