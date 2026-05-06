import pandas as pd
import numpy as np

# ==================== 读取交易记录 ====================
trades = pd.read_csv('trades_log.csv', parse_dates=['entry_date', 'exit_date'])
print(f"交易记录总笔数：{len(trades)}")

# 盈亏列名（通常为 'pnl'）
pnl_col = 'pnl'
if pnl_col not in trades.columns:
    # 尝试其他常见列名
    for col in ['pnl', 'return', 'profit']:
        if col in trades.columns:
            pnl_col = col
            break
    else:
        raise KeyError(f"找不到盈亏列，现有列：{trades.columns.tolist()}")

INITIAL_CASH = 1_000_000

# ==================== 指标计算函数 ====================
def max_drawdown_from_equity_array(equity_arr):
    """根据权益序列计算最大回撤（百分比）"""
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / peak
    return drawdown.min()   # 负值，绝对值越大回撤越大

def annualized_return(total_return, years):
    """安全年化，避免极端值"""
    if years <= 0 or total_return <= -1:
        return -1.0          # 亏损超过100%，视为-100%年化
    if total_return > 100:   # 收益超过100倍，视为数据异常
        return np.nan
    return (1 + total_return) ** (1 / years) - 1

# ==================== 按股票计算 ====================
results = []
for symbol, group in trades.groupby('symbol'):
    if group.empty:
        continue

    # ---------- 基本盈亏 ----------
    total_pnl = group[pnl_col].sum()
    total_return = total_pnl / INITIAL_CASH
    trade_count = len(group)

    wins = group[group[pnl_col] > 0]
    loses = group[group[pnl_col] <= 0]
    win_rate = len(wins) / trade_count if trade_count > 0 else np.nan
    avg_win  = wins[pnl_col].mean() if len(wins) > 0 else 0
    avg_loss = loses[pnl_col].mean() if len(loses) > 0 else 0

    # ---------- 最大回撤（用每次交易后的权益点）----------
    group_sorted = group.sort_values('exit_date')
    cumulative_pnl = group_sorted[pnl_col].cumsum()
    equity_points = INITIAL_CASH + cumulative_pnl
    mdd = max_drawdown_from_equity_array(equity_points.values)   # 负值

    # ---------- 年化收益 ----------
    first_entry = group['entry_date'].min()
    last_exit   = group['exit_date'].max()
    years = (last_exit - first_entry).days / 365.25
    ann_ret = annualized_return(total_return, years)

    results.append({
        'symbol': symbol,
        'total_return': total_return,
        'ann_return': ann_ret,
        'max_drawdown': mdd,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'trade_count': trade_count,
        'years': years
    })

stats = pd.DataFrame(results)
print(f"有效股票数：{len(stats)}")

# ==================== 过滤短期异常股票（交易时长 < 1个月）====================
mask_long = stats['years'] >= 1/12   # 至少交易一个月
stats_filtered = stats[mask_long].copy()
print(f"交易时长≥1个月的股票数：{len(stats_filtered)}")

# ==================== 全 A 汇总 ====================
def show_summary(title, df):
    print(f"\n======== {title} ========")
    print(f"平均累计收益(%)：{df['total_return'].mean():.2%}")
    print(f"中位数累计收益(%)：{df['total_return'].median():.2%}")
    print(f"平均年化收益(%)：{df['ann_return'].mean():.2%}")
    print(f"年化收益中位数(%)：{df['ann_return'].median():.2%}")
    print(f"平均最大回撤(%)：{df['max_drawdown'].mean():.2%}")
    print(f"最大回撤中位数(%)：{df['max_drawdown'].median():.2%}")
    print(f"平均胜率(%)：{df['win_rate'].mean():.2%}")
    print(f"平均盈利金额：{df['avg_win'].mean():.0f}")
    print(f"平均亏损金额：{df['avg_loss'].mean():.0f}")
    print(f"平均交易次数：{df['trade_count'].mean():.1f}")

    print("\n收益分位数(%)：")
    print(df['total_return'].describe(percentiles=[.1,.25,.5,.75,.9]).to_string(float_format='%.4f'))

    print("\n最大回撤分位数(%)：")
    print(df['max_drawdown'].describe(percentiles=[.1,.25,.5,.75,.9]).to_string(float_format='%.4f'))

show_summary("全部股票", stats)
show_summary("交易时长≥1个月的股票（过滤短期异常）", stats_filtered)