import os
import time
import modin.pandas as pd
from dask.distributed import Client


def main():
    os.environ["MODIN_ENGINE"] = "dask"
    # os.environ["MODIN_ENGINE"] = "ray"
    with Client(n_workers=4) as client:
        print(f"Dask dashboard 地址: {client.dashboard_link}")

        start = time.time()
        df = pd.DataFrame({
            'a': range(10_000_000),
            'b': range(10_000_000)
        })
        print("数据创建完成，行数：", len(df))

        df['c'] = df['a'] + df['b']
        print("计算完成")

        # 强制完成计算（聚合操作触发完整计算）
        total = df['c'].sum()  # 或者 df['c'].to_numpy()
        print(f"c 列总和为: {total}")  # 可选，证明计算完成

        print("\n数据框类型：", type(df))
        print("Modin 启动成功 ✅")
        print(f"总耗时：{time.time() - start:.2f}s")

if __name__ == '__main__':
    main()