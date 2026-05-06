import os

import pandas as pd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, '../data/stock_info_a_code_name.csv')
path = os.path.abspath(path)  # 最终绝对路径

def get_stock_info_a_code_name_and_save(update: bool = False) -> pd.DataFrame:
    if update:
        import akshare as ak
        stock_info_a_code_name_df = ak.stock_info_a_code_name()
        stock_info_a_code_name_df.to_csv(path, index=False)
        return stock_info_a_code_name_df
    return pd.read_csv(path)


def get_stock_info_a_code_name_by_file(
        st: bool = False,  # False=剔除ST
        exclude_bj: bool = True,  # True=剔除北交所
        exclude_kc: bool = True,  # True=剔除科创板
        exclude_sz: bool = False,  # True=剔除深主板
        exclude_sh: bool = False,  # True=剔除沪主板
        exclude_cy: bool = True,  # True=剔除创业板
) -> pd.DataFrame:
    # 1. 读取本地文件，无网络请求
    df = get_stock_info_a_code_name_and_save(update=False)

    # 代码格式化
    df["code"] = df["code"].astype(str).str.zfill(6)

    # --------------------------
    # 板块过滤（按你的参数）
    # --------------------------
    # 沪市主板 60xxxx
    if not exclude_sh:
        cond_sh = df["code"].str.startswith("60")
    else:
        cond_sh = pd.Series([False] * len(df))

    # 科创板 688xxxx
    if not exclude_kc:
        cond_kc = df["code"].str.startswith("688")
    else:
        cond_kc = pd.Series([False] * len(df))

    # 深市主板 000xxx
    if not exclude_sz:
        cond_sz = df["code"].str.startswith("000")
    else:
        cond_sz = pd.Series([False] * len(df))

    # 创业板 300xxx
    if not exclude_cy:
        cond_cy = df["code"].str.startswith("300")
    else:
        cond_cy = pd.Series([False] * len(df))

    # 北交所 83/87/43
    if not exclude_bj:
        cond_bj = df["code"].str.startswith(("83", "87", "43"))
    else:
        cond_bj = pd.Series([False] * len(df))

    # 合并保留条件
    df = df[cond_sh | cond_kc | cond_sz | cond_cy | cond_bj].copy()

    # --------------------------
    # 过滤 ST、*ST、退市
    # --------------------------
    if not st:
        df = df[~df["name"].str.contains(r"ST|\*ST|退市", regex=True)]

    # ==========================
    # 【新增】生成带前缀的市场代码列：sz/sh/bj + 6位代码
    # ==========================
    def add_market_prefix(code):
        if code.startswith(("60", "688")):
            return f"sh{code}"
        elif code.startswith(("000", "300")):
            return f"sz{code}"
        elif code.startswith(("83", "87", "43")):
            return f"bj{code}"
        else:
            return ""

    df["market"] = df["code"].apply(add_market_prefix)
    # 过滤掉无法识别市场的代码
    df = df[df["market"] != ""].copy()

    # 重置索引
    df = df.reset_index(drop=True)

    return df

if __name__ == '__main__':
    print(get_stock_info_a_code_name_by_file()['market'].values)