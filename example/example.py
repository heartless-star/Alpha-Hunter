#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例脚本：从预抽样的 CSV 文件加载数据，调用 Alpha_Hunter 引擎进行因子挖掘。

运行本脚本即可直接调用因子挖掘引擎，无需连接数据库。

用法:
    python example.py                          # 默认使用 Y_5d（5日收益）

"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# ---------- 路径 ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # example/ 目录
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                 # 项目根目录
sys.path.insert(0, PROJECT_ROOT)                           # 保证能找到 search.py

DATA_DIR = os.path.join(SCRIPT_DIR, "sampled_data")       # 预抽样 CSV 目录
from search import Alpha_Hunter as AlphaHunterEngine


def load_sampled_data(target: str = "5d"):
    """从 sampled_data 读取 X, y, date"""
    # 读取特征
    X = pd.read_csv(os.path.join(DATA_DIR, "X.csv")).values

    # 读取目标
    target_file = f"Y_{target}.csv"
    y_file = os.path.join(DATA_DIR, target_file)
    if not os.path.exists(y_file):
        print(f"❌ 找不到目标文件: {y_file}")
        print(f"   可选目标: 1d, 5d, 10d")
        sys.exit(1)
    y = pd.read_csv(y_file).values.ravel()

    # 读取日期
    date = pd.read_csv(os.path.join(DATA_DIR, "dates.csv"))["trade_date"].values

    # 验证长度一致
    assert len(X) == len(y) == len(date), \
        f"数据长度不一致: X={len(X)}, y={len(y)}, date={len(date)}"

    print(f"📊 数据加载完成: {len(X)} 个样本, 特征={X.shape[1]}列")
    return X, y, date


def main():
    parser = argparse.ArgumentParser(description="Alpha_Hunter 因子挖掘示例")
    parser.add_argument("--target", type=str, default="5d", choices=["1d", "5d", "10d"],
                        help="预测周期 (1d/5d/10d)")
    parser.add_argument("--gen", type=int, default=4, help="进化代数 (默认 10)")
    parser.add_argument("--pop", type=int, default=100, help="种群数量 (默认 320)")
    args = parser.parse_args()

    # ---------- 1. 加载数据 ----------
    print("📂 加载抽样数据...")
    X, y, date = load_sampled_data(args.target)

    print(f"🎯 预测周期: {args.target}")
    print(f"🧬 进化代数: {args.gen} | 种群数量: {args.pop}")
    print("-" * 50)

    # ---------- 2. 初始化引擎 ----------
    hunter = AlphaHunterEngine(
        gen=args.gen,
        population_size=args.pop,
        max=5,           # 公式最大长度
        min=3,           # 公式最小长度
        IC_limit=0.04,   # IC 最低门槛
        ICIR_limit=0.7,  # ICIR 最低门槛
        top_k=0.15,      # 每代前 15% 保留不变异
        IC_ICIR_weight=[6, 4],  # IC 和 ICIR 权重
        mutate_rate=0.6,
        mutate_point_rate=0.3
    )

    # ---------- 3. 启动挖掘 ----------
    print("\n🚀 启动因子挖掘...")
    hunter.fit(X, y, date)


if __name__ == "__main__":
    main()
