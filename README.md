# Alpha Hunter —— 自研因子挖掘引擎

> "我们不制造因子，我们只是市场的狩猎者。"

Alpha Hunter 是一个基于遗传编程（Genetic Programming）的因子挖掘框架，专为量化投研设计。它通过进化算法自动搜索高预测力、高稳定性的价量因子，并提供灵活的筛选与变异控制，支持从原始数据到候选因子池的全流程自动化。

## 核心特性

- **纯符号回归**：无常数硬编码，避免过拟合
- **多目标筛选**：同时考量 IC（信息系数）和 ICIR（信息比率），支持加权排序
- **自适应变异**：可分别控制整体变异率和节点变异率，精细调节搜索方向
- **模块化设计**：核心参数可配置，方便嵌入各种量化投研流程

---

## 安装与依赖

### 环境要求

- Python 3.8+
- 推荐 Linux / macOS（Windows 也支持，多进程建议使用 spawn 模式）

### 依赖库

```bash
pip install numpy pandas scikit-learn
```

### 克隆项目

```bash
git clone https://github.com/your-username/AlphaHunter.git
cd AlphaHunter
```

---

## 数据格式

Alpha Hunter 期望输入的数据格式为**面板数据（Panel Data）**，即包含多个股票在多个交易日的观察值。你需要准备以下三个对齐的数组：

| 变量 | 形状 | 说明 |
|------|------|------|
| X | (n_samples, n_features) | 特征矩阵，每列代表一个基础因子（如 ret_1, amp, vol_ratio 等） |
| Y | (n_samples,) | 目标变量，通常为未来 N 日的收益率（如 ret_future_5d） |
| dates | (n_samples,) | 日期数组，用于横截面分组计算 IC |

> **注意**：X, Y, dates 的行顺序必须一一对应，且不能打乱。

### 示例（从 DataFrame 构造）

```python
import pandas as pd
import numpy as np

df = pd.read_csv('your_data.csv')
# 假设 df 包含: date, stock_code, feature1, feature2, ..., target
X = df[['feature1', 'feature2', 'feature3']].values
Y = df['target'].values
dates = df['date'].values
```

---

## 快速开始

### 基础用法

```python
from alpha_hunter import AlphaHunter

# 1. 实例化引擎
hunter = AlphaHunter(
    gen=10,
    population_size=320,
    min_len=3,
    max_len=5,
    IC_limit=0.03,
    ICIR_limit=0.7,
    top_k=0.1,
    IC_ICIR_weight=0.6,
    mutate_rate=0.5,
    mutate_point_rate=0.3
)

# 2. 准备数据（假设 X, Y, dates 已定义）
# X: (n_samples, n_features), Y: (n_samples,), dates: (n_samples,)

# 3. 运行挖掘（自动进行多代进化）
factor_pool = hunter.fit(X, Y, dates)

# 4. 查看结果
print(f"共挖掘到 {len(factor_pool)} 个有效因子")
for formula, metrics in factor_pool.items():
    print(f"{formula}: IC={metrics['ic_mean']:.4f}, ICIR={metrics['icir']:.2f}")
```

### 输出格式

`fit` 方法返回一个字典，键为公式字符串（如 `neg(mul(X2,X1))`），值为包含以下字段的字典：

| 字段 | 类型 | 说明 |
|------|------|------|
| ic_mean | float | 样本内平均 IC |
| ic_std | float | IC 标准差 |
| icir | float | ICIR（IC均值 / IC标准差） |
| n_days | int | 有效交易日数 |

---

## 核心参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| gen | int | 10 | 进化代数 |
| population_size | int | 320 | 每代种群个体数 |
| min_len | int | 2 | 公式最小节点数（至少为1） |
| max_len | int | 6 | 公式最大节点数（至少为2） |
| IC_limit | float | 0.02 | 保留因子的最低 IC 阈值 |
| ICIR_limit | float | 0.5 | 保留因子的最低 ICIR 阈值 |
| top_k | float | 0.1 | 每代保留的精英比例（前 k% 不参与变异） |
| IC_ICIR_weight | float | 0.7 | 排序时 IC 与 ICIR 的权重（0~1） |
| mutate_rate | float | 0.4 | 个体发生变异的概率（整体探索强度） |
| mutate_point_rate | float | 0.3 | 变异个体中每个节点被替换的概率 |

> **提示**：初期探索可提高 mutate_rate 和 mutate_point_rate，后期收敛时降低。

---

## 特征说明

引擎预设了 4 个基础特征，用户可自行扩展：

| 特征 | 符号 | 计算方式 | 含义 |
|------|------|----------|------|
| 昨日收益率 | X0 | close / close.shift(1) - 1 | 短期动量 |
| 振幅 | X1 | (high - low) / close | 日内波动 |
| 量比 | X2 | volume / MA(volume, 5) | 相对成交量 |
| 日内强度 | X3 | close / open - 1 | 日内多空力量 |

---

## 运算符池

| 类型 | 运算符 | 说明 |
|------|--------|------|
| 二元 | add, sub, mul, div | 四则运算 |
| 一元 | neg, inv, log, sqrt, square | 非线性变换 |

> **注意**：所有一元运算符均内置数值保护（如 log 取绝对值，div 防除零），确保评估过程不产生 NaN 或 inf。

---

## 内置过滤器（免疫系统）

引擎在生成阶段会自动拦截以下垃圾结构：

| 拦截类型 | 示例 | 原因 |
|----------|------|------|
| 定义域错误 | log(neg(X)), sqrt(neg(X)) | 数学未定义 |
| 代数抵消 | add(neg(X), X), sub(X, X) | 恒为常数 |
| 双重否定 | neg(neg(X)), inv(inv(X)) | 等价于原特征 |
| 逆运算抵消 | div(mul(X,Y), Y) | 等价于 X |
| 连续压缩 | log(log(log(X))) | 方差趋近于零 |

---

## 功能矩阵

| 功能 | 状态 | 说明 |
|------|------|------|
| 公式随机生成 | ✅ | 长度可控，无常数硬编码 |
| 公式解析 | ✅ | 字符串 → 嵌套元组树 |
| 公式计算 | ✅ | 递归树解析 + numpy 向量化运算 |
| IC / ICIR 评估 | ✅ | 横截面分组，日频聚合 |
| 基础过滤器 | ✅ | 拦截 log(neg)、add(neg(X),X) 等毒瘤 |
| 精英保留机制 | ✅ | top_k 比例个体不参与变异 |
| 完整进化循环 | ✅ | 选择 → 变异 → 评估 → 筛选 → 补充 |
| 自适应变异 | ✅ | 分别控制 mutate_rate 和 mutate_point_rate |
| 多目标加权排序 | ✅ | IC 与 ICIR 按权重组合评分 |
| 多进程并行 | 🔄 | 规划中，当前为单进程 |
| 验证集自动回测 | 🔄 | 规划中 |
| 因子聚类去重 | 🔄 | 规划中 |

---

## 成果展示

以下为使用 Alpha Hunter 在 A 股全市场（2010-2025）上运行 3 代所挖掘出的实际表现（筛选条件：IC ≥ 0.04，ICIR ≥ 0.7）：

### 挖掘集表现（2010-2019）

```
IC平均数: 0.0527
ICIR平均数: 0.8735
有效因子数: 23
```

### 验证集表现（2020-2025）

```
IC平均数: 0.0438
ICIR平均数: 0.6762
```

> 验证集 IC 衰减约 17%，属于正常范围，说明引擎挖掘出的因子具有真实的预测能力，而非过拟合产物。

### 挖掘到的部分有效因子（IC ≥ 0.04, ICIR ≥ 0.7）

| 公式 | IC | ICIR | 逻辑解读 |
|------|----|------|----------|
| mul(sub(X1, X2), X1) | 0.0636 | 0.86 | (振幅 - 量比) × 振幅，识别无效波动 |
| neg(sub(square(X2), X0)) | 0.0590 | 1.04 | 量比平方减去收益率后取反，捕捉极端放量反转 |
| mul(sub(X0, X2), X2) | 0.0585 | 1.04 | (收益 - 量比) × 量比，缩量上涨信号放大 |
| mul(neg(X2), X2) | 0.0596 | 1.04 | 量比平方取反 |
| neg(mul(X2, X1)) | 0.0610 | 0.81 | 量比与振幅乘积取反，反向惩罚活跃股 |

---

## Roadmap

- [ ] 多进程并行评估：利用多核 CPU 加速
- [ ] 验证集自动回测：挖掘集 / 验证集 IC 对比报告
- [ ] 因子聚类与去重：剔除高度相关的近亲因子
- [ ] 自定义特征池：用户可扩展基础特征
- [ ] 神经网络集成：用挖掘出的因子训练 NN

---

## 贡献

Alpha Hunter 是一个个人项目，欢迎任何形式的建议、Issue 和 PR。作者将会长期维护该项目

## 许可证

MIT License

## 联系

> "静候花开，Alpha 终将降临。"
