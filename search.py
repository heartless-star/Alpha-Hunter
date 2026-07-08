###############################################################
#
# ██████   █████   ██████  ██    ██   █████   ██████   ██   ██
#   ███  ██   ██ ██       ██    ██  ██   ██  ██   ██  ██   ██
#  ███   ███████ ██       ████████  ███████  ██████    █████
# ███    ██   ██ ██       ██    ██  ██   ██  ██   ██    ███
# ██████  ██   ██  ██████  ██    ██  ██   ██  ██    ██   ███
#
##################################  coding by Zachary  ########

# -*- coding: utf-8 -*-

# @file:search.py
# @function:定义自然选择挖掘算法
# @time:2026/7/1
# @author:Zachary

import random
import pandas as pd
import re
from numpy import ndarray
import json
import numpy as np
from datetime import datetime
import os


class Alpha_Hunter:

    def __init__(self, gen: int, population_size: int, max: int, min: int,
                 IC_limit: float, ICIR_limit: float, top_k, IC_ICIR_weight: list,
                 mutate_rate: float, mutate_point_rate: float):
        '''
        :param gen: 淘汰代数
        :param population_size: 种群数量
        :param max: 公式最大长度 >= 2
        :param min: 公式最低长度 >=1
        :param top_k： 每代前百分之k停止变异，保留结构
        :param IC_limit : 保留因子ic最低值
        :param ICIR_limit :保留因子icir最低值
        :param IC_ICIR_weight :ic 和 icir 的权重比值
        :param mutate_rate: 整体变异率（发生变异的概率，调大有利于整体多向探索）
        :param mutate_point_rate: 节点变异率（变异个体内部节点发生替换的概率，调大有利于个体变换）
        '''
        self.gen = gen
        self.population = population_size
        self.long = max
        self.long_mini = min
        self.ic = IC_limit
        self.icir = ICIR_limit
        self.topk = top_k
        self.ic_icir_weight = IC_ICIR_weight
        self.mutate_rate = mutate_rate
        self.mutate_point_rate = mutate_point_rate

    def save_factor_pool(self, factor_dict, filepath=None, save_with_timestamp=True):
        """
        将因子池保存为 JSON 文件（自动处理 numpy 类型）

        参数:
            factor_pool: dict, 你的 fit() 方法返回的字典
            filepath: str, 保存路径（可选）
            save_with_timestamp: bool, 是否在文件名后加时间戳（防止覆盖）

            ---------by Deepseek

        """
        # 1. 处理文件路径
        if filepath is None:
            filename = "factor_pool.json"
        else:
            filename = filepath

        # 如果开启时间戳，在文件名后插入时间
        if save_with_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{timestamp}{ext}"

        # 2. 定义转换函数（核心：解决 numpy 无法序列化的问题）
        def convert_to_serializable(obj):
            """递归处理 numpy 类型，转换为 Python 原生类型"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        # 3. 转换数据
        cleaned_pool = convert_to_serializable(factor_dict)

        # 4. 保存文件（ensure_ascii=False 保证中文能正常显示）
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(cleaned_pool, f, ensure_ascii=False, indent=4)

        print(f"因子池已成功保存至: {filename}")
        return filename

    def calc_ic_icir(self, factor_values, df_meta, target_col='target_col', date_col='date_col'):
        """
        计算因子的 IC 和 ICIR（按天分组，横截面计算）

        参数:
        ----------
        factor_values : np.ndarray
            一维数组，长度 = df_meta 的行数，每个元素是该行股票当天的因子值
        df_meta : pd.DataFrame
            必须包含:
                - date_col: 日期列（用于分组）
                - target_col: 未来收益率列
        target_col : str,
            目标列名
        date_col : str,
            日期列名

        返回:
        ----------
        dict:
            {
                'ic_mean': float,   # 平均 IC
                'ic_std': float,    # IC 标准差
                'icir': float,      # ICIR (IC均值 / IC标准差)
                'daily_ic': pd.Series,  # 每日 IC 序列（可用于绘图）
                'n_days': int       # 有效天数
            }
        """
        # 1. 构造临时 DataFrame（避免修改原数据）
        df_temp = df_meta[[date_col, target_col]].copy()
        df_temp['factor'] = factor_values

        # 2. 剔除 NaN（因子值或目标值缺失的样本）
        df_temp = df_temp.dropna(subset=['factor', target_col])

        if len(df_temp) < 10:
            return {'ic_mean': np.nan, 'ic_std': np.nan, 'icir': np.nan, 'daily_ic': pd.Series(), 'n_days': 0}

        # 3. 按日期分组，计算每日 IC（横截面相关系数）
        def _daily_corr(group):
            # 对当天所有股票，计算因子值和目标值的相关系数
            # 如果因子值或目标值无波动（方差为0），直接返回 NaN，避免 numpy 除零警告
            if group['factor'].nunique() <= 1 or group[target_col].nunique() <= 1:
                return np.nan
            return group['factor'].corr(group[target_col])

        daily_ic = df_temp.groupby(date_col).apply(_daily_corr).dropna()

        if len(daily_ic) < 5:
            return {'ic_mean': np.nan, 'ic_std': np.nan, 'icir': np.nan, 'daily_ic': daily_ic, 'n_days': len(daily_ic)}

        # 4. 计算 IC 均值和 ICIR
        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std()
        icir = ic_mean / ic_std if ic_std > 0 else np.nan

        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': icir,
            # 'daily_ic': daily_ic,
            'n_days': len(daily_ic)
        }

    def evaluate_tree(self, tree, X):
        """
        将公式树（嵌套元组）翻译成 numpy 数组运算，返回一维因子值向量。

        参数:
            tree: 嵌套元组，如 ('div', 'X2', 'X1') 或 ('log', ('add', 'X0', 'X1'))
            X: numpy 数组，形状为 (n_samples, n_features)，即 (股票数, 特征数)

        返回:
            numpy 数组，形状为 (n_samples,)
        """
        # ---------- 1. 叶子节点：取特征列 ----------
        if isinstance(tree, str):
            # 支持 X0, X1, X2, X3... 映射到 X 矩阵的列
            if tree.startswith('X'):
                idx = int(tree[1:])
                if idx >= X.shape[1]:
                    raise ValueError(f"特征 {tree} 不存在，X 只有 {X.shape[1]} 列")
                return X[:, idx]
            else:
                raise ValueError(f"未知叶子节点: {tree}")

        # ---------- 2. 一元运算符（只有一个子节点） ----------
        if len(tree) == 2:
            op = tree[0]
            child = tree[1]
            # 递归计算子节点的值
            child_val = self.evaluate_tree(child, X)

            if op == 'neg':
                return -child_val
            elif op == 'inv':
                # 倒数保护：分母接近0时返回0
                return np.divide(1.0, child_val, where=np.abs(child_val) > 1e-9, out=np.zeros_like(child_val))
            elif op == 'log':
                # 对数保护：取绝对值并加极小值防止 log(0)
                return np.log(np.abs(child_val) + 1e-9)
            elif op == 'sqrt':
                # 开方保护：取绝对值
                return np.sqrt(np.abs(child_val))
            elif op == 'square':
                return child_val * child_val
            else:
                raise ValueError(f"未知一元运算符: {op}")

        # ---------- 3. 二元运算符（有两个子节点） ----------
        if len(tree) == 3:
            op = tree[0]
            left = tree[1]
            right = tree[2]
            # 递归计算左右子节点的值
            left_val = self.evaluate_tree(left, X)
            right_val = self.evaluate_tree(right, X)

            if op == 'add':
                return left_val + right_val
            elif op == 'sub':
                return left_val - right_val
            elif op == 'mul':
                return left_val * right_val
            elif op == 'div':
                # 除法保护：分母接近0时返回0，避免出现 inf
                return np.divide(left_val, right_val, where=np.abs(right_val) > 1e-9, out=np.zeros_like(left_val))
            else:
                raise ValueError(f"未知二元运算符: {op}")

        raise ValueError(f"无效的树结构: {tree}")

    def is_valid_formula(self, formula):

        weight = [0, 8]  # 连续两个运算符相同允许通过的权重,左边是通过，右边是拒绝

        """
        使用正则表达式过滤掉明显无意义的公式结构
        返回 True 表示保留，False 表示丢弃
        """
        # 1. 检测 "sqrt(square(...))" 和 "square(sqrt(...))" (互相抵消)
        if re.search(r"sqrt\s*\(\s*square\s*\(", formula):
            return False
        if re.search(r"square\s*\(\s*sqrt\s*\(", formula):
            return False

        # 2. 检测 "neg(neg(...))" 双重否定
        if re.search(r"neg\s*\(\s*neg\s*\(", formula):
            return False

        # 3. 检测 "inv(inv(...))" 双重倒数
        if re.search(r"inv\s*\(\s*inv\s*\(", formula):
            return False

        # 4. 检测 "log(neg(...))" 或 "sqrt(neg(...))" (定义域错误)
        if re.search(r"log\s*\(\s*neg\s*\(", formula):
            return False
        if re.search(r"sqrt\s*\(\s*neg\s*\(", formula):
            return False

        # 5. 检测 "sub(X, X)" 或 "div(X, X)" (左右完全相同的表达式)
        #    使用简单的模式匹配：sub( 任意内容, 同样的内容 )
        #    注意：这个检测无法处理嵌套复杂的情况，但能拦截最典型的垃圾
        if re.search(r"sub\s*\(\s*([^,]+?)\s*,\s*\1\s*\)", formula):
            return False
        if re.search(r"div\s*\(\s*([^,]+?)\s*,\s*\1\s*\)", formula):
            return False

        # 6. 检测 "add(neg(X), X)" 或 "add(X, neg(X))" (相加抵消)
        #    以及 "sub(X, X)" 已被上面覆盖，但 "sub(neg(X), X)" 等也可能抵消？
        #    更通用：检测 add(neg(...), ...) 且第二个参数是相同的表达式
        #    这里做简化：检测 add(neg( 任意 ), 任意 ) 且两者结构相似，但精确匹配复杂
        #    先拦截最明显的：add(neg(X0), X0) 和 add(X0, neg(X0))
        if re.search(r"add\s*\(\s*neg\s*\(\s*(X\d+)\s*\)\s*,\s*\1\s*\)", formula):
            return False
        if re.search(r"add\s*\(\s*(X\d+)\s*,\s*neg\s*\(\s*\1\s*\)\s*\)", formula):
            return False

        # 7. 拦截 "sub(neg(X), X)" 或 "sub(X, neg(X))" (减去相反数等于加)
        if re.search(r"sub\s*\(\s*neg\s*\(\s*(X\d+)\s*\)\s*,\s*\1\s*\)", formula):
            return False
        if re.search(r"sub\s*\(\s*(X\d+)\s*,\s*neg\s*\(\s*\1\s*\)\s*\)", formula):
            return False

        # 8. 拦截 "mul(inv(X), X)" 或 "mul(X, inv(X))" (乘倒数得1)
        if re.search(r"mul\s*\(\s*inv\s*\(\s*(X\d+)\s*\)\s*,\s*\1\s*\)", formula):
            return False
        if re.search(r"mul\s*\(\s*(X\d+)\s*,\s*inv\s*\(\s*\1\s*\)\s*\)", formula):
            return False

        # 9. 拦截 "div(X, X)" 已经包含在上面 (第5条)
        # 10. 拦截过深的 log 嵌套 ( log(log(log(...))) )，避免过度压缩
        if re.search(r"log\s*\(\s*log\s*\(\s*log\s*\(", formula):
            return False

        if re.search(r"([a-zA-Z]+)\s*\(\s*\1\s*\(", formula):
            ops = [True, False]
            res = random.choices(ops, weights=weight, k=1)[0]
            return res
        if re.search(r"div\(mul\((X\d), (X\d)\), (\1|\2)\)", formula):
            return False
        if re.search(r"sub\(add\((X\d), (X\d)\), (\1|\2)\)", formula):
            return False
        if re.search(r"mul\(div\((X\d), (X\d)\), (\1|\2)\)", formula):
            return False
        # 通过所有检测
        return True

    def parse_formula(self, formula_str):

        # deepseek
        """
        将字符串公式解析为嵌套元组树
        例如: 'log(div(X0, X1))' -> ('log', ('div', 'X0', 'X1'))
        """

        # 递归下降解析（简单版）
        def parse_expr(s):
            s = s.strip()
            # 如果是最简单的叶子节点（如 X0, X1）
            if re.match(r'^[X]\d$', s):
                return s

            # 查找最外层的操作符
            # 匹配模式: 操作符(参数1, 参数2, ...) 或 操作符(参数1)
            match = re.match(r'^(\w+)\((.+)\)$', s)
            if not match:
                raise ValueError(f"无法解析: {s}")

            op = match.group(1)
            args_str = match.group(2)

            # 按逗号分割参数（需要处理嵌套括号）
            args = []
            current = []
            depth = 0
            for ch in args_str:
                if ch == ',' and depth == 0:
                    args.append(''.join(current).strip())
                    current = []
                else:
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                    current.append(ch)
            if current:
                args.append(''.join(current).strip())

            # 递归解析每个参数
            parsed_args = [parse_expr(arg) for arg in args]

            # 如果是单参数运算符（log, sqrt, neg, inv, square），返回 (op, arg)
            if len(parsed_args) == 1:
                return (op, parsed_args[0])
            # 如果是双参数运算符（add, sub, mul, div），返回 (op, left, right)
            elif len(parsed_args) == 2:
                return (op, parsed_args[0], parsed_args[1])
            else:
                raise ValueError(f"不支持的参数数量: {op} 有 {len(parsed_args)} 个参数")

        return parse_expr(formula_str)

    def creat_gen_in_batch(self, X_long: int, all_dict=None, ):

        '''
               批量生产一代（活得也算，把总数补充到population_size）
               如果没有输入默认新建一个字典
               :return: 种群字典
        '''

        use_single_func_weight = [2, 10]  # 一元和二元算子使用概率

        function_allow = {
            0: "add",
            1: "sub",
            2: "mul",
            3: "div",
            4: "neg",  # 取负
            5: "inv",  # 取倒
            6: "log",
            7: "square",
            8: "sqrt"
        }
        single_func = [4, 5, 6, 7, 8]  # 一元算子
        double_func = [0, 1, 2, 3]  # 二元算子

        X_len = X_long  # 获取 X的长度

        if all_dict == None:
            all_gen = {}
        else:
            all_gen = all_dict

        if len(all_gen) < self.population:  # 这个是种群数量不够的情况

            num_compensate = self.population - len(all_gen)
            num = ["X" + str(fuck) for fuck in range(X_len)]

            limit_lon = self.long_mini
            while num_compensate > 0:  # 单次构造的循环

                long = random.randint(limit_lon, self.long)  # 单次公式随机长度
                # print(len(all_gen))
                if long == 1:  # 长度为1时别无选择，从X中挑一个得了,此时得看看all里面有没有还没被生成的单字
                    all_gen_key = set(all_gen.keys())
                    remain = [item for item in num if item not in all_gen_key]

                    if len(remain) != 0:
                        all_gen[random.choice(num)] = set(random.choice(num))
                        num_compensate -= 1
                        continue
                    else:
                        limit_lon = 2
                elif long == 2:  # 2特殊，因为2只能拼一元运算符号
                    symbol_single = random.choice(single_func)  # 找一个一元运算符  int
                    num_choice = random.choice(num)  # str
                    result = function_allow[symbol_single] + f"({num_choice})"
                    all_gen_key = set(all_gen.keys())
                    if result in all_gen_key:
                        continue
                    else:
                        all_gen[result] = self.parse_formula(result)
                        num_compensate -= 1
                else:  # 剩余大于2的情况，可以拼一元也可以拼2元
                    all_gen_key = set(all_gen.keys())
                    numm = long
                    result = ""
                    while numm != 0 and numm > 0:
                        if numm > 3:  # 至少得有4个吧
                            ops = ["single", "double"]
                            cho = random.choices(ops, weights=use_single_func_weight, k=1)[0]
                            if cho == "single":
                                funnc = random.choice(single_func)
                            else:
                                funnc = random.choice(double_func)

                            if funnc in double_func:  # 如果是取到二元算子
                                values = random.sample(num, 2)
                                if result == "":
                                    result = f"{function_allow[funnc]}({values[0]},{values[1]})"

                                    numm -= 3
                                else:  # 此时result里已经有值了，但由于限长大于3故没事，最后还会余下至少1位,外面拼一元运算符即可
                                    fuck = f"{function_allow[funnc]}({values[0]},{values[1]})"
                                    result = f"({result},{fuck})"
                                    syd = random.choice(double_func)  # 送一个二元符顺手的事
                                    result = f"{function_allow[syd]}{result}"

                                    numm -= 4
                            else:  # 取到一元算子的情况，位置多的很，直接拼
                                values = random.choice(num)
                                if result == "":
                                    result = f"{function_allow[funnc]}({values})"

                                    numm -= 2
                                else:  # 内容不为空时,要么拼一个二元加值在结尾，要么全部套一个一元符号，不管了
                                    fuck = f"{function_allow[funnc]}({values})"
                                    syd = random.choice(double_func)  # 送一个二元符顺手的事
                                    result = f"{function_allow[syd]}({result},{fuck})"

                                    numm -= 3
                        elif numm == 2:  # 到这个分支里result肯定不为空，可以整体拼一个三元也可以套一个一元
                            ops = ["single", "double"]
                            cho = random.choices(ops, weights=use_single_func_weight, k=1)[0]
                            if cho == "single":
                                funnc = random.choice(single_func)
                            else:
                                funnc = random.choice(double_func)
                            if funnc in single_func:
                                result = f"{function_allow[funnc]}({result})"

                                numm -= 1
                            else:
                                values = random.choice(num)
                                result = f"{function_allow[funnc]}({result},{values})"

                                numm -= 2
                        elif numm == 1:  # 只能套一元符号了
                            funnc = random.choice(single_func)
                            result = f"{function_allow[funnc]}({result})"

                            numm -= 1
                        elif numm == 3:
                            funnc = random.randint(0, 8)
                            if funnc in double_func:  # 如果是取到二元算子
                                values = random.sample(num, 2)
                                if result == "":
                                    result = f"{function_allow[funnc]}({values[0]},{values[1]})"

                                    numm -= 3
                                else:
                                    result = f"{function_allow[funnc]}({result},{random.choice(values)})"

                                    numm -= 2
                            else:  # 如果是取到一元算子
                                if result == "":
                                    values = random.choice(num)
                                    result = f"{function_allow[funnc]}({values})"

                                    numm -= 2
                                else:
                                    result = f"{function_allow[funnc]}({result})"

                                    numm -= 1

                    if result in all_gen_key:
                        continue
                    elif not self.is_valid_formula(result):
                        continue
                    else:
                        # all_gen[result] = self.parse_formula(result)
                        all_gen[result] = {}
                        num_compensate -= 1
        # print(all_gen)
        return all_gen

    def eva_gen(self, all_d, X, df_meta):  # 生成的数据进行评估
        all_dict = all_d
        for i in all_dict:
            # print(i)
            if all_dict[i] == {}:
                factor = self.parse_formula(i)
                # 再计算
                factor_value = self.evaluate_tree(factor, X)
                # 再评估
                ic_result = self.calc_ic_icir(factor_value, df_meta, )
                all_dict[i] = ic_result
        return all_dict

    def mutate(self, all_dict, x_long):

        def calculate_score(content, ic_w, icrc_w):  # 定义一个简单计算器
            # content是all_dict的value
            ic = content['ic_mean']
            icir = content['icir']
            score = round(float(ic * 20 * ic_w + icir * icrc_w), 3)  # ic乘以系数20缩放到同等量级
            return score

        def mut_c(self, formula, x_long):
            '''
            :param formula:接受一个公式，用于变异 ,变异不评估，能过is_valid_formula检测就行
            :return: new formula
            '''
            single_f = ['neg', 'inv', 'log', 'square', 'sqrt']
            double_f = ['add', 'sub', 'mul', 'div']

            li = re.findall(r'\w+|[^\w\s]', formula)
            # 现在融入节点变异率
            mr = self.mutate_point_rate * 10
            anw = 100 - mr
            weight = [mr, anw]
            ops = [True, False]
            for k in range(len(li)):
                i = li[k]
                if i != "(" and i != ")" and i != ',':  # 匹配节点
                    if random.choices(ops, weights=weight, k=1):
                        # 现在是变异的情况
                        if i in single_f:
                            fuckk = single_f.copy()
                            fuckk.remove(i)
                            mut_res = random.sample(fuckk, k=1)[0]
                            li[k] = mut_res
                        elif i in double_f:
                            fuckk = double_f.copy()
                            fuckk.remove(i)
                            mut_res = random.sample(fuckk, k=1)[0]
                            li[k] = mut_res
                        else:  # 剩下的是X变异
                            fuckk = ['X' + str(k) for k in range(x_long)]
                            fuckk.remove(i)
                            li[k] = random.sample(fuckk, k=1)[0]
            #  都搞完之后可以拼接了
            resu = ""
            for i in li:
                resu += i
            if self.is_valid_formula(resu):
                return resu
            else:
                return False

        # 接受一个all_dict, 先计算平均ic水平，划分区间，对不同区间使用不同的变异率，
        # 表现差的因子多变异，保留用户指定的top_k不动
        top_k = self.topk
        keep_alive = int(len(all_dict) * top_k)
        weight = self.ic_icir_weight  # 把用户设置的权重给搞出来
        ic_w = weight[0] / (weight[0] + weight[1])
        icir_w = weight[1] / (weight[0] + weight[1])

        sequence_dict = {}
        for i in all_dict:
            content = all_dict[i]
            score = calculate_score(content, ic_w, icir_w)
            sequence_dict[i] = score
        se_d = sorted(sequence_dict.items(), key=lambda x: x[1], reverse=True)  # 给总分排序，输出二元元组
        keep_alive = se_d[0:keep_alive + 1]
        # 到这里keep_alive里的就是top因子，保存不动，接下来开始变异
        for i in se_d:
            if i not in keep_alive:  # 过滤top因子
                # 这里遍历时要注意结合用户的整体变异率参数
                mr = self.mutate_rate * 100
                anw = 100 - mr
                weight = [mr, anw]
                ops = [True, False]
                if random.choices(ops, weights=weight, k=1)[0]:  # 抽到变异
                    while True:
                        ree = mut_c(self, i[0], x_long)
                        if ree is not False:
                            del all_dict[i[0]]
                            all_dict[ree] = {}
                            break
                        else:

                            continue
        # print(all_dict)
        return all_dict

    def fit(self, X: ndarray, Y: ndarray, date):
        '''
        处理数据接受，抽取，计算，测试的过程函数
        :return: dict
        '''

        def guolv(self, all_dict):
            print(f"数量：{len(all_dict)}，过滤中，IC,ICIR为nan值将被筛掉")

            for cnmb in list(all_dict.keys()):
                if np.isnan(all_dict[cnmb]['ic_mean']) or np.isnan(all_dict[cnmb]['icir']):
                    del all_dict[cnmb]

            print(f"数量：{len(all_dict)}，过滤中，IC< {self.ic} 或 ICIR < {self.icir} 将被筛掉")
            all_key = list(all_dict.keys())
            for i in all_key:
                if all_dict[i]['ic_mean'] < self.ic or all_dict[i]['icir'] < self.icir:
                    del all_dict[i]

        def calculatt(all_dict):
            total = 0
            valid_count = 0
            icir_total = 0
            icir_count = 0
            for i in all_dict:
                try:
                    ic = all_dict[i].get('ic_mean', np.nan)
                    icir_val = all_dict[i].get('icir', np.nan)
                except AttributeError:
                    continue
                if not np.isnan(ic):  # 跳过无效的NaN值
                    total += ic
                    valid_count += 1
                if not np.isnan(icir_val):
                    icir_total += icir_val
                    icir_count += 1

            ic_average = total / valid_count if valid_count > 0 else np.nan
            icir_average = icir_total / icir_count if icir_count > 0 else np.nan

            print(f'IC平均数:{ic_average}')
            print(f"ICIR平均数:{icir_average}")
            if valid_count == 0:
                print("⚠️  警告: 所有因子IC值均为NaN，请检查输入数据中因子值或目标收益率是否正常")

        #   print(f"有效因子数:{valid_count}, IC总和:{total:.3f}, ICIR总和:{icir_total:.3f}")

        self.x = X
        self.y = Y
        self.date = date
        assert len(X) == len(Y) and len(X) == len(date)
        # 拼接df_meta DF表
        df_meta = pd.DataFrame({
            'date_col': date,
            'target_col': Y
        })

        all_dict = self.creat_gen_in_batch(X_long=X.shape[1])  # 生成种群
        all_dict = self.eva_gen(all_dict, X, df_meta)

        # print(all_dict)
        for i in range(self.gen):
            print(f"第{i + 1}代")
            # 先变异，后筛选
            self.mutate(all_dict, x_long=X.shape[1])
            all_dict = self.eva_gen(all_dict, X, df_meta)
            guolv(self, all_dict)

            print(f"数量：{len(all_dict)},补充中...")
            calculatt(all_dict)
            all_dict = self.creat_gen_in_batch(X_long=X.shape[1], all_dict=all_dict)
            all_dict = self.eva_gen(all_dict, X, df_meta)
            # print(all_dict)

        guolv(self, all_dict)
        print(f"数量：{len(all_dict)}")

        all_dict = self.eva_gen(all_dict, X, df_meta)
        calculatt(all_dict)
        print(all_dict)
        return all_dict

    def valid(self, all_dict, X, Y, date):

        def calculatt(all_dict):
            total = 0
            valid_count = 0
            icir_total = 0
            icir_count = 0
            for i in all_dict:
                try:
                    ic = all_dict[i].get('ic_mean', np.nan)
                    icir_val = all_dict[i].get('icir', np.nan)
                except AttributeError:
                    continue
                if not np.isnan(ic):  # 跳过无效的NaN值
                    total += ic
                    valid_count += 1
                if not np.isnan(icir_val):
                    icir_total += icir_val
                    icir_count += 1

            ic_average = total / valid_count if valid_count > 0 else np.nan
            icir_average = icir_total / icir_count if icir_count > 0 else np.nan

            print(f'IC平均数:{ic_average}')
            print(f"ICIR平均数:{icir_average}")
            if valid_count == 0:
                print("⚠️  警告: 所有因子IC值均为NaN，请检查输入数据中因子值或目标收益率是否正常")

        fuc = {k: dict(v) for k, v in all_dict.items()}  # 深拷贝防止后续修改影响
        print("挖掘集数据表现")
        calculatt(fuc)
        print("====================")
        # 和fit一样，但是只评估all_dict在验证集上的表现
        assert len(X) == len(Y) and len(X) == len(date), f"X({len(X)}), Y({len(Y)}), date({len(date)}) 长度不一致"
        if len(X) == 0:
            print("❌ 错误: 验证集数据为空，请检查验证集的日期范围或抽样逻辑")
            return
        if len(X) < 10:
            print(f"⚠️  警告: 验证集仅 {len(X)} 行，calc_ic_icir 要求至少10行")
        # 拼接df_meta DF表
        df_meta = pd.DataFrame({
            'date_col': date,
            'target_col': Y
        })
        # 然后计算因子值传入计算
        for i in list(all_dict.keys()):
            tree = self.parse_formula(i)
            factor_values = self.evaluate_tree(tree, X)
            ret = self.calc_ic_icir(factor_values, df_meta)
            all_dict[i] = ret
        print("验证集表现：")
        calculatt(all_dict)
