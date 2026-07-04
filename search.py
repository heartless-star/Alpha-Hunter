###############################################################
#
#██████   █████   ██████  ██    ██   █████   ██████   ██   ██
#   ███  ██   ██ ██       ██    ██  ██   ██  ██   ██  ██   ██
#  ███   ███████ ██       ████████  ███████  ██████    █████
# ███    ██   ██ ██       ██    ██  ██   ██  ██   ██    ███
#██████  ██   ██  ██████  ██    ██  ██   ██  ██    ██   ███
#
##################################  coding by Zachary  ########


# -*- coding: utf-8 -*-

# @file:search.py
# @function:定义自然选择挖掘算法
# @time:2026/7/1
# @author:Zachary

'''
自研算法之自然选择原理
'''

import numpy as np
import random
import pandas as pd
import re

from numpy import ndarray


class search_factor():

    def __init__(self,gen:int,population_size:int,function_set,long:int,long_mini :int ,top_k:float,n_job,ran_test:bool):

        '''

        :param gen: 淘汰代数
        :param population_size: 种群数量
        :param function_set: 允许算子符号  (元组) 自动模式”auto“
        :param long: 公式长度限制 >= 2
        :param long_mini :最低长度 >=1
        :param top_k： 筛选每代前百分之k
        :param n_job: 多进程数
        :param ran_test 稀疏测试开关
        '''
        self.gen = gen
        self.population = population_size
        self.function_set = function_set
        self.long = long
        self.long_mini = long_mini
        self.n_job = n_job #多进程数
        self.ran_test = ran_test

    def calc_ic_icir(self,factor_values, df_meta, target_col='target_col', date_col='date_col'):
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

    def evaluate_tree(self,tree, X):
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

    def is_valid_formula(self,formula):

        weight = [0,8]  #连续两个运算符相同允许通过的权重,左边是通过，右边是拒绝

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

        if re.search(r"([a-zA-Z]+)\s*\(\s*\1\s*\(",formula):
            ops = [True,False]
            res = random.choices(ops, weights=weight, k=1)[0]
            return res
        if re.search(r"div\(mul\((X\d), (X\d)\), (\1|\2)\)",formula):
            return False
        if re.search(r"sub\(add\((X\d), (X\d)\), (\1|\2)\)",formula):
            return False
        if re.search(r"mul\(div\((X\d), (X\d)\), (\1|\2)\)",formula):
            return False
        # 通过所有检测
        return True

    def parse_formula(self,formula_str):

        #deepseek
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

    def creat_gen_in_batch(self,X_long:int, all_dict = None,):

        '''
               批量生产一代（活得也算，把总数补充到population_size）
               如果没有输入默认新建一个字典
               :return: 种群字典
        '''

        use_single_func_weight = [2,10]  #一元和二元算子使用概率


        function_allow = {
            0:"add",
            1:"sub",
            2:"mul",
            3:"div",
            4:"neg",  #取负
            5:"inv",  #取倒
            6:"log",
            7:"square",
            8:"sqrt"
        }
        name_func = function_allow.values()

        single_func = [4,5,6,7,8]# 一元算子
        double_func = [0,1,2,3] # 二元算子

        allow_index = [] # switch symbol to number
        if self.function_set != "auto":
            for i in self.function_set:
                allow_index.append(list(name_func).index(i))
        else:
            pass   #预留一个自动模式

        X_len = X_long  # 获取 X的长度

        if all_dict == None:
            all_gen = {}
        else:
            all_gen = all_dict

        if len(all_gen) < self.population:    # 这个是种群数量不够的情况

            num_compensate= self.population - len(all_gen)
            num = ["X" + str(fuck) for fuck in range(X_len)]

            limit_lon = self.long_mini
            while num_compensate > 0:  # 单次构造的循环

                long = random.randint(limit_lon,self.long)  # 单次公式随机长度
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
                elif long == 2: # 2特殊，因为2只能拼一元运算符号
                    symbol_single = random.choice(single_func) # 找一个一元运算符  int
                    num_choice = random.choice(num) # str
                    result = function_allow[symbol_single] + f"({num_choice})"
                    all_gen_key = set(all_gen.keys())
                    if result in all_gen_key:
                        continue
                    else:
                        all_gen[result] = self.parse_formula(result)
                        num_compensate -= 1
                else: #剩余大于2的情况，可以拼一元也可以拼2元
                    all_gen_key = set(all_gen.keys())
                    numm = long
                    result = ""
                    while numm != 0 and numm > 0:
                        if numm >3 : #至少得有4个吧
                            ops = ["single", "double"]
                            cho = random.choices(ops,weights=use_single_func_weight,k=1)[0]
                            if cho == "single":
                                funnc = random.choice(single_func)
                            else:
                                funnc = random.choice(double_func)

                            if funnc in double_func: # 如果是取到二元算子
                                values = random.sample(num,2)
                                if result == "":
                                    result = f"{function_allow[funnc]}({values[0]},{values[1]})"

                                    numm -= 3
                                else: # 此时result里已经有值了，但由于限长大于3故没事，最后还会余下至少1位,外面拼一元运算符即可
                                    fuck = f"{function_allow[funnc]}({values[0]},{values[1]})"
                                    result = f"({result},{fuck})"
                                    syd = random.choice(double_func)  # 送一个二元符顺手的事
                                    result = f"{function_allow[syd]}{result}"

                                    numm -= 4
                            else: #取到一元算子的情况，位置多的很，直接拼
                                values = random.choice(num)
                                if result == "":
                                    result = f"{function_allow[funnc]}({values})"

                                    numm -= 2
                                else: #内容不为空时,要么拼一个二元加值在结尾，要么全部套一个一元符号，不管了
                                    fuck = f"{function_allow[funnc]}({values})"
                                    syd = random.choice(double_func) # 送一个二元符顺手的事
                                    result = f"{function_allow[syd]}({result},{fuck})"

                                    numm -= 3
                        elif numm == 2: #到这个分支里result肯定不为空，可以整体拼一个三元也可以套一个一元
                            ops = ["single", "double"]
                            cho = random.choices(ops, weights=use_single_func_weight, k=1)[0]
                            if cho == "single":
                                funnc = random.choice(single_func)
                            else:
                                funnc = random.choice(double_func)
                            if funnc in single_func:
                                result =  f"{function_allow[funnc]}({result})"

                                numm -= 1
                            else:
                                values = random.choice(num)
                                result = f"{function_allow[funnc]}({result},{values})"

                                numm -= 2
                        elif numm == 1: #只能套一元符号了
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
                            else: # 如果是取到一元算子
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

    def eva_gen(self,all_d,X,df_meta): #生成的数据进行评估
        all_dict = all_d
        formula = list(all_dict.keys())
        for i in formula:
            # print(i)
            if all_dict[i] == {}:
                factor = self.parse_formula(i)
                # 再计算
                factor_value = self.evaluate_tree(factor, X)
                # 再评估
                ic_result = self.calc_ic_icir(factor_value, df_meta, )
                all_dict[i] = ic_result
        return all_dict

    def mutate(self,all_dict):
        pass

    def fit(self,X:ndarray,Y:ndarray,date):
        '''
        处理数据接受，抽取，计算，测试的过程函数
        :return: dict
        '''
        self.x = X
        self.y = Y
        self.date = date
        assert len(X) == len(Y) and len(X) == len(date)
        #拼接df_meta DF表
        df_meta = pd.DataFrame({
            'date_col': date,
            'target_col': Y
        })

        all_dict = self.creat_gen_in_batch(X_long=X.shape[1])  #生成种群

        all_dict = self.eva_gen(all_dict,X,df_meta)
        # print(all_dict)
        for i in range(self.gen):
            print(f"数量：{len(all_dict)}，过滤中，IC<0.02将被筛掉")
            all_key = list(all_dict.keys())
            for i in all_key:
                if all_dict[i]['ic_mean'] < 0.02:
                    del all_dict[i]
            print(f"数量：{len(all_dict)},补充中。。。")
            all_dict = self.creat_gen_in_batch(X_long=X.shape[1],all_dict=all_dict)
            all_dict = self.eva_gen(all_dict, X, df_meta)
            # print(all_dict)

        print(f"数量：{len(all_dict)}，过滤中，IC<0.02将被筛掉")
        all_key = list(all_dict.keys())
        for i in all_key:
            if all_dict[i]['ic_mean'] < 0.02:
                del all_dict[i]
        print(f"数量：{len(all_dict)}")
        print(all_dict)

    def multi_worker(self): # 多进程
        pass


# test = search_factor(2, 200, ('add','sub'),6,3,0.2,1,True)
# print(test.creat_gen_in_batch(4))