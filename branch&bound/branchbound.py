from scipy.optimize import linprog
from math import ceil, floor
from queue import Queue

import numpy as np
import math
import sys

# 判断一个float数是否足够接近一个整数
def is_integer(num:float):
    if(min(ceil(num)-num, num-floor(num)) < 1e-5):
        return True
    else:
        return False

# 如果足够接近 将该float数转化为最近的一个int
def convert_to_near_integer(num:float):
    if(ceil(num)-num < num-floor(num)):
        return ceil(num)
    else:
        return floor(num)

class ILP():
    def __init__(self, c, A_ub, b_ub, A_eq, b_eq, bounds):
        # 全局参数
        self.LOWER_BOUND = -sys.maxsize
        self.UPPER_BOUND = sys.maxsize
        self.opt_val = None
        self.opt_x = None
        self.Q = Queue()

        # 这些参数在每轮计算中都不会改变
        self.c = -c
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.bounds = bounds

        # 首先计算一下初始问题
        r = linprog(-c, A_ub, b_ub, A_eq, b_eq, bounds)

        # 若最初问题线性不可解
        if not r.success:
            raise ValueError('Not a feasible or bounded problem!')

        # 将解和约束参数放入队列
        self.Q.put((r, A_ub, b_ub))

    def solve(self):
        while not self.Q.empty():
            # 取出当前问题
            res, A_ub, b_ub = self.Q.get(block=False)

            # 当前最优值小于总下界，则排除此区域
            if -res.fun < self.LOWER_BOUND:
                continue

            # 若结果 x 中全为整数，则尝试更新全局下界、全局最优值和最优解
            if all(list(map(lambda f: is_integer(f), res.x))):
                if self.LOWER_BOUND < -res.fun:
                    self.LOWER_BOUND = -res.fun

                if self.opt_val is None or self.opt_val < -res.fun:
                    self.opt_val = -res.fun
                    self.opt_x = res.x

                continue

            # 进行分枝
            else:
                # 寻找 x 中第一个不是整数的，取其下标 idx
                idx = 0
                for i, x in enumerate(res.x):
                    if not x.is_integer():
                        break
                    idx += 1

                # 构建新的约束条件（分割
                new_con1 = np.zeros(A_ub.shape[1])
                new_con1[idx] = -1
                new_con2 = np.zeros(A_ub.shape[1])
                new_con2[idx] = 1
                new_A_ub1 = np.insert(A_ub, A_ub.shape[0], new_con1, axis=0)
                new_A_ub2 = np.insert(A_ub, A_ub.shape[0], new_con2, axis=0)
                new_b_ub1 = np.insert(
                    b_ub, b_ub.shape[0], -ceil(res.x[idx]), axis=0)
                new_b_ub2 = np.insert(
                    b_ub, b_ub.shape[0], floor(res.x[idx]), axis=0)

                # 将新约束条件加入队列，先加最优值大的那一支
                r1 = linprog(self.c, new_A_ub1, new_b_ub1, self.A_eq,
                             self.b_eq, self.bounds)
                r2 = linprog(self.c, new_A_ub2, new_b_ub2, self.A_eq,
                             self.b_eq, self.bounds)

                if not r1.success and r2.success:
                    self.Q.put((r2, new_A_ub2, new_b_ub2))
                elif not r2.success and r1.success:
                    self.Q.put((r1, new_A_ub1, new_b_ub1))
                elif r1.success and r2.success:
                    if -r1.fun > -r2.fun:
                        self.Q.put((r1, new_A_ub1, new_b_ub1))
                        self.Q.put((r2, new_A_ub2, new_b_ub2))
                    else:
                        self.Q.put((r2, new_A_ub2, new_b_ub2))
                        self.Q.put((r1, new_A_ub1, new_b_ub1))

# 从文件读入线性规划问题
def inPut():    
    model_str = []
    with open("input.txt",'r') as f:
        lines = f.readlines()
    for line in lines:
        numbers = line.strip('\n').split(' ')
        numbers = list(map(int, numbers))
        model_str.append(numbers)
    return model_str

# 将线性规划问题转化为linprog可以解决的标准形式 
# c, A, b, Aeq, beq, bounds
def str_to_arg(model_str):

    flag = 1
    if model_str[0][0] == 0:
        flag = -1

    c = flag * np.array(model_str[1])

    m, n = model_str[2][0], model_str[2][1]
    A = np.array(model_str[3:])
    A[A[:, n+1] == 2] *= -1

    Aeq = A[A[:, n+1] == 0].copy()
    if(Aeq.shape[0] == 0):
        Aeq = None
        beq = None
    else:
        beq = Aeq[:, n]
        Aeq = Aeq[:, :n]

    A = A[A[:, n+1] != 0]
    b = A[:, n]
    A = A[:, :n]

    bounds = n * [(0, None)]

    return flag, c, A, b, Aeq, beq, bounds

# 主函数入口
def solve():
    flag, c, A, b, Aeq, beq, bounds = str_to_arg(inPut())
    solver = ILP(c, A, b, Aeq, beq, bounds)
    solver.solve()

    print("z =", flag * convert_to_near_integer(solver.opt_val), "\nx* =", solver.opt_x)

if __name__ == '__main__':
    solve()