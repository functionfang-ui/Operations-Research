from __future__ import division
import re
import numpy as np


"""
editor:     方晓坤(Fang Xiaokun)
studentID:  2018202046
date:       2020/4/19
"""

def inPut():
    model_str = []
    with open("input.txt",'r') as f:
        lines = f.readlines()
    for line in lines:
        numbers = line.strip('\n').split(' ')
        numbers = list(map(int, numbers))
        model_str.append(numbers)
    return model_str

str = inPut()

def getShape(str):
    return str[2][0], str[2][1]             #获得约束矩阵的长和宽

def regularize(str, m, n, c_2):

    count = 0
    temp = 0
    for i in range(3, 3+m):                 #正则化添加的变量计数
        if str[i][-1] != 0:
            count+=1

    c_2_list = c_2.tolist()
    for i in range(count):                  #第二阶段的目标函数系数向量
        c_2_list.append(0)
    c_2 = np.array(c_2_list, dtype='float64')
    
    for i in range(3, 3+m):
        for _ in range(count):
            str[i].insert(n, 0)
        if(str[i][-1] == 2):
            str[i][n+temp] = -1
            temp+=1
        if(str[i][-1] == 1):
            str[i][n+temp] = 1
            temp+=1
        del str[i][-1]

    for i in range(3, 3+m):                 #每行再添加一个变量凑齐单位矩阵
        for _ in range(m):
            str[i].insert(n+count, 0)
    for i in range(m):
        str[i+3][n+count+i] = 1

    str[2][1] = len(str[3])-1               #更新 约束矩阵 长 宽

    return c_2

def getIndex(b, A, sigma):                  #获取 出基变量 入基变量 下标
    x = -1
    y = -1
    sigma_temp = sigma.tolist()
    y = sigma_temp.index(max(sigma_temp))

    y_row = A[:, y]
    temp_min = 10000000
    for i in range(A.shape[0]):
        if(y_row[i] <= 0):
            continue
        if(b[i] / y_row[i] < temp_min):
            x = i
            temp_min = b[i] / y_row[i]

    return x, y

def rowTransform(x, y, c_b, x_b, b, c, A, sigma):  #行变化 出入基

    main = A[x, y].copy()           # 主元素

    b[x] = b[x] / main              # 单位化
    A[x, :] = A[x, :] / main

    for i in range(A.shape[0]):
        if i == x:
            continue
        b[i] = b[i] - A[i, y] * b[x]     # 行变化
        A[i] = A[i] - A[i, y] * A[x]
        

    sigma[:] = sigma - sigma[y] * A[x]  # 调整检验数

    x_b[x] = y          # 入基
    c_b[x] = c[y]

def outPut(c, c_b, x_b, b, A, sigma, stage, step):
    num = c.shape[0]                                # 根据变量数目 打印分割线

    print("\nStage", stage, " Step", step)

    # print("--------------------------------------------------------------------------------------------")
    for _ in range(num * 9 + 20):
        print("-", end="")
    print("")

    print("           C           ", end="")        # 打印 C
    for element in c:
        print("\t", element, end="")

    # print("\n--------------------------------------------------------------------------------------------")
    print("")
    for _ in range(num * 9 + 20):
        print("-", end="")
    print("")

    print("   C_b     x_b     b   ", end="")        # 打印 C_b x_b b x_1 …… x_n 表头
    for i in range(c.shape[0]):
        print("\t", "x_", end="")
        print(i+1, end="")

    # print("\n--------------------------------------------------------------------------------------------")
    print("")
    for _ in range(num * 9 + 20):
        print("-", end="")
    print("")

    for i in range(A.shape[0]):                     # 打印 A
        print("   %0.0f"%c_b[i], end="")
        print("     x_", end="")
        print("%0.0f"%(x_b[i]+1), end="")
        print("     ", b[i], end="")
        for j in range(A.shape[1]):
            print("\t", "%0.2f"%A[i][j], end="")
        print("\n", end="")

    # print("--------------------------------------------------------------------------------------------")
    for _ in range(num * 9 + 20):
        print("-", end="")
    print("")

    print("         sigma         ", end="")        # 打印 sigma
    for element in sigma:
        print("\t", "%0.2f"%element, end="")
    print("")

def is_no_solution(c, b, x_b, sigma):             # 第一阶段 检验数均小于0 最终解z*小于0
    if(np.all(sigma <= 0)):
        for i in range(x_b.shape[0]):
            if c[x_b[i].astype(int)] == -1:
                return True
    return False

def stage_one(c, c_b, x_b, b, A, sigma):        #第一阶段
    check = x_b.copy()

    stage = 1
    step = 1

    while(True):
        outPut(c, c_b, x_b, b, A, sigma, stage, step)

        if(is_no_solution(c, b, x_b, sigma)):
            print("\nThe problem doesn’t has a feasible solution.")
            exit()

        if(not (set(check) & set(x_b))):
            break

        x, y = getIndex(b, A, sigma)
        rowTransform(x, y, c_b, x_b, b, c, A, sigma)
        step += 1

def is_unbounded(A, sigma):                      # 第二阶段 检验数大于0 对应A一列全部不大于0
    for i in range(A.shape[1]):
        if(sigma[i] > 0 and np.all(A[:, i] <= 0)):
            return True
    return False

def is_infinite(A, sigma, x_b):                 # 第二阶段 非基变量检验数等于0 对应列有大于0
    for i in range(A.shape[1]):
        if(sigma[i] == 0 and (i not in x_b) and np.any(A[:, i] > 0)):
            return True
    return False

def is_end(sigma):                               # 第二阶段 检验数均小于等于0
    if(np.all(sigma <= 0)):
        return True
    return False

def get_ans(c_b, x_b, b, length):                   # 返回 最优解
    x = np.zeros(length)
    for i in range(length):
        for j in range(b.shape[0]):
            if(x_b[j] == i):
                x[i] = b[j]

    z = np.sum(c_b * b)
    if str[0][0] == 0:                              # z* = -z
        z = -z
    return x, z

def stage_two(c, c_b, x_b, b, A, sigma, length):        #第二阶段
    stage = 2
    step = 1

    while(True):
        outPut(c, c_b, x_b, b, A, sigma, stage, step)

        if(is_unbounded(A, sigma)):
            print("\nThe optimal solution of the problem is unbounded.\nx is unbounded\nz is unbounded")
            break
        if(is_infinite(A, sigma, x_b)):
            x, z = get_ans(c_b, x_b, b, length)
            print("\nThe number of optimal solution is unlimited.\nOne of the optimal solution is", z, "\nx = ", x)
            break
        if(is_end(sigma)):
            x, z = get_ans(c_b, x_b, b, length)
            print("\nThe optimal solution of the problem is", z, "\nx =", x)
            break
    
        x, y = getIndex(b, A, sigma)
        rowTransform(x, y, c_b, x_b, b, c, A, sigma)
        step += 1

def solve():
    m, n = getShape(str)

    if str[0][0] == 0:                      # min -> max    目标函数最大
        for i in range(len(str[1])):
            str[1][i] = -1*str[1][i]

    c_2 = np.zeros(len(str[1]))             # 目标函数
    for i in range(len(str[1])):
        c_2[i] = str[1][i]

    for i in range(3, m+3):                 # (b < 0) -> (b > 0)    资源变量非负
        if(str[i][n] < 0):
            for j in range(n+1):
                str[i][j] = -1 * str[i][j]
            if(str[i][n+1] == 1):
                str[i][n+1] = 2
            if(str[i][n+1] == 2):
                str[i][n+1] = 1

    c_2 = regularize(str, m, n, c_2)        # 第二阶段的目标函数系数向量 c_2    约束条件等式
    m, n = getShape(str)

    c_1 = []                                # 第一阶段的目标函数系数向量 c_1
    for i in range(n):
        if i < m:
            c_1.insert(0, -1)
        else:
            c_1.insert(0, 0)
    c_1 = np.array(c_1, dtype='float64')

    b = np.zeros(m)                         # 资源变量 b
    for i in range(m):
        b[i] = str[i+3][n]
    
    c_b = np.zeros(m)                       #变量对应的目标函数系数 c_b
    c_b[:] = -1

    x_b = np.zeros(m)                       #选中的变量 x_b
    for i in range(m):
        x_b[i] = n-m+i
    
    A = np.zeros((m,n))                     #约束矩阵 A
    for i in range(m):
        for j in range(n):
            A[i][j] = str[i+3][j]
    
    sigma = np.zeros(n)
    sigma = c_1 - (c_b.T @ A).T             #检验数 sigma

    stage_one(c_1, c_b, x_b, b, A, sigma)   # 第一阶段求解

    A = A[:,:(n-m)].copy()                  # 修改A c 和 c_b sigma

    for i in range(m):
        c_b[i] = c_2[x_b[i].astype(int)]
    sigma = c_2 - (c_b.T @ A).T
    
    length = len(str[1])                           # length 用于判断sigma对应的变量是否在目标函数内
    stage_two(c_2, c_b, x_b, b, A, sigma, length)  #  第二阶段求解

solve()
