import numpy as np
from random import randint
np.set_printoptions(suppress=True)
import sys
sys.setrecursionlimit(100000)
"""
    A   prod   m   
   sell
    n

last edited by Fang Xiaokun 2018202046
2020/5/5

"""

isfind = 0  # 判断有无找到回路的全局变量

def init(m, n):     # 初始化A
    A = np.zeros((m, n))
    A = np.full((m, n), np.nan)
    return A

def index_to_pos(index, m, n):  # 将 index 转化为 行列坐标
    return index//n, index%n

def random_choose(x, y, m, n, A):   # 随机选取一个合适的位置赋0
    a = -1
    b = -1

    rand_num = randint(0, 1)
    if(rand_num == 0):
        a = x
        b = randint(0, n-1)
    if(rand_num == 1):
        a = randint(0, m-1)
        b = y

    if((a == x and b == y) or A[a][b] >= 0 or a > b):
        return random_choose(x, y, m, n, A)

    return a, b

def min_pos(temp_price, temp_prod, temp_sell, m, n):    # 找到当前最小费用点
    index = np.argmin(temp_price)
    x, y = index_to_pos(index, m, n)

    if(temp_price[x][y] == 10000):
        for i in range(m):
            for j in range(n):
                if(temp_prod[i] > 0 and temp_sell[j] > 0):
                    return i, j
    return x, y

def total_cost(A, price):   # 返回总开销
    A[np.isnan(A)] = 0
    price[price < 0] = 0
    return np.sum(A * price)

def check_sigma(price, u, v):   # 通过 u v 生成 sigma
    judge = np.zeros(price.shape)
    for i in range(judge.shape[0]):
        judge[i, :] = price[i, :] - u[i] - v
    return judge

def generate_equations(A, price):  # 从A生成方程组
    m, n = A.shape
    equations = np.zeros((m+n-1, 3), dtype=int)

    count = 0
    for i in range(m):
        for j in range(n):
            if not np.isnan(A[i, j]):
                equations[count, 0] = i
                equations[count, 1] = j
                equations[count, 2] = price[i, j]
                count += 1
            if count >= (m+n-1):
                return equations

def dual_variable(equations, price):    # 解方程组 u v
    m, n = price.shape
    u = np.zeros(m) * np.nan    # u 和 v
    v = np.zeros(n) * np.nan    
    u[0] = 0    # 为u首元素赋0
    time = 1
    while(np.any(np.isnan(u)) or np.any(np.isnan(v))):  # 一个非常愚蠢但有效的求解方式
        for equation in equations:
            if np.isnan(u[equation[0]]) and not np.isnan(v[equation[1]]):
                u[equation[0]] = equation[2] - v[equation[1]]
            if not np.isnan(u[equation[0]]) and np.isnan(v[equation[1]]):
                v[equation[1]] = equation[2] - u[equation[0]]
            if np.isnan(u[equation[0]]) and np.isnan(v[equation[1]]) and time > max(u.shape[0], v.shape[0]):
                u[equation[0]] = equation[2]/2
                v[equation[1]] = equation[2]/2
        time += 1
    return u, v

def res(A, row, col, R):    # 通过 R 返回回路 转折点 奇偶转折点
    R = np.array(R)
    m1, n1 = R.shape
    m, n = A.shape

    offset = np.zeros((m1-1, n1))
    for i in range(m1-1):
        offset[i, :] = R[i+1, :] - R[i, :]
    point = []
    point += [[row, col, A[row, col]]]
    for i in range(1, m1-1):
        if np.any((offset[i, :] - offset[i-1, :]) != 0):
            turning_point = R[i, :].tolist() + [A[R[i, 0], R[i, 1]]]
            point += [turning_point]
    return point, point[1::2], point[::2]

def mark(A, x, y):  # 无用函数
    A[x, y] = 0
    B = A.copy()
    m, n = A.shape
    marks = np.zeros((m+2, n+2, 2))
    for p in range(1, m+1):
        T1 = ~np.isnan(A[p-1, :])
        for i in range(T1.shape[0]):
            if(T1[i]==True):
                marks[p, i+1, 0] = 1

    for p in range(1, n+1):
        T2 = ~np.isnan(A[:, p-1])
        for i in range(T2.shape[0]):
            if(T2[i]==True):
                marks[i+1, p, 1] = 1

    row = x
    col = y
    i = x
    j = y
    return marks, i, j, row, col, B

def seek_path(A, move, R, prev, times, i, j, row, col): # 递归找回路

    if(i < 0 or i >= A.shape[0] or j < 0 or j >= A.shape[1]):
        R.pop()
        return R

    if (i == row and j == col):
        times += 1
        if times == 2:
            global isfind
            isfind = 1
            return R
    
    if (np.isnan(A[i][j])):
        R.append((i+prev[0], j+prev[1]))
        R = seek_path(A, move, R, prev, times, i+prev[0], j+prev[1], row, col)
        if isfind:
            return R
        R.pop()
        return R

    elif(~np.isnan(A[i][j])):
        for k in range(4):
            next = move[k]
            if (next[0]*prev[0]+next[1]*prev[1] != -1):
                R.append((i+next[0], j+next[1]))
                R = seek_path(A, move, R, next, times, i+next[0], j+next[1], row, col)
                if isfind:
                    return R
        R.pop()
        return R

def adjust_distribution(A, point1, point2): # 调整运输方案
    point1 = np.array(point1)
    point2 = np.array(point2)

    index = np.argmin(point1[:, 2])
    p_min = np.min(point1[:, 2])

    point1[:, 2] -= p_min
    point2[:, 2] += p_min

    point2[np.isnan(point2)] = p_min

    for point in point1:
        A[int(point[0]), int(point[1])] = point[2]
    for point in point2:
        A[int(point[0]), int(point[1])] = point[2]
    A[int(point1[index, 0]), int(point1[index, 1])] = np.nan

    return A

def find_close_path(A, i, j):   # 找一条闭回路R
    move = np.array([[1, 0],[-1, 0], [0, 1], [0, -1]])
    m, n = A.shape
    x = i
    y = j
    R = []

    marks, i, j, row, col, B = mark(A, x, y)
    prev = np.array([0, 0])
    R = []
    R.append((i, j))
    R = seek_path(B, move, R, prev, 0, i, j, row, col)

    return res(A, row, col, R)

def min_element_method(A, price, prod, sell, m, n):     # 最小元素法给出初始解
    temp_price = price.copy()
    temp_prod = prod.copy()
    temp_sell = sell.copy()
    
    count = 0
    degradation_point = []

    while(~np.all(temp_prod==0) or ~np.all(temp_sell==0)):
        x, y = min_pos(temp_price, temp_prod, temp_sell, m, n)

        if(temp_prod[x] <=0 or temp_sell[y] <= 0):
            temp_price[x][y] = 10000
            continue

        move = min(temp_prod[x], temp_sell[y])
        temp_prod[x] -= move
        temp_sell[y] -= move
        A[x][y] = move
        temp_price[x][y] = 10000
        count += 1

        if(temp_prod[x] == 0 and temp_sell[y] == 0):
            degradation_point += [[x, y]]

    degradation_point = np.array(degradation_point)
    for i in range(m+n-1 - count):
        index = degradation_point.shape[0] - 1 - i
        a, b = random_choose(degradation_point[index, 0], degradation_point[index, 1], m, n, A)
        A[a, b] = 0

    return A

if __name__ == "__main__":

    M = 10000
    price =np.array([[0, 1, 3, 2, 1, 4, 3, 3, 11, 3, 10],
                     [1, 0, M, 3, 5, M, 2, 1, 9, 2, 8],
                     [3, M, 0, 1, M, 2, 3, 7, 4, 10, 5],
                     [2, 3, 1, 0, 1, 3, 2, 2, 8, 4, 6],
                     [1, 5, M, 1, 0, 1, 1, 4, 5, 2, 7],
                     [4, M, 2, 3, 1, 0, 2, 1, 8, 2, 4],
                     [3, 2, 3, 2, 1, 2, 0, 1, M, 2, 6],
                     [3, 1, 7, 2, 4, 1, 1, 0, 1, 4, 2],
                     [11, 9, 4, 8, 5, 8, M, 1, 0, 2, 1],
                     [3, 2, 10, 4, 2, 2, 2, 4, 2, 0, 3],
                     [10, 8, 5, 6, 7, 4, 6, 2, 1, 3, 0]])
    prod = np.array([27, 24, 29, 20, 20, 20, 20, 20, 20, 20, 20])   # 发送量
    sell = np.array([20, 20, 20, 20, 20, 20, 20, 23, 26, 25, 26])   # 接收量

    # price =np.array([[4, 1, 4, 6, 6],
    #                  [4, 1, 4, 6, 6],    # 退化为 指派问题
    #                  [1, 2, 5, 0, 0],
    #                  [1, 2, 5, 0, 0],
    #                  [3, 7, 5, 1, 1]], dtype=float)
    # prod = np.array([1, 1, 1, 1, 1])   # 发送量
    # sell = np.array([1, 1, 1, 1, 1])   # 接收量

    # M = 10000
    # price =np.array([[-4, 5, 3, 2, M],  # 运费系数  M表示不相连  负值表示本地转运
    #                  [5, -1, 2, M, 4],
    #                  [3, 2, -2, 5, 5],
    #                  [2, M, 5, -3, 6],
    #                  [M, 4, 5, 6, -5]])
    # prod = np.array([40, 80, 50, 50, 50])   # 发送量
    # sell = np.array([30, 40, 50, 80, 70])   # 接收量

    # price =np.array([[4, 12, 4, 11],
    #                  [2, 10, 2, 9],
    #                  [8, 5, 11, 6]], dtype=float)
    # prod = np.array([16, 10, 22])   # 发送量
    # sell = np.array([8, 14, 12, 14])   # 接收量

    # price =np.array([[4, 1, 4, 6],
    #                  [1, 2, 6, 1],
    #                  [3, 7, 5, 1]], dtype=float)
    # prod = np.array([8, 10, 4])   # 发送量
    # sell = np.array([8, 5, 6, 3])   # 接收量

    # M = 10000
    # price =np.array([[-1, M, 3, 8, M],  # 运费系数  M表示不相连  负值表示本地转运
    #                  [M, -1, 2, M, 9],
    #                  [3, 3, -1, 2, 5],
    #                  [8, M, 3, -1, M],
    #                  [M, 9, 5, M, -1]])
    # prod = np.array([80, 70, 50, 50, 50])   # 发送量
    # sell = np.array([50, 50, 50, 70, 80])   # 接收量
    
    m = len(prod)
    n = len(sell)
    A = init(m, n)
    A = min_element_method(A, price, prod, sell, m, n)  # 最小元素法给出初始表

    equations = generate_equations(A, price)
    u, v = dual_variable(equations, price)
    judge = check_sigma(price, u, v)

    row, col = index_to_pos(np.argmin(judge), m, n)

    times = 1
    while(np.min(judge) < 0):

        i = row
        j = col

        result, point1, point2 = find_close_path(A, i, j)
        isfind = 0
        A = adjust_distribution(A, point1, point2)

        equations = generate_equations(A, price)
        u, v = dual_variable(equations, price)
        judge = check_sigma(price, u, v)
        row, col = index_to_pos(np.argmin(judge), m, n)

        times = times + 1
        if times > 100:
            break

    print(A)
    print(total_cost(A, price))