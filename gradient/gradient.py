import sys
import math
import time
import numpy as np
from sympy import symbols, diff, Matrix, hessian
from sympy.matrices.dense import matrix2numpy

class GradientDescent():
    """多种优化器来优化函数 f(x1, x_2).

    默认参数:
    eta=0.1, mu=0.9, beta1=0.9, beta2=0.99, rho=0.9, epsilon=1e-10, stop_condition=0.0015

    每次参数改变为(d1, d2).梯度为(dx1, dx2)
    
    t+1次迭代
    标准GD
        d1_{t+1} = - eta * dx1
        d2_{t+1} = - eta * dx2
    带Momentum
        d1_{t+1} = eta * (mu * d1_t - dx1_{t+1})
        d2_{t+1} = eta * (mu * d2_t - dx2_{t+1})    
    带Nesterov Accent
        d1_{t+1} = eta * (mu * d1_t - dx1^{nag}_{t+1})
        d2_{t+1} = eta * (mu * d2_t - dx2^{nag}_{t+1})
        其中(dx1^{nag}, dx2^{nag})为(x1 + eta * mu * d1_t, x2 + eta * mu * d2_t)处的梯度
    RMSProp
        w1_{t+1} = beta2 * w1_t + (1 - beta2) * dx1_t^2
        w2_{t+1} = beta2 * w2_t + (1 - beta2) * dx2_t^2
        d1_{t+1} = - eta * dx1_t / (sqrt(w1_{t+1}) + epsilon)
        d2_{t+1} = - eta * dx2_t / (sqrt(w2_{t+1}) + epsilon)
    Adam 每次参数改变为(d1, d2)
        v1_{t+1} = beta1 * v1_t + (1 - beta1) * dx1_t
        v2_{t+1} = beta1 * v2_t + (1 - beta1) * dx2_t
        w1_{t+1} = beta2 * w1_t + (1 - beta2) * dx1_t^2
        w2_{t+1} = beta2 * w2_t + (1 - beta2) * dx2_t^2
        v1_corrected = v1_{t+1} / (1 - beta1^{t+1})
        v2_corrected = v2_{t+1} / (1 - beta1^{t+1})
        w1_corrected = w1_{t+1} / (1 - beta2^{t+1})
        w2_corrected = w2_{t+1} / (1 - beta2^{t+1})
        d1_{t+1} = - eta * v1_corrected / (sqrt(w1_corrected) + epsilon)
        d2_{t+1} = - eta * v2_corrected / (sqrt(w2_corrected) + epsilon)
    AdaGrad
        w1_{t+1} = w1_t + dx1_t^2
        w2_{t+1} = w2_t + dx2_t^2
        d1_{t+1} = - eta * dx1_t / sqrt(w1_{t+1} + epsilon)
        d2_{t+1} = - eta * dx2_t / sqrt(w2_{t+1} + epsilon)
    Adadelta
        update1_{t+1} = rho * update1_t + (1 - rho) * d1_t^2
        update2_{t+1} = rho * update2_t + (1 - rho) * d2_t^2
        w1_{t+1} = rho * w1_t + (1 - rho) * dx1_t^2
        w2_{t+1} = rho * w2_t + (1 - rho) * dx2_t^2
        d1_{t+1} = - dx1 * rms(update1_{t+1}) / rms(w1_{t+1})
        d2_{t+1} = - dx2 * rms(update2_{t+1}) / rms(w2_{t+1})
        定义 rms(x) = sqrt(x + epsilon)
    """

    def __init__(self, eta=0.01, beta1=0.9, beta2=0.99, epsilon=1e-10, stop_condition=0.0001):
        # 算法的学习率  不同的算法会做针对性调整
        self.eta = eta

        # 迭代终止条件
        self.stop_condition = stop_condition

        # Adam超参数
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # 录入函数
        x, y = symbols('x y')
        self.x, self.y = x, y
        self.f = 0.5 * ( x**4 - 16*x**2 + 5*x + y**4 - 16*y**2 + 5*y)

        # 给定一个初始值
        self.x1_init, self.x2_init = -4, -4
        self.x1, self.x2 = self.x1_init, self.x2_init

    def gd(self, max_ite=1000):
        # 初始化
        setattr(self, "ite", 0)
        setattr(self, "x1", self.x1_init)
        setattr(self, "x2", self.x2_init)

        # 迭代
        # start = time.time()
        for _ in range(max_ite):
            self.d1 = -self.eta * self.dx1
            self.d2 = -self.eta * self.dx2
            self.ite += 1

            self.x1 += self.d1
            self.x2 += self.d2
            print("第{}次迭代后,坐标为({}, {})".format(self.ite, self.x1, self.x2))
            if self.offset < self.stop_condition:
                break
        # end = time.time()
        # print((end-start)*1000)
        print("\n迭代结束点为({}, {})".format(self.x1, self.x2))
        print("最小值为{}".format(self.f.evalf(subs={self.x:self.x1, self.y:self.x2})))

    def newton(self, max_ite=1000):
        # 初始化
        setattr(self, "ite", 0)
        setattr(self, "x1", self.x1_init)
        setattr(self, "x2", self.x2_init)

        fx = diff(self.f, self.x)
        fy = diff(self.f, self.y)
        grad_f1 = Matrix([[fx], [fy]])
        grad_H2 = hessian(self.f, (self.x, self.y))
        x_tmp = self.x1_init
        y_tmp = self.x2_init

        #迭代
        # start = time.time()
        for _ in range(max_ite):
            grad_f1 = np.array([[float(fx.evalf(subs={self.x:x_tmp, self.y:y_tmp}))],
                                [float(fy.evalf(subs={self.x:x_tmp, self.y:y_tmp}))]])
            tmp = matrix2numpy(grad_H2.evalf(subs={self.x:x_tmp, self.y:y_tmp}), dtype=float)
            ans_tmp = np.array([[x_tmp], [y_tmp]]) - np.dot(np.linalg.inv(tmp), grad_f1)    
            acc_tmp = ( (ans_tmp[0,0]-x_tmp)**2 + (ans_tmp[1,0]-y_tmp)**2 )**0.5
            self.ite += 1

            print("第{}次迭代后,坐标为({}, {})".format(self.ite, ans_tmp[0, 0], ans_tmp[1, 0]))
            
            x_tmp = ans_tmp[0,0]
            y_tmp = ans_tmp[1,0]
            f_tmp = self.f.evalf(subs={self.x:x_tmp, self.y:y_tmp})

            if acc_tmp <= self.stop_condition:
                self.x1 = ans_tmp[0, 0]
                self.x2 = ans_tmp[1, 0]
                break
        # end = time.time()
        # print((end-start)*1000)
        print("\n迭代结束点为({}, {})".format(self.x1, self.x2))
        print("最小值为{}".format(self.f.evalf(subs={self.x:self.x1, self.y:self.x2})))

    def adam(self, max_ite=1000):
        # 初始化
        setattr(self, "ite", 0)
        setattr(self, "x1", self.x1_init)
        setattr(self, "x2", self.x2_init)
        setattr(self, "w1_pre", 0)
        setattr(self, "w2_pre", 0)
        setattr(self, "v1_pre", 0)
        setattr(self, "v2_pre", 0)

        # 迭代
        start = time.time()
        for _ in range(max_ite):
            w1 = self.beta2 * self.w1_pre + (1 - self.beta2) * (self.dx1 ** 2)
            w2 = self.beta2 * self.w2_pre + (1 - self.beta2) * (self.dx2 ** 2)
            v1 = self.beta1 * self.v1_pre + (1 - self.beta1) * self.dx1
            v2 = self.beta1 * self.v2_pre + (1 - self.beta1) * self.dx2
            self.ite += 1
            self.v1_pre, self.v2_pre = v1, v2
            self.w1_pre, self.w2_pre = w1, w2

            v1_corr = v1 / (1 - math.pow(self.beta1, self.ite))
            v2_corr = v2 / (1 - math.pow(self.beta1, self.ite))
            w1_corr = w1 / (1 - math.pow(self.beta2, self.ite))
            w2_corr = w2 / (1 - math.pow(self.beta2, self.ite))

            self.d1 = -self.eta * v1_corr / (math.sqrt(w1_corr) + self.epsilon)
            self.d2 = -self.eta * v2_corr / (math.sqrt(w2_corr) + self.epsilon)

            self.x1 += self.d1
            self.x2 += self.d2

            print("第{}次迭代后,坐标为({}, {})".format(self.ite, self.x1, self.x2))
            if self.offset < self.stop_condition:
                break
        end = time.time()
        print((end-start)*1000)
        print("\n迭代结束点为({}, {})".format(self.x1, self.x2))
        print("最小值为{}".format(self.f.evalf(subs={self.x:self.x1, self.y:self.x2})))

    def adagrad(self, max_ite=1000):
        # 初始化
        setattr(self, "ite", 0)
        setattr(self, "x1", self.x1_init)
        setattr(self, "x2", self.x2_init)
        setattr(self, "w1_pre", 0)
        setattr(self, "w2_pre", 0)

        # 迭代
        # start = time.time()
        for _ in range(max_ite):
            w1 = self.w1_pre + self.dx1 ** 2
            w2 = self.w2_pre + self.dx2 ** 2
            self.ite += 1
            self.w1_pre, self.w2_pre = w1, w2
            self.d1 = -self.eta * self.dx1 / math.sqrt(w1 + self.epsilon)
            self.d2 = -self.eta * self.dx2 / math.sqrt(w2 + self.epsilon)

            self.x1 += self.d1
            self.x2 += self.d2

            print("第{}次迭代后,坐标为({}, {})".format(self.ite, self.x1, self.x2))
            if self.offset < self.stop_condition:
                break
        # end = time.time()
        # print((end-start)*1000)
        print("\n迭代结束点为({}, {})".format(self.x1, self.x2))
        print("最小值为{}".format(self.f.evalf(subs={self.x:self.x1, self.y:self.x2})))

    @property
    def dx1(self, x1=None):
        return diff(self.f, self.x).evalf(subs={self.x:self.x1, self.y:self.x2})

    @property
    def dx2(self):
        return diff(self.f, self.y).evalf(subs={self.x:self.x1, self.y:self.x2})

    @property
    def offset(self):
        return math.sqrt(self.d1**2 + self.d2**2)

def gd_test(eta=0.01):
    func = GradientDescent()
    print("起始点为({}, {})".format(func.x1_init, func.x2_init))
    func.eta = eta
    func.gd()

def newton_test(eta=0.001):
    func = GradientDescent()
    print("起始点为({}, {})".format(func.x1_init, func.x2_init))
    func.eta = eta
    func.newton()

def adam_test(eta=1):
    func = GradientDescent()
    print("起始点为({}, {})".format(func.x1_init, func.x2_init))
    func.eta = eta
    func.adam()

def ada_test(eta=10):
    func = GradientDescent()
    print("起始点为({}, {})".format(func.x1_init, func.x2_init))
    func.eta = eta
    func.adagrad()

if __name__ == '__main__':
    getattr(sys.modules[__name__], 'adam_test')()
