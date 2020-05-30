import numpy as np
from linreg import LinearRegression

# 説明変数が一つの時のみ
class PolynomialRegression:
    def __init__(self, degree):
        self.degree = degree
        self.weight = None
    
    def fit(self, x, y):
        x_pow = []
        xx = x.reshape(len(x), 1)
        for i in range(1, self.degree + 1):
            x_pow.append(xx**i)
        
        mat = np.concatenate(x_pow, axis = 1) # ベクトルを横に繋いだ行列の生成

        # print(xx)
        # print(x_pow)
        # print(mat)

        lr = LinearRegression() # 線形回帰を利用
        lr.fit(mat, y)
        self.weight = lr.weight
        
    def predict(self, x):
        r = 0
        for i in range(self.degree + 1):
            r += x**i * self.weight[i]
            
        return r