import numpy as np

class SimpleRegression:
    def __init__(self):
        self.intercept = None # 切片
        self.coefficient = None # 傾き
        
    def fit(self, x, y):
        n = len(x) # 要素数
        self.coefficient = (np.dot(x, y) - x.sum() * y.sum() / n) / ((x**2).sum() - x.sum()**2 / n)
        self.intercept = (y.sum() - self.coefficient * x.sum()) / n
        
    def predict(self, x):
        return self.coefficient * x + self.intercept