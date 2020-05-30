import numpy as np

class LinearRegression:
    def __init__(self):
        self.weight = None # 重み
        self.intercept = None # 定数項
        self.coefficient = None # 偏回帰係数
        
    def fit(self, x, y):
        x_tilda = np.c_[np.ones(x.shape[0]), x]
        #print(x_tilda.T)
        A = np.dot(x_tilda.T, x_tilda)
        B = np.dot(x_tilda.T, y)
        #print(A, B)
        self.weight = np.dot(np.linalg.inv(A), B) # np.linalg.inv: 逆行列に変換
        self.intercept = self.weight[0]
        self.coefficient = self.weight[1:]
        
    def predict(self, x):
        if x.ndim == 1: # 1次元配列の場合
            x = x.reshape(1, -1) # 2次元配列に変換
            
        x_tilda = np.c_[np.ones(x.shape[0]), x]
        return np.dot(x_tilda, self.weight)