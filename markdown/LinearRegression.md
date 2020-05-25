# 線形回帰
線形回帰とは、回帰式が線形なものである。ここでは、まず、単回帰について説明し、そのあと、単回帰を拡張した重回帰について説明する。

## 単回帰
単回帰とは、データ分布に対して一次元の当てはまりの良い直線をもとめることである。


```python
import numpy as np
import matplotlib.pyplot as plt

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
    
if __name__=="__main__":
    x = np.array([1, 2, 3, 6, 7, 9]) # 学習データの説明変数x
    y = np.array([1, 3, 3, 5, 4, 6]) # 学習データの目的変数y
    model = SimpleRegression()
    model.fit(x, y)
    
    # グラフの表示
    plt.scatter(x, y) # 学習データの分布を表示
    x_max = x.max()
    plt.plot([0, x_max], [model.intercept, model.coefficient * x_max + model.intercept]) # 回帰直線の表示
    plt.show()
```

## 重回帰
上述した単回帰は、特徴量ベクトルXが一次元であった。対して、重回帰は、特徴量ベクトルをd次元に拡張したものである。
