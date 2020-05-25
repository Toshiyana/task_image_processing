from linreg import LinearRegression

class PolynomialRegression:
    def __init__(self, degree):
        self.degree = degree
        self.weight = None
    
    def fit(self, x, y):
        x_pow = []
        xx = x.reshape(len(x), 1)
        for i in range(1, self.degree + 1):
            x_pow.append(xx**i)
        
        matrix = np.concatenate(x_pow, axis = 1)
        lr = LinearRegression() # 線形回帰を利用
        lr.fit(matrix, y)
        self.weight = lr.weight