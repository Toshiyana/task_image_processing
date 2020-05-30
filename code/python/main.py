import numpy as np
import matplotlib.pyplot as plt
from linreg import LinearRegression
from polyreg import PolynomialRegression
from simplereg import SimpleRegression

if __name__=="__main__":
    '''
    # 単回帰
    x = np.array([1, 2, 3, 6, 7, 9]) # 学習データの説明変数x
    y = np.array([1, 3, 3, 5, 4, 6]) # 学習データの目的変数y
    model = SimpleRegression()
    model.fit(x, y)
    
    #print(model.intercept, model.coefficient)
    print(model.predict(x))

    # グラフの表示
    plt.scatter(x, y, color = "blue") # 学習データの分布を表示
    x_max = x.max()
    plt.plot([0, x_max], [model.intercept, model.coefficient * x_max + model.intercept], color = "red") # 回帰直線の表示
    plt.show()
    '''

    # 重回帰
    # 適当な学習データを生成
    # n = 50 # 要素数
    # scale = 100
    # np.random.seed(0) # 乱数生成器のシード値を固定
    #x = np.random.random((n, 2)) * scale # 説明変数x
    #w0, w1, w2 = 1, 2, 3
    #y = np.random.randn(n) + w0 + w1 * x[:, 0] + w2 * x[:, 1] # 目的変数y
    
    # x = np.array([[33.0, 22.0],
    #               [31.0, 26.0],
    #               [32.0, 28.0]])

    # y = np.array([382.0, 324.0, 350])

    # model = LinearRegression()
    # model.fit(x, y)

    # print(model.predict(x))
    
    # print("定数項:", model.intercept)
    # print("偏回帰係数:", model.coefficient)
    # print("重み:", model.weight)
    
    # # グラフの表示
    # x_grid, y_grid = np.meshgrid(np.linspace(0, scale, 20), np.linspace(0, scale, 20))
    # z = (model.weight[0] + model.weight[1] * x_grid.flatten() + model.weight[2] * y_grid.flatten()).reshape(x_grid.shape)
    # ax = Axes3D(plt.figure(figsize = (8, 5)))
    # ax.scatter3D(x[:, 0], x[:, 1], y, color = "blue") # 入力データの分布を表示
    # ax.plot_wireframe(x_grid, y_grid, z, color = "red") # 回帰平面の表示


    # 多項式回帰
    # 適当な学習データを生成
    # np.random.seed(0) # 乱数生成器のシード値を固定
    # x = np.random.random(10) * 10
    # y = x + np.random.randn(10) # 2*x + 1 にノイズが乗った点
    
    x = np.array([1, 2, 3, 6, 7, 9]) # 学習データの説明変数x
    y = np.array([1, 3, 3, 5, 4, 6]) # 学習データの目的変数y

    model = PolynomialRegression(3) # 6次関数で近似 (例)
    model.fit(x, y)
    
    #print("重み:", model.weight)
    print("推定値:", model.predict(x))

    # # グラフの表示
    # plt.scatter(x, y, color = "blue")
    # xx = np.linspace(x.min(), x.max(), 300) # x_minからx_maxまで等間隔に300の点を生成
    # yy = np.array([model.predict(i) for i in xx])
    # plt.plot(xx, yy, color = "red")
    # plt.show()