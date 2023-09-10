import numpy as np
import matplotlib.pyplot as plt


def single_lr(x, y):
    """
    单特征训练简单线性模型
    """
    m = len(x)
    std_distance, mean_distance = 0, 0
    for i in range(m):
        mean_distance += (x[i] - np.mean(x)) * y[i]  # mean均值
        std_distance += (x[i] - np.mean(x)) ** 2
    w = mean_distance / float(std_distance)  # 回归线斜率
    b = np.mean(y) - w * np.mean(x)  # 回归线截距
    return w, b


def multiplex_lr(multi_x, y):
    """
    多元训练简单线性模型
    """
    f_num = len(multi_x)
    m = len(multi_x[0])
    x_means, ws = list(), list()
    for k in range(f_num):
        x = multi_x[k]
        std_distance, mean_distance = 0, 0
        for i in range(m):
            mean_distance += (x[i] - np.mean(x)) * y[i]  # mean均值
            std_distance += (x[i] - np.mean(x)) ** 2
        w = mean_distance / float(std_distance)  # 回归线斜率

        x_means.append(np.mean(x))
        ws.append(w)

    b = np.mean(y)
    for w, x_mean in zip(ws, x_means):
        b -= w * x_mean  # 回归线截距
    return ws, b


def inference(x, w, b):
    """ y = wx + b """
    if isinstance(w, list):
        y_pre = np.zeros(len(x[0]))
        for xx, ww in zip(x, w):
            y_pre += np.array(xx) * ww
        y_pre += b
        return y_pre
    elif isinstance(w, float):
        return np.array(x) * w + b


def inverse_inference(y, w, b):
    """ x_j = (y - b - w_i * x_i) / w_j """
    if isinstance(w, list):
        raise NotImplementedError
    elif isinstance(w, float):
        return np.array(y) - b / w
