import numpy as np


def perceptron(train_data, train_label, threshold=1, rate=1):
    """
    感知机 返回训练得到的参数(一般方法)
    参数：
        train_data: 数据
        train_label: 数据标签(-1 or 1)
    """
    x_ = train_data
    y_ = train_label
    w = np.zeros(x_[0].shape)
    b = 0.0
    total_x_num = x_.shape[0]

    while True:
        right_class_num = 0
        for i in range(total_x_num):
            if y_[i] * (np.dot(x_[i], w) + b) <= 0:
                # print(i)
                w += rate * y_[i] * x_[i]
                b += rate * y_[i]
                right_class_num = 0
                break
            else:
                right_class_num += 1

        correct = right_class_num / total_x_num
        if correct >= threshold:
            return w, b


def perceptron_dual(train_data, train_label, threshold=1, rate=1):
    """
    感知机 返回训练得到的参数(对偶形式)
    参数：
        train_data: 数据
        train_label: 数据标签(-1 or 1)
    """
    x_ = train_data
    y_ = train_label
    total_x_num = x_.shape[0]
    alpha = np.zeros(total_x_num)
    b = 0.0

    gram = np.zeros((total_x_num, total_x_num))

    for i in range(total_x_num):
        for j in range(total_x_num):
            gram[i][j] = x_[i] * np.mat(x_[j]).transpose()

    while True:
        right_class_num = 0
        for i in range(total_x_num):
            w_x = 0.0
            for j in range(total_x_num):  # 算出 W * x
                w_x += alpha[j] * y_[j] * gram[j][i]
            if y_[i] * (w_x + b) <= 0:
                # print(i)
                alpha[i] += rate
                b += rate * y_[i]
                right_class_num = 0
                break
            else:
                right_class_num += 1

        correct = right_class_num / total_x_num
        if correct >= threshold:
            # w = 0
            # for i in range(total_x_num):
            #     w += alpha[i] * y_[i] * x_[i]
            return alpha, b


if __name__ == "__main__":
    train_data = [[3, 3],
                  [4, 3],
                  [1, 1]]
    train_label = [1, 1, -1]
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    print(perceptron_dual(train_data, train_label))
