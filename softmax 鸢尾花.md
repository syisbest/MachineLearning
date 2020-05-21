# softmax 鸢尾花

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def iris_type(s):
    class_label = {'setosa':0, 'versicolor':1, 'virginica':2}
    return class_label[s]
#读取数据并且将第五列字符串映射成数字
def load_dataset(file_path):
    datas = pd.io.parsers.read_csv(file_path, converters={4: iris_type})
    data = np.array(datas)
    # 用np.split按列（axis=1）进行分割
    # (4,):分割位置，前4列作为x的数据，第4列之后都是y的数据
    X, Y = np.split(data, (4,), axis=1)
    #X = X[:, 0:2]  # 取前两列特征
    return X, Y


def train(data_arr, label_arr, n_class, iters=1000, alpha=0.1, lam=0.01):
    n_samples, n_features = data_arr.shape  #150*4
    n_classes = n_class
    # 随机初始化权重矩阵
    weights = np.random.rand(n_class, n_features)
    # 计算 one-hot 矩阵
    y_one_hot = one_hot(label_arr, n_samples, n_classes)
    for i in range(iters):
        # 计算 m * k 的分数矩阵
        scores = np.dot(data_arr, weights.T)
        # 计算 softmax 的值
        probs = softmax(scores)
        # 求解梯度
        dw = -(1.0 / n_samples) * np.dot((y_one_hot - probs).T, data_arr) + lam * weights
        dw[:, 0] = dw[:, 0] - lam * weights[:, 0]
        # 更新权重矩阵
        weights = weights - alpha * dw
    return weights


def softmax(scores):
    # 计算总和
    sum_exp = np.sum(np.exp(scores), axis=1, keepdims=True)
    softmax = np.exp(scores) / sum_exp
    return softmax


def one_hot(label_arr, n_samples, n_classes):
    one_hot = np.zeros((n_samples, n_classes))#150*3
    one_hot[np.arange(n_samples), label_arr.T] = 1
    return one_hot


def predict(test_dataset, weights):
    scores = np.dot(test_dataset, weights.T)
    probs = softmax(scores)
    return np.argmax(probs, axis=1).reshape((-1, 1))


if __name__ == "__main__":
    X, Y = load_dataset('/home/shenyan/Desktop/数据集/iris.csv')
    #这里结果集要转成int
    Y=Y.astype(np.int32)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    weights = train(x_train, y_train, n_class=3)

    # 计算预测的准确率
    n_test_samples = x_test.shape[0]#150
    y_predict = predict(x_test, weights)
    accuray = np.sum(y_predict == y_test) / n_test_samples
    print("1707004642沈岩")
    print("准确率：")
    print(accuray)


```

![img](https://syisbest.github.io/yuanweihua/softmax.png)

