import numpy as np
import torch
import matplotlib.pyplot as plt


# 第4题
# 随机生成正交矩阵
def rvs(dim=3):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,), int)
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder分解
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


# 第5题
# 生成随机正定矩阵
# 生成随机正定矩阵的算法，简单的说，生成一个随机的满秩方阵，再用方阵乘以他自己的转置即可
def positiveDefiniteMatrix(dim=3):
    A = np.random.rand(dim, dim)
    B = np.dot(A, A.transpose())
    return B


# 第6题
# 梯度下降的详细实现
def func_1d(P, q, x, r):
    """
   给定目标函数
    :param r: 随机生成的r
    :param q: 随机生成的一维列向量q
    :param P:随机生成的正定矩阵P
    :param x: x
    :return: 函数结果
   """
    return x.t().mm(P).mm(x).mm(torch.tensor([[0.5]])) + q.t().mm(x) + r


def get_learning_rate(P, q, x, grad):
    """
    精确直线搜索（exact line search）求学习率的方法 将Xn-t▽f(Xn)代入f 最终为t的一元二次函数 对t求导可求最优的t值。
    :param x: 初始值
    :param r: 随机生成的常数r
    :param q: 随机生成的向量q
    :param P: 随机生成的正定矩阵P
    """

    # 此处将
    a = grad.T.mm(grad)
    b = torch.tensor([[-0.5]]).mm(grad.T).mm(P).mm(x) - torch.tensor([[0.5]]).mm(x.T).mm(P).mm(grad) - q.T.mm(grad)
    step = -(b.item()) / (a.item())
    return step


def gradient_descent_1d(grad, cur_x, precision, max_iters, P, q, r):
    """
    一维问题的梯度下降法
    :param r: 随机生成的常数r
    :param q: 随机生成的向量q
    :param P: 随机生成的正定矩阵P
    :param grad: 目标函数的梯度
    :param cur_x: 当前 x 值，通过参数可以提供初始值
    :param precision: 设置收敛精度
    :param max_iters: 最大迭代次数
    :return: 局部最小值 x*
    """
    tensors = []
    for i in range(max_iters):
        # 迭代的停止条件，梯度的二范数小于指定的精度
        if torch.norm(grad) < precision:
            break

        # exactly line search learning rate calc
        learning_rate = get_learning_rate(P, q, cur_x, grad)
        cur_x = cur_x - grad * learning_rate
        tensors.append(cur_x)
        grad.data.zero_()
        res = func_1d(P, q, cur_x, r)
        res.backward()

        print("epoch:", i, "grad norm", torch.norm(grad).item(), " min value", res.item())

    print("min value x =", cur_x)

    # 绘图
    x = []
    y = []
    for i in range(len(tensors)):
        x.append(tensors[i][0][0].item())
        y.append(tensors[i][1][0].item())

    plt.plot(x, y)
    plt.show()
    return cur_x


# 第6题 End

# 第7题 给定n=2 随机生成P,q,x0,r,进行搜索
def dim2elsDemo():
    dim = 2
    P = torch.from_numpy(positiveDefiniteMatrix(dim).astype(np.float32))
    print("随机生成的初始正定矩阵P:")
    print(P)
    q = torch.randn((dim, 1))
    print("随机生成的向量q:")
    print(q)

    # x打开自动求梯度
    x = torch.randn((dim, 1), requires_grad=True)
    print("随机生成的初始值x:")
    print(x)
    r = torch.randn(1)
    print("随机生成的常数r:")
    print(r)

    print("函数结果r:")
    res = func_1d(P, q, x, r)
    # 求梯度
    res.backward()
    # 迭代精度
    precision = 0.001
    # 迭代周期
    epoch = 10000
    gradient_descent_1d(x.grad, x, precision, epoch, P, q, r)


if __name__ == '__main__':
    dim2elsDemo()
