import numpy as np
import pandas as pd

x1 = np.array([0, 1])


def get_x(n):
    if n == 1:
        return x1
    else:
        z = get_x(n - 1)
        return np.concatenate([z, 2 ** (n - 1) + z[::-1]])


def manhattan(x, y):
    return np.abs(x - y)


def euclidean(x, y):
    return np.power(x - y, 2)


def generate_dfs(n, fun):
    x = get_x(n)
    xb = [f'{j:0{n}b}' for j in x]
    t = np.array([x] * 2 ** n).reshape(2 ** n, 2 ** n)
    v = fun(t, t.T)
    df = pd.DataFrame(columns=xb, index=xb, data=v)

    max_distance = v.max().max()
    bit_max_distance = 0
    while 2 ** bit_max_distance < max_distance:
        bit_max_distance += 1

    ret = dict()

    for j in range(bit_max_distance):
        ret[j] = (df & (2 ** j)) // (2 ** j)

    return ret


def print_df(n, fun):
    d = generate_dfs(n, fun)
    for i in range(len(d)):
        print(i, d[i])


if __name__ == '__main__':
    print_df(3, euclidean)
