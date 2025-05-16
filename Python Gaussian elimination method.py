import numpy as np


def gaussian_elimination(A, b):
    rows = cols = len(A)
    augm_mat = np.column_stack((A, b.T)) if b is not None else A.copy()

    # 前向消元
    for col in range(cols):
        # 检查主元是否为0
        if augm_mat[col, col] == 0:
            print("主元为0，无法求解")
            exit()

        # 归一化当前行
        augm_mat[col] = augm_mat[col] / augm_mat[col, col]

        # 消去下方行
        for row in range(col + 1, rows):
            augm_mat[row] = augm_mat[row] - (augm_mat[row][col]/augm_mat[col][col])*augm_mat[col]

    print("前向消元后的矩阵:")
    print(augm_mat)
    print("-" * 50)

    # 回代
    count = rows - 2 # 需要回代的行的序号
    while count >= 0:
        for i in range(rows - count - 1):
            augm_mat[count] -= \
                ((augm_mat[count][count + 1 + i] /
                 augm_mat[count + 1 + i][count + 1 + i]) *
                 augm_mat[count + 1 + i])
        count -= 1

    print("回代后的矩阵:")
    print(augm_mat)
    print("-" * 50)

    # 提取结果
    solution = augm_mat[:, cols]
    print("解向量:")
    print(solution)

    return solution


# 测试
A = np.array([[2, -1, 3], [4, 2, 5], [1, 2, 0]], dtype=float)
b = np.array([[1, 4, 7]], dtype=float)
gaussian_elimination(A, b)