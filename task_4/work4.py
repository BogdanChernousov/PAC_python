import numpy as np

# --------------------------------------------- 1

def sort_arr(arr):
    unique, counts = np.unique(arr, return_counts=True)

    indices = np.searchsorted(unique, arr)

    freqs_list = counts[indices]

    sorted_indices = np.argsort(-freqs_list)
    return arr[sorted_indices]

# --------------------------------------------- 2

def unq_colors(h, w, arr=None):

    if arr is None:
        arr = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)

    flat_arr = arr.reshape(-1, 3)

    unique_colors = np.unique(flat_arr, axis=0) # axis = 0 - ищет уникальные среди строк

    cnt = len(unique_colors)

    return cnt

# --------------------------------------------- 3

def movav(arr, win):
    kernel = np.ones(win) / win
    return np.convolve(arr, kernel, mode='valid')

# --------------------------------------------- 4

def triangle(arr):

    a = arr[:, 0]
    b = arr[:, 1]
    c = arr[:, 2]

    crit1 = a + b > c
    crit2 = b + c > a
    crit3 = a + c > b

    result = arr[crit1 & crit2 & crit3]

    return result

# --------------------------------------------- 5

def equation(arr, arr2):

    arr_inv = np.linalg.inv(arr)
    x = arr_inv @ arr2     # @ - матричное умножение

    return x

# --------------------------------------------- 6
def svd_mat(arr):

    U, S, Vt = np.linalg.svd(arr)

    return U, S, Vt

#----------------------------------------------------------------
def main():
    print("------------------------------1------------------------------")
    arr = np.array(['b', 'a', 'b', 'c', 'd', 'd', 'c', 'e', 'e', 'e', 'e', 'c', 'd', 'e', 'd'])
    print("Изначальный массив: ",arr)
    print("Отсортированный по частоте: ", sort_arr(arr))

    print("------------------------------2------------------------------")
    arr2 = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],[[255, 0, 0], [0, 0, 255], [255, 255, 255]]])
    count = unq_colors(2, 3, arr2)
    count2 = unq_colors(10, 10)
    print("Уникальных цветов: ", count)
    print("Уникальных цветов: ", count2)

    print("------------------------------3------------------------------")
    arr3 = np.array([10, 2, 13, 54, 75, 1, 8, 19])
    mov_av_vec = movav(arr3, 3)
    print(mov_av_vec)

    print("------------------------------4------------------------------")
    arr4 = np.array([[3, 4, 5],[1, 2, 3],[5, 5, 5],[1, 10, 12],])
    tri = triangle(arr4)
    print("Изначальный массив: \n", arr4)
    print("Треугольнки: \n",tri)

    print("------------------------------5------------------------------")
    coef = np.array([[3, 4, 2],
                     [5, 2, 3],
                     [4, 3, 2]])
    print("Изначальная матрица: \n", coef)
    const = np.array([17, 23, 19])
    print("Константы: ",const)
    urav_res = equation(coef, const)
    print("Решения уравнения(x,y,z): ", urav_res)

    print("------------------------------6------------------------------")
    arr5 = np.matrix("1 0 1; 0 1 0; 1 0 1")
    print("Изначальная матрица: \n", arr5)
    U, S, Vt = svd_mat(arr5)
    print("U (левые сингулярные векторы):\n", U)
    print("\nS (сингулярные значения):\n", S)
    print("\nVt (V транспонированная(правые)):\n", Vt)


if __name__ == '__main__':
    main()