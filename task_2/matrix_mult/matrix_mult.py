import argparse

def read_matrices_from_file(filename):
    matrix_1 = []
    matrix_2 = []
    cur_matrix = matrix_1

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line == '':
                cur_matrix = matrix_2
                continue

            row = list(map(int, line.split()))
            cur_matrix.append(row)

    return matrix_1, matrix_2


def multiply_matrices(matrix_1, matrix_2):
    if len(matrix_1[0]) != len(matrix_2):
        raise ValueError("Нельзя умножить матрицы")

    result = []
    for i in range(len(matrix_1)):
        row = []
        for j in range(len(matrix_2[0])):
            element = 0
            for k in range(len(matrix_2)):
                element += matrix_1[i][k] * matrix_2[k][j]
            row.append(element)
        result.append(row)

    return result


def write_matrix_to_file(filename, matrix):
    with open(filename, 'w', encoding='utf-8') as file:

        for row in matrix:
            line = ' '.join(map(str, row)) + '\n'
            file.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_file", type=str)
    parser.add_argument("res_file", type=str)
    args = parser.parse_args()

    try:
        matrix_1, matrix_2 = read_matrices_from_file(args.matrix_file)

        result_matrix = multiply_matrices(matrix_1, matrix_2)

        write_matrix_to_file(args.res_file, result_matrix)

        print(f"Выполнено")

    except FileNotFoundError:
        print(f"Ошибка: файл {args.matrix_file} не найден")
    except ValueError as e:
        print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")


if __name__ == "__main__":
    main()