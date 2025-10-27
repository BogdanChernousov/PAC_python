import argparse

def read_matrices_from_file(filename):
    matrix = []
    kernel = []
    current_matrix = matrix

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line == '':
                current_matrix = kernel
                continue
            row = list(map(int, line.split()))
            current_matrix.append(row)

    return matrix, kernel


def fold_matrices(matrix, kernel):
    if len(kernel) > len(matrix) or len(kernel[0]) > len(matrix[0]):
        raise ValueError("Ядро больше матрицы")

    result_height = len(matrix) - len(kernel) + 1
    result_width = len(matrix[0]) - len(kernel[0]) + 1

    result = []

    for i in range(result_height):
        row = []
        for j in range(result_width):
            fold_sum = 0
            for ki in range(len(kernel)):
                for kj in range(len(kernel[0])):
                    fold_sum += matrix[i + ki][j + kj] * kernel[ki][kj]
            row.append(fold_sum)
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
        matrix, kernel = read_matrices_from_file(args.matrix_file)
        result_matrix = fold_matrices(matrix, kernel)
        write_matrix_to_file(args.res_file, result_matrix)
        print("Выполнено")

    except FileNotFoundError:
        print(f"Ошибка: файл {args.matrix_file} не найден")
    except ValueError as e:
        print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")


if __name__ == "__main__":
    main()