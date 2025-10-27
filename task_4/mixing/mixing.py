import numpy as np
import random
import argparse

def load_numbers_from_file(filename):
    with open(filename, 'r') as f:
        content = f.read().strip()
        numbers = [float(x) for x in content.split()]
    return np.array(numbers)

def mixing(text1, text2, p):
    real = load_numbers_from_file(text1)
    synt = load_numbers_from_file(text2)

    if len(real) != len(synt):
        raise ValueError("Массивы должны быть одинаковой длины")

    mask1 = np.random.random(len(real)) <= p
    mix1 = np.where(mask1, synt, real)

    mask2 = np.random.binomial(1, p, size=len(real))
    mix2 = np.where(mask2 == 1, synt, real)

    n_replace = int(p * len(real))
    mask3 = np.zeros(len(real), dtype=bool)
    mask3[:n_replace] = True
    mask3 = np.random.permutation(mask3)
    mix3 = np.where(mask3, synt, real)

    return mix1, mix2, mix3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_1", type=str)
    parser.add_argument("file_2", type=str)
    parser.add_argument("p", type=float)
    args = parser.parse_args()

    try:
        mix1, mix2, mix3 = mixing(args.file_1, args.file_2, args.p)
        print("mix1(random):     ", mix1)
        print("mix2(binomial):   ", mix2)
        print("mix3(permutation):", mix3)
    except FileNotFoundError as e:
        print(f"Ошибка: файл не найден - {e}")
    except ValueError as e:
        print(f"Ошибка в данных: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")

if __name__ == "__main__":
    main()