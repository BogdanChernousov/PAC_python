import random

#1
st = input("Введите слово: ")
st = "".join(st.split())

for i in range(len(st) // 2):
    if st[i] != st[-i-1]:
        print("Не палиндром")
        break
else:
    print("Палиндром")

#2
# st = input("Введите строку: ")
# lst = st.split(' ')
# max_len = 0
# for i in range(len(lst)):
#     temp_len = len(lst[i])
#     if temp_len > max_len:
#         max_len = temp_len
# print("Длина самого длинного слова: ", max_len)

#3
# lst = []
# even = 0
#
# for i in range(10):
#     r = random.randint(1,10)
#     lst.append(r)
# for i in range (len(lst)):
#     if lst[i] % 2 == 0:
#         even += 1
#
# print("Четных: ", even, " Нечетных: ", len(lst)-even)

#4
# dct = {'правда': 'истина', 'атака': 'нападение', 'земля': 'почва'}
# st = input("Введите текст: ")
#
# lst = st.split()
# res = []
#
# for i in lst:
#     if i in dct:
#         res.append(dct[i])
#     else:
#         res.append(i)
#
# new_st = ' '.join(res)
# print("Синонимичная строка: ", new_st)

#5
# n = int(input("Введите номер числа Фиббоначи: "))
#
# def fib(num):
#     if num < 2:
#         return num
#     return fib(num - 1) + fib(num - 2)
#
# print("Число Фиббоначи: ", fib(n-1))

#6
# with open('text.txt', 'r') as f:
#     lines = f.readlines()
#
# print("Кол-во строк: ", len(lines))
#
# all_words = 0
# all_sym = 0
# for i in range(len(lines)):
#     all_words += lines[i].count(' ') + 1
#     all_sym += len(lines[i]) - lines[i].count(' ')
#
# print("Кол-во слов: ", all_words)
# print("Кол-во символов:", all_sym - len(lines) + 1)

#7
# def geo_prog(b, q, n):
#     cur = b
#     for _ in range(n):
#         yield cur
#         cur *= q
#
# b = int(input("Введите начальное число: "))
# q = int(input("Введите знаменатель прогрессии: "))
# n = int(input("Введите количество элементов: "))
#
# print("Первые ", n, " членов геометрической прогрессии:")
# for i in geo_prog(b, q, n):
#     print(i)


