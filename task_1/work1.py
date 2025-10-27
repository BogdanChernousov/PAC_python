import random
import math
import os
import argparse

#1
""" a = random.randint(100, 1000)
b = 0
print(a)
while (a > 0):
    b += a % 10
    a //= 10
print(b) """

#2
""" r = random.randint(1, 100000)
sum = 0
print(r)
while (r > 0):
    sum += r % 10
    r //= 10
print(sum) """

""" #3
r = input()
S = 4*(math.pi)*(int(r)**2)
V = (4/3)*math.pi*(int(r)**3)
print(S, V) """

#4
""" y = input()
y = int(y)
if ((y%4==0 and y%100!=0) or y%400==0):
    print("Високосный")
else:
    print("Не високосный") """
    
#5
""" N = int(input())
print(2)
for i in range(1, N+1):
    for j in range(2, i+1):
        if i%j == 0:
            break
        if j == i-1:
            print(i) """
            
#6
""" X, Y = input().split()
X = int(X)
Y = int(Y)
S = X * 1.1**Y
print(int(S)) """

#7
t = os.walk("fff")
for i in t:
    print(i)
    
    
