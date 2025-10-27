import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("N", type = int)
args = parser.parse_args()

arr = []
for i in range(0, args.N):
    r = random.random()
    arr.append(r)
    
for i in range(args.N-1):
    for j in range(args.N-1-i):
        if arr[j] > arr[j+1]:
            arr[j], arr[j+1] = arr[j+1], arr[j]
    
print(f"{arr}")