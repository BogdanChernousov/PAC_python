import argparse

parser = argparse.ArgumentParser()
parser.add_argument("N", type = int)
args = parser.parse_args()

row = [1]
for i in range(args.N):
    o = (2*args.N-i)*" " 
    print(o, f"{row}")
    row = [sum(x) for x in zip([0]+row, row+[0])]