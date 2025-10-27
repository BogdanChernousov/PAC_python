class Lupa:
    def __init__(self, name):
        self.name = name
        self.salary = 0

    def take_salary(self, count):
        self.salary += count

    def do_work(self, file1, file2):
        a = []
        b = []

        with open(file1, "r") as f:
            for line in f:
                row = [int(x) for x in line.split()]
                a.append(row)

        with open(file2, "r") as f:
            for line in f:
                row = [int(x) for x in line.split()]
                b.append(row)

        result = []
        for i in range(len(a)):
            row = []
            for j in range(len(a[0])):
                row.append(a[i][j] - b[i][j])
            result.append(row)

        print(f"результат работы {self.name}: ")
        for row in result:
            print(row)


class Pupa:
    def __init__(self, name):
        self.name = name
        self.salary = 0

    def take_salary(self, count):
        self.salary += count

    def do_work(self, file1, file2):
        a = []
        b = []

        with open(file1, "r") as f:
            for line in f:
                row = [int(x) for x in line.split()]
                a.append(row)

        with open(file2, "r") as f:
            for line in f:
                row = [int(x) for x in line.split()]
                b.append(row)

        result = []
        for i in range(len(a)):
            row = []
            for j in range(len(a[0])):
                row.append(a[i][j] + b[i][j])
            result.append(row)

        print(f"результат работы {self.name}: ")
        for row in result:
            print(row)


class Accountant:
    def __init__(self, name):
        self.name = name

    def give_salary(self, worker, amount):
        worker.take_salary(amount)
        print(f"{worker.name} получил {amount}")

if __name__ == "__main__":
    pupa = Pupa("Pupa")
    lupa = Lupa("Lupa")
    acc = Accountant("Бухгалтерия")

    # Даем зарплату
    acc.give_salary(pupa, 123)
    acc.give_salary(lupa, 456)


    pupa.do_work("matrix_1.txt", "matrix_2.txt")
    print()
    lupa.do_work("matrix_1.txt", "matrix_2.txt")
    print()
    print("Счета: ", pupa.salary, lupa.salary)