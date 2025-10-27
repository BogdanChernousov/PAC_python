class Item:
    def __init__(self, ripe, count=1, max_count=32, color='green', saturation=10):
        self._count = count
        self._max_count = max_count
        self._color = color
        self._ripe = ripe
        self._saturation = saturation

    def update_count(self, val):
        if val <= self._max_count:
            self._count = val
            return True
        return False

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, val):
        if val <= self._max_count:
            self._count = val

    @staticmethod
    def static():
        print('I am function')

    @classmethod
    def my_name(cls):
        return cls.__name__

    @property
    def color(self):
        return self._color

    @property
    def eatable(self):
        return self._ripe

    def __add__(self, num):
        return self.count + num if 0 < self.count + num < self._max_count else False


    def __sub__(self, num):
        return self.count - num if 0 < self.count - num < self._max_count else False

    def __e_add__(self, num):
        self.count += num
        return self.count if 0 < self.count < self._max_count else False

    def __e_sub__(self, num):
        self.count -= num
        return self.count if 0 < self.count < self._max_count else False

    def __e_mul__(self, num):
        self.count *= num
        return self.count if 0 < self.count < self._max_count else False

    def __mul__(self, num):
        """ Умножение на число """
        return self.count * num if 0 < self.count * num < self._max_count else False

    def __lt__(self, num):
        """ Сравнение меньше """
        return self.count < num

    def __gt__(self, num):
        return self.count > num

    def __le__(self, num):
        return self.count <= num

    def __ge__(self, num):
        return self.count >= num

    def __eq__(self, num):
        return self.count == num

    def __len__(self):
        """ Получение длины объекта """
        return self.count