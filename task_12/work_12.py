import numpy as np

class Neuron:
    def __init__(self, n_inputs):
        self.weights = np.random.uniform(-0.5, 0.5, n_inputs)
        self.bias = np.random.uniform(-0.5, 0.5)
        self.output = 0
        self.inputs = None
        self.z = 0  # Добавляем сохранение z (сумма до активации)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Принимает уже активированное значение self.output
        return x * (1 - x)

    def forward(self, x):
        self.inputs = x
        self.z = np.dot(x, self.weights) + self.bias
        self.output = self.sigmoid(self.z)
        return self.output

    def backward(self, delta, learning_rate):
        # delta уже включает производную от следующего слоя
        # Обновляем веса
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * delta * self.inputs[i]
        self.bias -= learning_rate * delta


class Model:
    def __init__(self):
        self.h1 = Neuron(2)
        self.h2 = Neuron(2)
        self.out = Neuron(2)

    def forward(self, x):
        h1_out = self.h1.forward(x)
        h2_out = self.h2.forward(x)
        return self.out.forward([h1_out, h2_out])

    def backward(self, x, target, learning_rate=0.7):
        # 1. Вычисляем ошибку выходного слоя
        output_error = self.out.output - target

        # 2. Дельта для выходного нейрона (учитываем производную его сигмоиды)
        delta_out = output_error * self.out.sigmoid_derivative(self.out.output)

        # 3. Ошибки для скрытых нейронов (цепное правило!)
        # Для каждого скрытого нейрона: его вклад в ошибку = вес соединения * delta_out
        h1_error = self.out.weights[0] * delta_out
        h2_error = self.out.weights[1] * delta_out

        # 4. Дельты для скрытых нейронов (учитываем их производные сигмоиды!)
        delta_h1 = h1_error * self.h1.sigmoid_derivative(self.h1.output)
        delta_h2 = h2_error * self.h2.sigmoid_derivative(self.h2.output)

        # 5. Обновляем веса (сначала скрытые, потом выходной - порядок не критичен)
        self.h1.backward(delta_h1, learning_rate)
        self.h2.backward(delta_h2, learning_rate)
        self.out.backward(delta_out, learning_rate)

def train():
    X = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]

    y = [0, 1, 1, 0]

    model = Model()

    for epoch in range(10000):
        total_error = 0
        for i in range(len(X)):
            pred = model.forward(X[i])
            error = (pred - y[i]) ** 2
            total_error += error
            model.backward(X[i], y[i], 0.5)

        # Печатаем ошибку каждые 1000 эпох
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Error: {total_error / len(X):.6f}")

    print("\nРезультаты после обучения:")
    for i in range(len(X)):
        pred = model.forward(X[i])
        print(f"{X[i]} -> {pred:.4f} (≈ {1 if pred > 0.5 else 0}) [true: {y[i]}]")

if __name__ == "__main__":
    train()