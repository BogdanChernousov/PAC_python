import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import precision_score, recall_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. Загрузка MNIST данных
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(int)
X = X / 255.0  # нормализация

# Разделение на train/test и преобразование в массивы numpy
X_train, y_train = X[:60000].values, y[:60000].values
X_test, y_test = X[60000:].values, y[60000:].values

# 2. Рассчитать average_digit для каждой цифры 0-9
avg_digits = []
for digit in range(10):
    mask = y_train == digit
    avg_digit = X_train[mask].mean(axis=0)
    avg_digits.append(avg_digit)

avg_digits = np.array(avg_digits)  # Преобразуем в numpy массив для векторизации

# 3. Классификатор для одной цифры
class SimpleClassifier:
    def __init__(self, avg_digit, bias=0.0):
        self.weights = avg_digit
        self.bias = bias

    def predict(self, x):
        similarity = np.dot(x, self.weights) + self.bias
        return 1 if similarity >= 0 else 0

    def get_similarity(self, x):
        return np.dot(x, self.weights) + self.bias

# 4. Создать 10 классификаторов с рассчитаными bias
print("\nРасчёт bias для каждого классификатора:")
biases = np.zeros(10)
for i in range(10):
    w = avg_digits[i]
    scores = X_train @ w

    pos_scores = scores[y_train == i]
    neg_scores = scores[y_train != i]

    threshold = 0.5 * (pos_scores.mean() + neg_scores.mean())
    biases[i] = -threshold

    print(f"Классификатор {i}: threshold = {threshold:.4f}, bias = {biases[i]:.4f}")

classifiers = [SimpleClassifier(avg_digits[i], bias=biases[i]) for i in range(10)]

# 4.1. Рассчитать точность каждого классификатора
print("\nТочность каждого классификатора:")
for i in range(10):
    correct = 0
    total = 0
    for j in range(min(1000, len(X_test))):
        x = X_test[j]
        true_label = 1 if y_test[j] == i else 0
        pred_label = classifiers[i].predict(x)
        if pred_label == true_label:
            correct += 1
        total += 1
    accuracy = correct / total
    print(f"Классификатор {i}: {accuracy:.4f}")


# 5. Объединить в одну модель
class MultiDigitModel:
    def __init__(self, classifiers, biases):
        self.classifiers = classifiers
        self.biases = biases
        self.weights = avg_digits  # Сохраняем веса для векторизации

    def predict_vector(self, x):
        return np.array([clf.predict(x) for clf in self.classifiers])    # [0,0,0,1,0,0...]

    def predict_similarity_vector(self, x):
        return np.array([clf.get_similarity(x) for clf in self.classifiers])   # [12.5, 10.0, -8.0, 45.6, 0.2...]

    def predict_digit(self, x):

        scores = [clf.get_similarity(x) for clf in self.classifiers]     # возвращает число с большей схожестью
        return np.argmax(scores)

# Создаём модель
model = MultiDigitModel(classifiers, biases)

# 6. Тестирование на всех тестовых примерах
n_test = len(X_test)
y_pred_digits = np.array([model.predict_digit(x) for x in X_test])

# 7. Рассчитать precision и recall
precision_macro = precision_score(y_test, y_pred_digits, average='macro')
recall_macro = recall_score(y_test, y_pred_digits, average='macro')

print(f"\nРезультаты на ВСЕХ тестовых примерах ({n_test}):")
print(f"Accuracy: {np.mean(y_pred_digits == y_test):.4f}")
print(f"Precision (macro): {precision_macro:.4f}")
print(f"Recall (macro): {recall_macro:.4f}")

# Точность классификации цифр
accuracy = np.mean(np.array(y_pred_digits) == y_test[:n_test])
print(f"Accuracy (цифры): {accuracy:.4f}")

# 8. Визуализация t-SNE (необработанные данные - исходные пиксели)
# Берём по 30 изображений каждого класса
n_per_class = 30
X_sample = []
y_sample = []

for digit in range(10):
    indices = np.where(y_test == digit)[0][:n_per_class]
    X_sample.extend(X_test[indices])
    y_sample.extend(y_test[indices])

X_sample = np.array(X_sample)
y_sample = np.array(y_sample)

# Применяем t-SNE к исходным пикселям
tsne_raw = TSNE(n_components=2, random_state=42)
X_tsne_raw = tsne_raw.fit_transform(X_sample)

plt.figure(figsize=(10, 8))
for digit in range(10):
    mask = y_sample == digit
    plt.scatter(X_tsne_raw[mask, 0], X_tsne_raw[mask, 1], label=f'Digit {digit}', alpha=0.7)
plt.title("t-SNE: Исходные пиксели (784D -> 2D)")
plt.legend()
plt.tight_layout()

# 9. Визуализация t-SNE (вектора модели - логиты)
similarity_vectors = np.array([model.predict_similarity_vector(x) for x in X_sample])

# t-SNE для векторов логитов
tsne_similarity = TSNE(n_components=2, random_state=42)
similarity_tsne = tsne_similarity.fit_transform(similarity_vectors)

plt.figure(figsize=(10, 8))
for digit in range(10):
    mask = y_sample == digit
    plt.scatter(similarity_tsne[mask, 0], similarity_tsne[mask, 1], label=f'Digit {digit}', alpha=0.7)
plt.title("t-SNE: Вектора сходства (логиты) (10D -> 2D)")
plt.legend()
plt.tight_layout()

plt.show()