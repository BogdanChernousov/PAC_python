import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class MyRandomForest:
    """Упрощенная реализация Random Forest"""

    def __init__(self, n_estimators=50, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        """Обучение случайного леса"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        self.trees = []

        for i in range(self.n_estimators):
            # Бутстрэп выборка
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Простое дерево без лишних параметров
            tree = DecisionTreeClassifier(
                max_features='sqrt',  # Главное - ограничение признаков
                random_state=self.random_state + i
            )
            tree.fit(X_boot, y_boot)

            self.trees.append(tree)

    def predict(self, X):
        """Предсказание методом голосования"""

        # Собираем предсказания всех деревьев
        all_predictions = np.array([tree.predict(X) for tree in self.trees])

        # Majority voting
        final_predictions = []
        for i in range(X.shape[0]):
            values, counts = np.unique(all_predictions[:, i], return_counts=True)
            final_predictions.append(values[np.argmax(counts)])

        return np.array(final_predictions)

def load_data():
    """Загрузка данных"""
    df = pd.read_csv('titanic_prepared.csv')
    return df


def prepare_data(df):
    """Подготовка данных"""
    X = df.drop('label', axis=1)
    y = df['label']
    return X, y

def split_data_special(X, y):
    """Разделение на train/test (90%/10%)"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def train_single_tree(X_train, X_test, y_train, y_test):
    """Обучение одного Decision Tree"""
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def train_my_random_forest(X_train, X_test, y_train, y_test):
    """Обучение нашего Random Forest"""
    rf = MyRandomForest(n_estimators=50, random_state=42)
    rf.fit(X_train.values, y_train.values)
    y_pred = rf.predict(X_test.values)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def train_official_random_forest(X_train, X_test, y_train, y_test):
    """Обучение официального Random Forest для сравнения"""
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def main():

    # 1. Загрузка данных
    df = load_data()
    X, y = prepare_data(df)

    # 2. Разделение на train/test (90%/10%)
    X_train, X_test, y_train, y_test = split_data_special(X, y)

    # 3. Обучение одного дерева
    tree_accuracy = train_single_tree(X_train, X_test, y_train, y_test)
    print(f"\nТочность одного Decision Tree: {tree_accuracy:.3f}")

    # 4. Обучение нашего Random Forest
    my_rf_accuracy = train_my_random_forest(X_train, X_test, y_train, y_test)
    print(f"Точность нашего Random Forest: {my_rf_accuracy:.3f}")

    # 5. Обучение официального Random Forest
    official_rf_accuracy = train_official_random_forest(X_train, X_test, y_train, y_test)
    print(f"Точность официального Random Forest: {official_rf_accuracy:.3f}")

if __name__ == "__main__":
    main()