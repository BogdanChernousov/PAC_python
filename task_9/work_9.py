import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data():
    """Загрузка и подготовка данных"""
    df = pd.read_csv('train.csv')

    # Заполнение пропусков
    df = df.fillna({
        'Age': df['Age'].median(),
        'Embarked': df['Embarked'].mode()[0],
    })

    # Базовые признаки
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Подготовка фич
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
    X = pd.get_dummies(df[features])
    y = df['Survived']

    return X, y


def split_data(X, y):
    """Разделение данных на train/val/test"""
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_data(X_train, X_val, X_test):
    """Масштабирование данных"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled


def train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test):
    """Обучение Random Forest"""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None]
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    return {'val_accuracy': best_model.score(X_val, y_val),
            'test_accuracy': best_model.score(X_test, y_test)}


def train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test):
    """Обучение XGBoost"""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.2]
    }

    grid = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric='logloss'),
        param_grid,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    return {
        'val_accuracy': best_model.score(X_val, y_val),
        'test_accuracy': best_model.score(X_test, y_test)
    }


def train_logistic_regression(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test):
    """Обучение Logistic Regression"""
    # требует масштабирования
    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['liblinear']
    }

    grid = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        param_grid,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)
    best_model = grid.best_estimator_

    return {
        'val_accuracy': best_model.score(X_val_scaled, y_val),
        'test_accuracy': best_model.score(X_test_scaled, y_test)
    }


def train_knn(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test):
    """Обучение KNN"""
    # требует масштабирования
    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }

    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)
    best_model = grid.best_estimator_

    return {
        'val_accuracy': best_model.score(X_val_scaled, y_val),
        'test_accuracy': best_model.score(X_test_scaled, y_test)
    }


def print_results(results):
    """Вывод результатов"""

    for name, metrics in results.items():
        print(f"{name}: ")
        print(f"   Val: {metrics['val_accuracy']:.3f}")  # ← Исправлено
        print(f"   Test: {metrics['test_accuracy']:.3f}")  # ← Исправлено

    best_name = max(results, key=lambda x: results[x]['test_accuracy'])
    best_acc = results[best_name]['test_accuracy']

    print(f"\nЛучшая модель: {best_name}")
    print(f"Точность на тесте: {best_acc:.3f}")


def main():

    # 1. Загрузка и подготовка данных
    X, y = load_and_prepare_data()

    # 2. Разделение данных
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 3. Масштабирование
    X_train_scaled, X_val_scaled, X_test_scaled = scale_data(X_train, X_val, X_test)

    # 4. Обучение моделей
    results = {}

    results['Random Forest'] = train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test)
    results['XGBoost'] = train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test)
    results['Logistic Regression'] = train_logistic_regression(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)
    results['KNN'] = train_knn(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)

    # 5. Вывод результатов
    print_results(results)


if __name__ == "__main__":
    main()