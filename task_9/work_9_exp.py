import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------- 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ----------------------

def load_and_prepare_data():
    """
    Загрузка исходных данных
    """
    df = pd.read_csv('train.csv')
    return df


def prepare_features(df):
    """
    Подготовка признаков
    """
    # Заполнение пропущенных значений
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Создание новых признаков
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Извлечение Title
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # Группировка редких Title
    common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
    df['Title'] = df['Title'].apply(lambda x: x if x in common_titles else 'Rare')

    # Кодирование категориальных переменных
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title', 'Pclass'],
                        prefix=['Sex', 'Emb', 'Title', 'Pclass'])

    # Явно указываем фичи которые создадутся
    features = ['Age', 'Fare', 'FamilySize', 'IsAlone',
                'Sex_female', 'Sex_male',
                'Emb_C', 'Emb_Q', 'Emb_S',
                'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare',
                'Pclass_1', 'Pclass_2', 'Pclass_3']

    return df[features]


# ---------------------- 2. РАЗДЕЛЕНИЕ НА TRAIN/VAL/TEST ----------------------

def split_data(X, y):
    """
    Разделение на тренировочную, валидационную и тестовую части
    """
    # 60% train, 20% val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------- 3. МАСШТАБИРОВАНИЕ ДАННЫХ ----------------------

def scale_data(X_train, X_val, X_test):
    """
    Масштабирование данных (нужно для LR и KNN)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled


# ---------------------- 4. ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ ----------------------

def train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Обучение Random Forest с подбором гиперпараметров
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    val_accuracy = best_model.score(X_val, y_val)
    test_accuracy = best_model.score(X_test, y_test)

    return {
        'model': best_model,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy
    }


def train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Обучение XGBoost с подбором гиперпараметров
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }

    grid = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric='logloss'),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    val_accuracy = best_model.score(X_val, y_val)
    test_accuracy = best_model.score(X_test, y_test)

    return {
        'model': best_model,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy
    }


def train_logistic_regression(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test):
    """
    Обучение Logistic Regression с подбором гиперпараметров
    """
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [2000, 5000]
    }

    grid = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_
    val_accuracy = best_model.score(X_val_scaled, y_val)
    test_accuracy = best_model.score(X_test_scaled, y_test)

    return {
        'model': best_model,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy
    }


def train_knn(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test):
    """
    Обучение KNN с подбором гиперпараметров
    """
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_
    val_accuracy = best_model.score(X_val_scaled, y_val)
    test_accuracy = best_model.score(X_test_scaled, y_test)

    return {
        'model': best_model,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy
    }


# ---------------------- 5. ОСНОВНАЯ ЛОГИКА ----------------------

def main():
    """
    Основная функция выполнения задания
    """
    # 1. Загрузка и подготовка данных
    df = load_and_prepare_data()
    X = prepare_features(df)
    y = df['Survived']

    # 2. Разделение на train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 3. Масштабирование данных
    X_train_scaled, X_val_scaled, X_test_scaled = scale_data(X_train, X_val, X_test)

    # 4. Обучение моделей
    results = {}

    # Random Forest
    results['Random Forest'] = train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test)

    # XGBoost
    results['XGBoost'] = train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test)

    # Logistic Regression
    results['Logistic Regression'] = train_logistic_regression(
        X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    )

    # KNN
    results['KNN'] = train_knn(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)

    # 5. Вывод результатов
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
    best_test_accuracy = results[best_model_name]['test_accuracy']

    print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model_name}")
    print(f"ТОЧНОСТЬ НА ТЕСТЕ: {best_test_accuracy:.4f}")


if __name__ == "__main__":
    main()