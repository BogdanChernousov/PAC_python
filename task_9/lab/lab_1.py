import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np


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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    """Масштабирование данных для Logistic Regression"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_decision_tree(X_train, X_test, y_train, y_test):
    """Обучение Decision Tree"""
    dt = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    feature_importance = dt.feature_importances_
    return accuracy, feature_importance


def train_xgboost_model(X_train, X_test, y_train, y_test):
    """Обучение XGBoost"""
    xgb = XGBClassifier(random_state=42, max_depth=3, n_estimators=100)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def train_logistic_regression_model(X_train_scaled, X_test_scaled, y_train, y_test):
    """Обучение Logistic Regression"""
    lr = LogisticRegression(random_state=42, max_iter=2000)
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def get_top_two_features(X, feature_importance):
    """Выбор 2 самых важных признаков"""
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    top_features = feature_importance_df.head(2)['feature'].tolist()
    return top_features


def main():

    # 1. Загрузка данных
    df = load_data()
    X, y = prepare_data(df)

    # 2. Разделение на train/test (90%/10%)
    X_train, X_test, y_train, y_test = split_data_special(X, y)

    # 3. Масштабирование для Logistic Regression
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # 4. Обучение моделей
    print("\nТочность моделей:")

    # Decision Tree
    dt_accuracy, dt_importance = train_decision_tree(X_train, X_test, y_train, y_test)
    print(f"Decision Tree: {dt_accuracy:.3f}")

    # XGBoost
    xgb_accuracy = train_xgboost_model(X_train, X_test, y_train, y_test)
    print(f"XGBoost: {xgb_accuracy:.3f}")

    # Logistic Regression
    lr_accuracy = train_logistic_regression_model(X_train_scaled, X_test_scaled, y_train, y_test)
    print(f"Logistic Regression: {lr_accuracy:.3f}")

    # 5. Выбор 2 важных признаков
    top_features = get_top_two_features(X, dt_importance)
    print(f"\n2 самых важных признака: {top_features}")

    # Обучение на 2 признаках
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]

    dt_top_accuracy, _ = train_decision_tree(X_train_top, X_test_top, y_train, y_test)
    print(f"Decision Tree на 2 признаках: {dt_top_accuracy:.3f}")


if __name__ == "__main__":
    main()