from work_9 import (
    load_and_prepare_data,
    split_data,
    scale_data,
    train_random_forest,
    train_xgboost,
    train_logistic_regression,
    train_knn
)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def get_top_features(X, y, n_features):
    """Выбор n самых важных признаков с помощью RandomForest"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Получаем важность признаков
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # Выбираем топ-n признаков
    top_features = feature_importance.head(n_features)['feature'].tolist()

    print(f"{'-' * 30}")
    print(f"Топ-{n_features} самых важных признаков:")
    for i, (_, row) in enumerate(feature_importance.head(n_features).iterrows(), 1):
        print(f"{i}. {row['feature']} - {row['importance']:.3f}")
    print(f"{'-' * 30}")

    return top_features


def print_feature_results(results, n_features):
    """Вывод результатов для анализа признаков"""

    for name, metrics in results.items():
        print(f"{name}: ")
        print(f"   Val: {metrics['val_accuracy']:.3f}")
        print(f"   Test: {metrics['test_accuracy']:.3f}")

    best_name = max(results, key=lambda x: results[x]['test_accuracy'])
    best_acc = results[best_name]['test_accuracy']

    print(f"\nЛучшая модель: {best_name}")
    print(f"Точность на тесте: {best_acc:.3f}\n")


def main():
    """Основная функция для выбора признаков"""
    # 1. Загрузка всех данных
    X_full, y = load_and_prepare_data()

    # 2. Тестируем для разного количества признаков
    feature_counts = [2, 4, 8]

    for n_features in feature_counts:
        # Выбираем топ-n признаков
        top_features = get_top_features(X_full, y, n_features)

        # Создаем подмножество данных с выбранными признаками
        X_subset = X_full[top_features]

        # Разделение данных
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_subset, y)

        # Масштабирование (для моделей, которые в этом нуждаются)
        X_train_scaled, X_val_scaled, X_test_scaled = scale_data(X_train, X_val, X_test)

        # Обучение моделей
        results = {}

        results['Random Forest'] = train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test)
        results['XGBoost'] = train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test)
        results['Logistic Regression'] = train_logistic_regression(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)
        results['KNN'] = train_knn(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)

        # Вывод результатов
        print_feature_results(results, n_features)


if __name__ == "__main__":
    main()