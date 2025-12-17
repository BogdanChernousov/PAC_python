import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ---------------------- 1. ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ----------------------

def load_and_prepare_data():
    """
    Загрузка исходных данных
    """
    df = pd.read_csv('wells_info_with_prod.csv')

    # Сохраняем исходные столбцы (дата и категориальный)
    df['SpudDate'] = pd.to_datetime(df['SpudDate'])
    #df['operatorNameIHS'] = df['operatorNameIHS']

    # Извлекаем новые признаки из даты
    df['SpudMonth'] = df['SpudDate'].dt.month
    df['SpudQuarter'] = df['SpudDate'].dt.quarter
    df['SpudYear'] = df['SpudDate'].dt.year

    # Временные интервалы
    df['PermitDate'] = pd.to_datetime(df['PermitDate'])
    df['CompletionDate'] = pd.to_datetime(df['CompletionDate'])
    df['FirstProductionDate'] = pd.to_datetime(df['FirstProductionDate'])

    df['PermitToSpudDays'] = (df['SpudDate'] - df['PermitDate']).dt.days
    df['DrillingDurationDays'] = (df['CompletionDate'] - df['SpudDate']).dt.days
    df['CompletionToProductionDays'] = (df['FirstProductionDate'] - df['CompletionDate']).dt.days

    # Технологические признаки
    df['TotalProppant'] = df['PROP_PER_FOOT'] * df['LATERAL_LENGTH_BLEND']
    df['TotalWater'] = df['WATER_PER_FOOT'] * df['LATERAL_LENGTH_BLEND']
    df['ProppantIntensity'] = df['PROP_PER_FOOT'] / df['LATERAL_LENGTH_BLEND']

    # Географический признак
    df['LateralDistance'] = np.sqrt(
        (df['LatWGS84'] - df['BottomHoleLatitude']) ** 2 +
        (df['LonWGS84'] - df['BottomHoleLongitude']) ** 2
    )

    # Категория длины ствола
    df['LateralLengthCategory'] = pd.cut(
        df['LATERAL_LENGTH_BLEND'],
        bins=[0, 4500, 7500, float('inf')],
        labels=['short', 'medium', 'long']
    )

    return df


# ---------------------- 2. РАЗДЕЛЕНИЕ НА TRAIN/TEST ----------------------

def split_train_test(df):
    """
    Выбор финальных признаков для модели
    """
    features = [
        'SpudMonth', 'SpudQuarter', 'SpudYear',
        'PermitToSpudDays', 'DrillingDurationDays', 'CompletionToProductionDays',
        'LATERAL_LENGTH_BLEND', 'PROP_PER_FOOT', 'WATER_PER_FOOT',
        'TotalProppant', 'TotalWater', 'ProppantIntensity',
        'LateralDistance',
        'operatorNameIHS', 'LateralLengthCategory',
        'BasinName', 'formation'
    ]

    X = df[features]
    y = df['Prod1Year']

    # Разделение на train/test (80/20)
    train_size = int(0.8 * len(df))
    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    return X_train, X_test, y_train, y_test


# ---------------------- 3. МАСШТАБИРОВАНИЕ ДАННЫХ ----------------------

def scale_data(X_train, X_test, y_train, y_test):
    print("\nМАСШТАБИРОВАНИЕ ДАННЫХ:")

    # Разделяем на числовые и категориальные признаки
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train.select_dtypes(exclude=['int64', 'float64']).columns

    # Масштабирование числовых признаков
    scaler_X = StandardScaler()
    X_train_num_scaled = scaler_X.fit_transform(X_train[num_cols])
    X_test_num_scaled = scaler_X.transform(X_test[num_cols])

    # Создаём DataFrame с масштабированными числовыми признаками
    X_train_scaled = pd.DataFrame(X_train_num_scaled, columns=num_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_num_scaled, columns=num_cols, index=X_test.index)

    # Добавляем категориальные признаки обратно
    X_train_scaled[cat_cols] = X_train[cat_cols]
    X_test_scaled[cat_cols] = X_test[cat_cols]

    # Масштабирование целевой переменной
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    # Краткий вывод
    print("\nПример целевой переменной (первые 3 значения):")
    print(f"До масштабирования:   {y_train.iloc[:3].values}")
    print(f"После масштабирования: {y_train_scaled[:3]}")

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled


# ---------------------- 4. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ----------------------

def save_results(df, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled):
    """
    Сохранение данных с признаками
    """
    df.to_csv('wells_with_features.csv', index=False)

    # Сохраняем масштабированные данные
    X_train_scaled.to_csv('X_train_scaled.csv', index=False)
    X_test_scaled.to_csv('X_test_scaled.csv', index=False)
    pd.DataFrame({'Prod1Year': y_train_scaled}).to_csv('y_train_scaled.csv', index=False)
    pd.DataFrame({'Prod1Year': y_test_scaled}).to_csv('y_test_scaled.csv', index=False)


# ---------------------- MAIN ----------------------

def main():
    # 1. Загрузка и подготовка данных
    df = load_and_prepare_data()

    # 2. Разделение на train/test
    X_train, X_test, y_train, y_test = split_train_test(df)

    # 3. Масштабирование данных
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = scale_data(X_train, X_test, y_train, y_test)

    # 4. Сохранение результатов
    save_results(df, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled)


if __name__ == "__main__":
    main()