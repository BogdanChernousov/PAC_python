import pandas as pd
import numpy as np

cinema_sessions = pd.read_csv('cinema_sessions.csv', sep=r'\s+')
titanic_data = pd.read_csv('titanic_with_labels.csv', sep=r'\s+')


print(titanic_data['drink'].unique())
# -------------------------------1--------------------------------------

# 1. Пол (sex): отфильтровать строки где пол не указан, преобразовать в 0/1
titanic_data = titanic_data[titanic_data['sex'].notna()]
titanic_data['sex'] = titanic_data['sex'].str.lower().str.replace('"', '').str.strip()
titanic_data = titanic_data[titanic_data['sex'].isin(['м', 'ж'])]
titanic_data['sex'] = titanic_data['sex'].map({'ж': 0, 'м': 1})

# 2. Номер ряда (row_number): заполнить NaN и исправить отрицательные значения
titanic_data['row_number'] = pd.to_numeric(titanic_data['row_number'], errors='coerce')
max_row = titanic_data['row_number'].max()
titanic_data['row_number'] = titanic_data['row_number'].fillna(max_row)

# 3. Количество выпитого (liters_drunk): отфильтровать выбросы
titanic_data['liters_drunk'] = pd.to_numeric(titanic_data['liters_drunk'], errors='coerce')
reasonable_mask = (titanic_data['liters_drunk'] >= 0) & (titanic_data['liters_drunk'] <= 5)
mean_liters = titanic_data[reasonable_mask]['liters_drunk'].mean()
titanic_data['liters_drunk'] = titanic_data['liters_drunk'].astype(float)
titanic_data.loc[~reasonable_mask, 'liters_drunk'] = mean_liters

# -------------------------------2--------------------------------------

# 1. Возраст (age): разделить на 3 группы
titanic_data['age'] = pd.to_numeric(titanic_data['age'], errors='coerce')

def get_age_group(age):
    if age < 18:
        return 'child'
    elif age <= 50:
        return 'adult'
    else:
        return 'elderly'

titanic_data['age_group'] = titanic_data['age'].apply(get_age_group)
age_dummies = pd.get_dummies(titanic_data['age_group'], prefix='age')
titanic_data = pd.concat([titanic_data, age_dummies], axis=1)
titanic_data = titanic_data.drop(['age', 'age_group'], axis=1)

# 2. Напиток (drink): преобразовать в 0/1
titanic_data['drink'] = titanic_data['drink'].str.lower().str.replace('"', '').str.strip()

# Фильтруем строки где drink - это число (ошибка в данных)
titanic_data = titanic_data[~titanic_data['drink'].str.isnumeric()]
alcoholic_drinks = ['strong beer', 'beerbeer', 'bugbeer', 'Наше пиво']
titanic_data['is_alcoholic'] = titanic_data['drink'].isin(alcoholic_drinks).astype(int)

# 3. Время сеанса по номеру чека
merged_data = titanic_data.merge(cinema_sessions, on='check_number', how='left')

def get_session_period(time_str):
    hour = int(time_str.split(':')[0])
    if hour < 12:
        return 'morning'
    elif hour < 18:
        return 'day'
    else:
        return 'evening'

merged_data['session_period'] = merged_data['session_start'].apply(get_session_period)
session_dummies = pd.get_dummies(merged_data['session_period'], prefix='session')
final_data = pd.concat([merged_data, session_dummies], axis=1)
final_data = final_data.drop(['session_start', 'session_period'], axis=1)

# ФИНАЛЬНАЯ ОЧИСТКА ДАННЫХ
final_data = final_data.dropna()

for col in final_data.columns:
    if final_data[col].dtype == 'bool':
        final_data[col] = final_data[col].astype(int)

final_data['sex'] = final_data['sex'].astype(int)
final_data['label'] = final_data['label'].astype(int)
final_data['row_number'] = final_data['row_number'].astype(int)

final_data['liters_drunk'] = final_data['liters_drunk'].round(2)

final_data.to_csv('prepared_data.csv', index=False, sep=',')
