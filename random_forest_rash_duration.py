import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('длительность.xlsx', sheet_name='Лист1')

X = df.drop(columns=['Длит-ть сыпи (x)'])
y = df['Длит-ть сыпи (x)']

print(" Анализ данных ")
print(f"Размер данных: {X.shape}")
print(f"Пропуски в X: {X.isnull().sum().sum()}")
print(f"Пропуски в y: {y.isnull().sum()}")
print(f"\nРаспределение целевой переменной:")
print(f"Среднее: {y.mean():.2f}")
print(f"Медиана: {y.median():.2f}")
print(f"Станд. отклонение: {y.std():.2f}")
print(f"Мин: {y.min()}, Макс: {y.max()}")

# Заполняем пропуски
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nРазмер обучающей: {X_train.shape[0]}")
print(f"Размер тестовой: {X_test.shape[0]}")

# Подбор гиперпараметров
print("\n Подбор гиперпараметров ")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print(f"\nЛучшие параметры: {grid_search.best_params_}")

# Предсказание и оценка
y_pred = best_rf.predict(X_test)

print("\n СЛУЧАЙНЫЙ ЛЕС - ДЛИТЕЛЬНОСТЬ СЫПИ ")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
print(f"R²:  {r2_score(y_test, y_pred):.3f}")

# Кросс-валидация
cv_scores = cross_val_score(best_rf, X_scaled, y, cv=5, scoring='r2')
print(f"\nCV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# Базовое сравнение (предсказание среднего)
baseline_pred = np.full_like(y_test, y_test.mean())
baseline_r2 = r2_score(y_test, baseline_pred)
print(f"\nR² базовой модели (среднее): {baseline_r2:.3f}")
print(f"Улучшение над базовой: {(r2_score(y_test, y_pred) - baseline_r2):.3f}")

# Важность признаков
importance = pd.DataFrame({
    'Признак': X.columns,
    'Важность': best_rf.feature_importances_
}).sort_values('Важность', ascending=False)
print("\nВажность признаков:")
print(importance)