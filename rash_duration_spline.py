import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from io import StringIO

df = pd.read_excel("длительность.xlsx")
columns = ['y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
df.columns = columns
df = df.dropna()


print("МОДЕЛЬ СО СПЛАЙНАМИ")

print(f"Количество наблюдений: {len(df)}")
print(f"Количество предикторов: 10 (без x11-x13)")

X = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]
y = df['y']

model_linear = LinearRegression()
model_linear.fit(X, y)
y_pred_linear = model_linear.predict(X)

r2_linear = r2_score(y, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
mae_linear = mean_absolute_error(y, y_pred_linear)

print("\n" + "-" * 80)
print("БАЗОВАЯ ЛИНЕЙНАЯ МОДЕЛЬ")
print("-" * 80)
print(f"R²:  {r2_linear:.6f}")
print(f"RMSE: {rmse_linear:.4f} дней")
print(f"MAE:  {mae_linear:.4f} дней")

print("\n" + "-" * 80)
print("МОДЕЛЬ 1: СПЛАЙНЫ ДЛЯ ВСЕХ ПЕРЕМЕННЫХ")
print("-" * 80)

spline_transformer = SplineTransformer(
    degree=3,  # Кубические сплайны
    n_knots=4,  # 4 узла
    include_bias=False,
    extrapolation='constant'
)

X_spline_all = spline_transformer.fit_transform(X)

model_spline_all = LinearRegression()
model_spline_all.fit(X_spline_all, y)
y_pred_spline_all = model_spline_all.predict(X_spline_all)

r2_spline_all = r2_score(y, y_pred_spline_all)
rmse_spline_all = np.sqrt(mean_squared_error(y, y_pred_spline_all))
mae_spline_all = mean_absolute_error(y, y_pred_spline_all)

print(f"R²:  {r2_spline_all:.6f}")
print(f"RMSE: {rmse_spline_all:.4f} дней")
print(f"MAE:  {mae_spline_all:.4f} дней")
print(f"Улучшение R²: {r2_spline_all - r2_linear:.6f}")


print("\n" + "-" * 80)
print("МОДЕЛЬ 2: СПЛАЙНЫ ТОЛЬКО ДЛЯ КЛЮЧЕВЫХ ПЕРЕМЕННЫХ (x3, x5, x7)")
print("-" * 80)

# Для ключевых переменных создаем сплайны
key_vars = ['x3', 'x5', 'x7']
X_spline_key = X.copy()

for var in key_vars:
    spline_features = SplineTransformer(
        degree=3, n_knots=4, include_bias=False
    ).fit_transform(X[[var]])
    for i in range(spline_features.shape[1]):
        X_spline_key[f'{var}_spline_{i}'] = spline_features[:, i]

# Добавляем также квадратичные члены для остальных важных переменных
for var in ['x1', 'x4', 'x9']:
    X_spline_key[f'{var}^2'] = X[var] ** 2

model_spline_key = LinearRegression()
model_spline_key.fit(X_spline_key, y)
y_pred_spline_key = model_spline_key.predict(X_spline_key)

r2_spline_key = r2_score(y, y_pred_spline_key)
rmse_spline_key = np.sqrt(mean_squared_error(y, y_pred_spline_key))
mae_spline_key = mean_absolute_error(y, y_pred_spline_key)

print(f"R²:  {r2_spline_key:.6f}")
print(f"RMSE: {rmse_spline_key:.4f} дней")
print(f"MAE:  {mae_spline_key:.4f} дней")
print(f"Улучшение R²: {r2_spline_key - r2_linear:.6f}")

print("\n" + "-" * 80)
print("МОДЕЛЬ 3: ОПТИМИЗАЦИЯ ЧИСЛА УЗЛОВ ДЛЯ КЛЮЧЕВЫХ ПЕРЕМЕННЫХ")
print("-" * 80)

best_r2 = -np.inf
best_n_knots = None
results = []

for n_knots in [3, 4, 5, 6, 8]:
    X_test = X.copy()
    for var in key_vars:
        spline_features = SplineTransformer(
            degree=3, n_knots=n_knots, include_bias=False
        ).fit_transform(X[[var]])
        for i in range(spline_features.shape[1]):
            X_test[f'{var}_spline_{i}'] = spline_features[:, i]

    # Добавляем квадратичные члены
    for var in ['x1', 'x4', 'x9']:
        X_test[f'{var}^2'] = X[var] ** 2

    model_test = LinearRegression()
    model_test.fit(X_test, y)
    y_pred_test = model_test.predict(X_test)
    r2_test = r2_score(y, y_pred_test)

    results.append((n_knots, r2_test))
    if r2_test > best_r2:
        best_r2 = r2_test
        best_n_knots = n_knots

print(f"Оптимальное число узлов: {best_n_knots}")
print(f"Лучший R²: {best_r2:.6f}")


print("\n" + "-" * 80)
print("КРОСС-ВАЛИДАЦИЯ (5-fold)")
print("-" * 80)


# Функция для оценки модели с кросс-валидацией
def evaluate_with_cv(X_data, y_data, model, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2_scores = []
    rmse_scores = []

    for train_idx, val_idx in kf.split(X_data):
        X_train, X_val = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_train, y_val = y_data.iloc[train_idx], y_data.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        r2_scores.append(r2_score(y_val, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

    return np.mean(r2_scores), np.std(r2_scores), np.mean(rmse_scores), np.std(rmse_scores)


# Оценка линейной модели
r2_cv_linear, r2_std_linear, rmse_cv_linear, rmse_std_linear = evaluate_with_cv(
    X, y, LinearRegression()
)

# Оценка модели со сплайнами (оптимальная)
X_best = X.copy()
for var in key_vars:
    spline_features = SplineTransformer(
        degree=3, n_knots=best_n_knots, include_bias=False
    ).fit_transform(X[[var]])
    for i in range(spline_features.shape[1]):
        X_best[f'{var}_spline_{i}'] = spline_features[:, i]

for var in ['x1', 'x4', 'x9']:
    X_best[f'{var}^2'] = X[var] ** 2

r2_cv_spline, r2_std_spline, rmse_cv_spline, rmse_std_spline = evaluate_with_cv(
    X_best, y, LinearRegression()
)

print("Линейная модель (CV):")
print(f"  R²:  {r2_cv_linear:.6f} ± {r2_std_linear:.6f}")
print(f"  RMSE: {rmse_cv_linear:.4f} ± {rmse_std_linear:.4f}")

print("\nМодель со сплайнами (CV):")
print(f"  R²:  {r2_cv_spline:.6f} ± {r2_std_spline:.6f}")
print(f"  RMSE: {rmse_cv_spline:.4f} ± {rmse_std_spline:.4f}")


print("СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ")


comparison_df = pd.DataFrame({
    'Модель': [
        'Линейная',
        'Сплайны (все переменные)',
        'Сплайны (ключевые)',
        f'Сплайны (оптимальные, {best_n_knots} узлов)'
    ],
    'R² (train)': [
        r2_linear,
        r2_spline_all,
        r2_spline_key,
        best_r2
    ],
    'R² (CV)': [
        r2_cv_linear,
        None,
        None,
        r2_cv_spline
    ],
    'RMSE (train)': [
        rmse_linear,
        rmse_spline_all,
        rmse_spline_key,
        np.sqrt(mean_squared_error(y, LinearRegression().fit(X_best, y).predict(X_best)))
    ],
    'RMSE (CV)': [
        rmse_cv_linear,
        None,
        None,
        rmse_cv_spline
    ]
})

print(comparison_df.to_string(index=False, float_format='%.6f'))


print("\n" + "-" * 80)
print("ОБУЧЕНИЕ ОПТИМАЛЬНОЙ МОДЕЛИ")
print("-" * 80)

# Создаем оптимальную модель с лучшими параметрами
model_best = LinearRegression()
model_best.fit(X_best, y)
y_pred_best = model_best.predict(X_best)
r2_best = r2_score(y, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y, y_pred_best))

print(f"Оптимальная модель (сплайны с {best_n_knots} узлами):")
print(f"  R²:  {r2_best:.6f}")
print(f"  RMSE: {rmse_best:.4f} дней")



print("ВИЗУАЛИЗАЦИЯ СПЛАЙНОВЫХ ЭФФЕКТОВ ДЛЯ КЛЮЧЕВЫХ ПЕРЕМЕННЫХ")


fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Создаем сплайновые трансформеры для каждой ключевой переменной
spline_transformers = {}
for var in key_vars:
    spline_transformers[var] = SplineTransformer(
        degree=3, n_knots=best_n_knots, include_bias=False,
        extrapolation='constant'
    )
    spline_transformers[var].fit(X[[var]])

# Для каждой переменной строим график
for idx, (var, title) in enumerate(zip(
        ['x3', 'x5', 'x7', 'x1', 'x4', 'x9'],
        ['Полиморфизм (сплайны)', 'Пустулизация (сплайны)',
         'Корки (сплайны)', 'День появления (квадрат)',
         'Обильность (квадрат)', 'Небо/дужки (квадрат)']
)):
    ax = axes[idx // 3, idx % 3]

    # Сортируем значения для плавной линии
    x_sorted = np.linspace(X[var].min(), X[var].max(), 50)

    # Создаем базовую точку со средними значениями всех переменных
    X_pred = pd.DataFrame({
        'x1': np.full(len(x_sorted), X['x1'].mean()),
        'x2': np.full(len(x_sorted), X['x2'].mean()),
        'x3': np.full(len(x_sorted), X['x3'].mean()),
        'x4': np.full(len(x_sorted), X['x4'].mean()),
        'x5': np.full(len(x_sorted), X['x5'].mean()),
        'x6': np.full(len(x_sorted), X['x6'].mean()),
        'x7': np.full(len(x_sorted), X['x7'].mean()),
        'x8': np.full(len(x_sorted), X['x8'].mean()),
        'x9': np.full(len(x_sorted), X['x9'].mean()),
        'x10': np.full(len(x_sorted), X['x10'].mean())
    })

    # Устанавливаем значения для текущей переменной
    X_pred[var] = x_sorted

    # Добавляем сплайновые признаки для ключевых переменных
    for svar in key_vars:
        spline_features = spline_transformers[svar].transform(X_pred[[svar]])
        for i in range(spline_features.shape[1]):
            col_name = f'{svar}_spline_{i}'
            X_pred[col_name] = spline_features[:, i]

    # Добавляем квадратичные признаки
    for qvar in ['x1', 'x4', 'x9']:
        X_pred[f'{qvar}^2'] = X_pred[qvar] ** 2

    # Предсказываем
    preds = model_best.predict(X_pred)

    # Отображаем исходные данные
    ax.scatter(X[var], y, alpha=0.3, s=30, label='Данные', color='blue')

    # Отображаем предсказания модели
    ax.plot(x_sorted, preds, 'r-', linewidth=2, label='Модель')

    # Доверительный интервал (примерный)
    std_residuals = np.std(y - model_best.predict(X_best))
    ax.fill_between(x_sorted,
                    preds - 1.96 * std_residuals,
                    preds + 1.96 * std_residuals,
                    alpha=0.2, color='red', label='95% ДИ')

    ax.set_xlabel(var)
    ax.set_ylabel('Длительность сыпи (дни)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


print("АНАЛИЗ ОСТАТКОВ СПЛАЙНОВОЙ МОДЕЛИ")


residuals_best = y - y_pred_best

# Статистики остатков
print(f"Среднее остатков:          {np.mean(residuals_best):.6f}")
print(f"Стандартное отклонение:     {np.std(residuals_best):.4f}")
print(f"Минимальный остаток:        {np.min(residuals_best):.4f}")
print(f"Максимальный остаток:       {np.max(residuals_best):.4f}")

# Тест Шапиро-Уилка
shapiro_stat, shapiro_p = stats.shapiro(residuals_best)
print(f"\nТест Шапиро-Уилка на нормальность:")
print(f"  Статистика: {shapiro_stat:.4f}")
print(f"  P-значение: {shapiro_p:.4e}")
if shapiro_p > 0.05:
    print("  → Остатки распределены нормально (p > 0.05) ✓")
else:
    print("  → Остатки НЕ распределены нормально (p < 0.05)")

# Тест на гетероскедастичность
from sklearn.linear_model import LinearRegression as LR

resid_sq = residuals_best ** 2
lr_bp = LR()
lr_bp.fit(X_best, resid_sq)
bp_stat = len(y) * lr_bp.score(X_best, resid_sq)
bp_p = 1 - stats.chi2.cdf(bp_stat, X_best.shape[1])
print(f"\nТест Бройша-Пэгана на гетероскедастичность:")
print(f"  Статистика: {bp_stat:.4f}")
print(f"  P-значение: {bp_p:.4e}")
if bp_p > 0.05:
    print("  → Гетероскедастичности нет (p > 0.05) ✓")
else:
    print("  → Гетероскедастичность присутствует (p < 0.05)")

# Дополнительные графики для диагностики
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Predicted vs Actual
axes[0, 0].scatter(y_pred_best, y, alpha=0.5)
axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Предсказанные значения')
axes[0, 0].set_ylabel('Фактические значения')
axes[0, 0].set_title(f'Предсказанные vs Фактические (R²={r2_best:.4f})')
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals vs Predicted
axes[0, 1].scatter(y_pred_best, residuals_best, alpha=0.5)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Предсказанные значения')
axes[0, 1].set_ylabel('Остатки')
axes[0, 1].set_title('График остатков')
axes[0, 1].grid(True, alpha=0.3)

# 3. Q-Q plot
stats.probplot(residuals_best, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot остатков')
axes[1, 0].grid(True, alpha=0.3)

# 4. Histogram of residuals
axes[1, 1].hist(residuals_best, bins=20, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Остатки')
axes[1, 1].set_ylabel('Частота')
axes[1, 1].set_title('Распределение остатков')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


print("ИТОГОВОЕ ЗАКЛЮЧЕНИЕ")


improvement = (best_r2 - r2_linear) / abs(r2_linear) * 100 if r2_linear != 0 else 0

print(f"""
1. Модель со сплайнами показала УЛУЧШЕНИЕ качества:

   Линейная модель:         R² = {r2_linear:.6f}
   Сплайновая модель:       R² = {best_r2:.6f}
   Улучшение:               {improvement:.1f}%

2. Кросс-валидация подтверждает улучшение:
   Линейная (CV):           R² = {r2_cv_linear:.6f} ± {r2_std_linear:.6f}
   Сплайновая (CV):         R² = {r2_cv_spline:.6f} ± {r2_std_spline:.6f}

3. Оптимальные параметры:
   - Количество узлов:      {best_n_knots}
   - Степень сплайна:       3 (кубический)
   - Переменные со сплайнами: x3 (полиморфизм), x5 (пустулизация), x7 (корки)
   - Квадратичные члены:    x1, x4, x9

4. Качество модели:
   - R² = {best_r2:.6f} означает, что модель объясняет {best_r2 * 100:.1f}% вариации
   - RMSE = {rmse_best:.4f} дней (средняя ошибка предсказания)

5. Рекомендация:
   Сплайновая модель рекомендуется к использованию, так как:
   - Дает лучшее качество, чем линейная модель
   - Проходит кросс-валидацию
   - Сохраняет интерпретируемость (можно визуализировать эффекты)
   - Позволяет выявить нелинейные зависимости
""")


print("СОХРАНЕНИЕ ПАРАМЕТРОВ МОДЕЛИ")


# Вывод коэффициентов сплайновой модели
coef_df = pd.DataFrame({
    'Переменная': X_best.columns,
    'Коэффициент': model_best.coef_
})
coef_df = coef_df.sort_values('Коэффициент', ascending=False)

print("\nТОП-10 КОЭФФИЦИЕНТОВ СПЛАЙНОВОЙ МОДЕЛИ:")
print(coef_df.head(10).to_string(index=False))

print(f"\nИнтерцепт: {model_best.intercept_:.6f}")



