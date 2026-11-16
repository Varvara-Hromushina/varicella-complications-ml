import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

df = pd.read_excel("длительность.xlsx")
columns = ['duration_days','x1_day_of_rash', 'x2_locations', 'x3_polymorphism', 'x4_rash_abundance',
           'x5_pustulization', 'x6_rash_character', 'x7_crusts', 'x8_vesicle_content',
           'x9_palate_arches', 'x10_pharynx', 'x11_antiviral', 'x12_antibiotic',
           'x13_antiseptic']
df.columns = columns

print(f"Исходное количество строк: {len(df)}")
df = df.dropna()
print(f"После удаления пропусков: {len(df)}")

X = df.drop('duration_days', axis=1)
y = df['duration_days']

X_with_intercept = np.column_stack([np.ones(len(X)), X])
feature_names = ['Intercept'] + list(X.columns)

# Решение нормальных уравнений: β = (X^T * X)^(-1) * X^T * y
X_matrix = X_with_intercept
y_matrix = y.values.reshape(-1, 1)

# Вычисляем коэффициенты через МНК
beta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_matrix
beta = beta.flatten()

coefficients_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': beta,
    'Interpretation': ['Base level'] + [f'Change per 1 unit of {col}' for col in X.columns]
})

print("КОЭФФИЦИЕНТЫ РЕГРЕССИИ (МНК)")
print(coefficients_df.to_string(index=False))

y_pred = X_matrix @ beta.reshape(-1, 1)
y_pred = y_pred.flatten()

# Метрики качества
r2 = r2_score(y, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)

# Дополнительные метрики
y_mean = np.mean(y)
ss_tot = np.sum((y - y_mean) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
ss_reg = ss_tot - ss_res
f_stat = (ss_reg / X.shape[1]) / (ss_res / (len(y) - X.shape[1] - 1))
p_value_f = 1 - stats.f.cdf(f_stat, X.shape[1], len(y) - X.shape[1] - 1)

print("МЕТРИКИ КАЧЕСТВА МОДЕЛИ")
print(f"R² (коэффициент детерминации):       {r2:.4f}")
print(f"Скорректированный R²:                 {adj_r2:.4f}")
print(f"Среднеквадратичная ошибка (MSE):     {mse:.4f}")
print(f"Корень из MSE (RMSE):                 {rmse:.4f}")
print(f"Средняя абсолютная ошибка (MAE):     {mae:.4f}")
print(f"F-статистика:                         {f_stat:.4f}")
print(f"P-значение F-теста:                   {p_value_f:.4e}")

# Стандартные ошибки коэффициентов
residuals = y - y_pred
sse = np.sum(residuals ** 2)
mse_model = sse / (len(y) - X.shape[1] - 1)  # Несмещенная оценка дисперсии
var_cov_matrix = mse_model * np.linalg.inv(X_matrix.T @ X_matrix)
std_errors = np.sqrt(np.diag(var_cov_matrix))

# t-статистики и p-значения
t_stats = beta / std_errors
p_values = [2 * (1 - stats.t.cdf(np.abs(t), len(y) - X.shape[1] - 1)) for t in t_stats]

# Добавляем статистики в таблицу коэффициентов
coefficients_df['Std_Error'] = std_errors
coefficients_df['t_stat'] = t_stats
coefficients_df['p_value'] = p_values
coefficients_df['Significant'] = ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ' ' for p in
                                  p_values]


print("ПОЛНАЯ ТАБЛИЦА КОЭФФИЦИЕНТОВ СО СТАТИСТИКАМИ")
print(coefficients_df.round(4).to_string(index=False))


print("КОРРЕЛЯЦИЯ МЕЖДУ ПРЕДИКТОРАМИ")


correlation_matrix = X.corr()
print("\nКорреляционная матрица (13x13):")
print(correlation_matrix.round(3))

# Выводим самые сильные корреляции

print("ТОП-5 САМЫХ СИЛЬНЫХ КОРРЕЛЯЦИЙ")

corr_pairs = correlation_matrix.unstack().sort_values(ascending=False)
corr_pairs = corr_pairs[corr_pairs < 1]  # Убираем диагональ
for pair, value in corr_pairs.head(10).items():
    if pair[0] < pair[1]:  # Чтобы не дублировать
        print(f"{pair[0]} & {pair[1]}: {value:.3f}")

print("АНАЛИЗ ОСТАТКОВ")


# Тест на нормальность (Шапиро-Уилк)
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"Тест Шапиро-Уилка на нормальность остатков:")
print(f"  Статистика: {shapiro_stat:.4f}, P-значение: {shapiro_p:.4e}")
if shapiro_p > 0.05:
    print("  → Остатки распределены нормально (p > 0.05)")
else:
    print("  → Остатки не распределены нормально (p < 0.05)")

# Тест на гетероскедастичность (Бройша-Пэгана)
from sklearn.linear_model import LinearRegression

resid_sq = residuals ** 2
lr_bp = LinearRegression().fit(X, resid_sq)
bp_stat = len(y) * lr_bp.score(X, resid_sq)
bp_p = 1 - stats.chi2.cdf(bp_stat, X.shape[1])
print(f"\nТест Бройша-Пэгана на гетероскедастичность:")
print(f"  Статистика: {bp_stat:.4f}, P-значение: {bp_p:.4e}")
if bp_p > 0.05:
    print("  → Гетероскедастичности нет (p > 0.05)")
else:
    print("  → Гетероскедастичность присутствует (p < 0.05)")

from statsmodels.stats.outliers_influence import variance_inflation_factor


print("VIF (ФАКТОР ИНФЛЯЦИИ ДИСПЕРСИИ)")


vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# Для VIF добавляем столбец с оценкой
vif_data['Severity'] = vif_data['VIF'].apply(
    lambda x: 'High (VIF>10)' if x > 10 else
    'Moderate (5-10)' if x > 5 else
    'Low (<5)'
)
print(vif_data.to_string(index=False))


print("ПРИМЕР ПРЕДСКАЗАНИЯ")


# Создаем типичного пациента (медианные значения)
typical_patient = X.median().values.reshape(1, -1)
typical_with_intercept = np.column_stack([np.ones(1), typical_patient])
prediction = typical_with_intercept @ beta.reshape(-1, 1)
print(f"Для пациента со средними/медианными значениями предикторов:")
print(f"  Предсказанная длительность сыпи: {prediction[0][0]:.2f} дня")
print(f"  Фактическая средняя длительность: {np.mean(y):.2f} дня")


print("РЕЗЮМЕ МОДЕЛИ")


print(f"""
Модель множественной линейной регрессии
----------------------------------------
Количество наблюдений: {len(y)}
Количество предикторов: {X.shape[1]}
R²: {r2:.4f}
Скорректированный R²: {adj_r2:.4f}
RMSE: {rmse:.4f} дней
F-статистика: {f_stat:.2f} (p = {p_value_f:.4e})

Интерпретация:
- {100 * r2:.1f}% вариации длительности сыпи объясняется моделью
- Средняя ошибка предсказания: {mae:.2f} дня
- Статистически значимые предикторы (p < 0.05): 
  {', '.join(coefficients_df[coefficients_df['p_value'] < 0.05]['Feature'].tolist())}
""")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Предсказанные vs фактические значения
    axes[0, 0].scatter(y, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Фактическая длительность (дни)')
    axes[0, 0].set_ylabel('Предсказанная длительность (дни)')
    axes[0, 0].set_title(f'Предсказанные vs Фактические значения (R² = {r2:.3f})')

    # 2. Остатки
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Предсказанные значения')
    axes[0, 1].set_ylabel('Остатки')
    axes[0, 1].set_title('График остатков')

    # 3. Q-Q plot для проверки нормальности
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot остатков')

    # 4. Гистограмма остатков
    axes[1, 1].hist(residuals, bins=20, edgecolor='k', alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Остатки')
    axes[1, 1].set_ylabel('Частота')
    axes[1, 1].set_title('Распределение остатков')

    plt.tight_layout()
    plt.show()

except ImportError:
    print("\nДля визуализации установите matplotlib и seaborn: pip install matplotlib seaborn")


