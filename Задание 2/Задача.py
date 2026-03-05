import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
np.random.seed(40)

def generate_microbiology_data(n_samples=50):
    n1 = 50
    mean1 = 14.0
    std1 = 1.8

    n2 = 30
    mean2 = 22.0
    std2 = 1.8

    sample1 = np.random.normal(mean1, std1, n1)
    sample2 = np.random.normal(mean2, std2, n2)
    combined = np.concatenate([sample1, sample2])
    combined = np.round(combined).astype(int)
    combined = np.clip(combined, 5, 35)
    np.random.shuffle(combined)

    return combined

data = generate_microbiology_data(50)

print("=" * 60)
print("ЗАДАНИЕ: Анализ одной выборки")
print("=" * 60)

data_sorted = np.sort(data)
print(f"\nДанные для анализа (диаметр зон задержки роста, мм):")
for i in range(0, len(data_sorted), 10):
    print(f"  {i+1:2d}-{min(i+10, len(data_sorted)):2d}: {data_sorted[i:i+10]}")
print(f"\nКоличество измерений (n): {len(data)}")

mean_val = np.mean(data)
median_val = np.median(data)
std_val = np.std(data, ddof=1)
se_val = std_val / np.sqrt(len(data))

confidence_level = 0.95
degrees_freedom = len(data) - 1
t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
margin_of_error = t_critical * se_val
ci_lower = mean_val - margin_of_error
ci_upper = mean_val + margin_of_error

print(f"Сумма всех значений: {np.sum(data)} мм")
print(f"Среднее арифметическое: {mean_val:.2f} мм")
print(f"Медиана: {median_val:.2f} мм")
print(f"Стандартное отклонение (выборочное): {std_val:.2f} мм")
print(f"Стандартная ошибка среднего: {se_val:.3f} мм")
print(f"95% Доверительный интервал для среднего: ({ci_lower:.2f}, {ci_upper:.2f}) мм")
np.savetxt('ДАННЫЕ ДЛЯ АНАЛИЗА.csv', data, fmt='%d', delimiter=',', header='Diameter_mm', comments='')

#Проверка гипотезы
shapiro_stat, shapiro_p = stats.shapiro(data)

print("\n" + "="*60)
print("ПРОВЕРКА НОРМАЛЬНОСТИ РАСПРЕДЕЛЕНИЯ")
print("="*60)
print(f"Критерий Шапиро-Уилка:")
print(f"  W-статистика = {shapiro_stat:.4f}")
print(f"  p-значение   = {shapiro_p:.4f}")

alpha = 0.05
if shapiro_p > alpha:
    print("  Вывод: p > 0.05 => НЕТ оснований отвергнуть H0.")
    print("  Данные не противоречат нормальному распределению.")
else:
    print("  Вывод: p <= 0.05 => H0 отвергается.")
    print("  Распределение значимо отличается от нормального.")

#Гистограмма
plt.figure(figsize=(14, 8))

counts, bins, patches = plt.hist(data, bins=15, density=False, alpha=0.7,
                                 color='steelblue', edgecolor='black',)

x = np.linspace(min(data), max(data), 200)
bin_width = bins[1] - bins[0]
scaling_factor = len(data) * bin_width
pdf_scaled = stats.norm.pdf(x, mean_val, std_val) * scaling_factor

plt.plot(x, pdf_scaled, 'r-', lw=3, label=f'Нормальная кривая (μ={mean_val:.2f}, σ={std_val:.2f})')

plt.axvline(mean_val, color='darkblue', linestyle='--', lw=2, label=f'Среднее = {mean_val:.2f} мм')
plt.axvline(median_val, color='green', linestyle='--', lw=2, label=f'Медиана = {median_val:.2f} мм')

textstr = f'Критерий Шапиро-Уилка:\nW = {shapiro_stat:.4f}\np = {shapiro_p:.4f}'
if shapiro_p > 0.05:
    textstr += '\nВывод: p > 0.05\nРаспределение нормальное'
else:
    textstr += '\nВывод: p < 0.05\nРаспределение НЕ нормальное'

# Размещаем текст в левом верхнем углу графика
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.title('Распределение диаметров зон лизиса', fontsize=16, fontweight='bold')
plt.xlabel('Диаметр зоны (мм)', fontsize=14)
plt.ylabel('Частота', fontsize=14)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')

# Добавим подписи значений над столбцами для наглядности
for i, (count, bin_edge) in enumerate(zip(counts, bins[:-1])):
    if count > 0:  # Только для ненулевых столбцов
        plt.text(bin_edge + bin_width/2, count + 0.5, str(int(count)),
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('гистограмма_частот.png', dpi=150, bbox_inches='tight')
plt.show()
