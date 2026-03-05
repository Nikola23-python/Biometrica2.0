import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# =========================
# ЗАДАЧА 1
# =========================

strain_A = np.array([1.2, 1.5, 1.8, 1.1, 1.6, 1.7, 1.3, 1.5, 1.4, 1.8, 1.9, 1.6, 1.5, 1.4, 1.7])
strain_B = np.array([2.5, 2.8, 3.0, 2.6, 2.9, 3.2, 3.1, 2.7, 3.0, 2.9, 2.8, 3.1, 3.2, 2.6, 2.9])

results = []

results.append("===== ЗАДАЧА 1 =====\n")

# Средние и стандартные отклонения
mean_A = np.mean(strain_A)
mean_B = np.mean(strain_B)
std_A = np.std(strain_A, ddof=1)
std_B = np.std(strain_B, ddof=1)

results.append(f"Среднее A: {mean_A:.4f}")
results.append(f"Среднее B: {mean_B:.4f}")
results.append(f"Std A: {std_A:.4f}")
results.append(f"Std B: {std_B:.4f}\n")

# Проверка нормальности
shapiro_A = stats.shapiro(strain_A)
shapiro_B = stats.shapiro(strain_B)

results.append("Критерий Шапиро-Уилка:")
results.append(f"A: statistic={shapiro_A.statistic:.4f}, p={shapiro_A.pvalue:.6f}")
results.append(f"B: statistic={shapiro_B.statistic:.4f}, p={shapiro_B.pvalue:.6f}\n")

# t-тест
ttest = stats.ttest_ind(strain_A, strain_B, equal_var=True)
results.append("t-критерий Стьюдента:")
results.append(f"t={ttest.statistic:.4f}, p={ttest.pvalue:.6f}\n")

# 95% доверительный интервал
diff_mean = mean_A - mean_B
se = np.sqrt(std_A**2/len(strain_A) + std_B**2/len(strain_B))
df = len(strain_A) + len(strain_B) - 2
t_crit = stats.t.ppf(0.975, df)
ci_low = diff_mean - t_crit * se
ci_high = diff_mean + t_crit * se

results.append(f"95% ДИ разности средних: [{ci_low:.4f}, {ci_high:.4f}]\n")

# Манна–Уитни
mw = stats.mannwhitneyu(strain_A, strain_B, alternative='two-sided')
results.append("Критерий Манна–Уитни:")
results.append(f"U={mw.statistic:.4f}, p={mw.pvalue:.6f}\n")

# =========================
# ЗАДАЧА 3
# =========================

results.append("\n===== ЗАДАЧА 3 =====\n")

ph = np.array([6.0, 6.2, 6.5, 6.8, 7.0, 7.2, 7.5, 7.8, 8.0, 8.2])
lum = np.array([100, 120, 150, 180, 200, 210, 190, 170, 140, 110])

# Корреляция Пирсона
pearson = stats.pearsonr(ph, lum)
results.append("Корреляция Пирсона:")
results.append(f"r={pearson.statistic:.4f}, p={pearson.pvalue:.6f}\n")

# Построение графика
plt.figure()
plt.scatter(ph, lum)
plt.xlabel("pH")
plt.ylabel("Биолюминесценция")
plt.title("Зависимость биолюминесценции от pH")
plt.savefig("task3_scatter.png")
plt.close()

results.append("Диаграмма рассеяния сохранена как task3_scatter.png\n")

# =========================
# Сохранение результатов
# =========================

with open("results_tasks_1_3.txt", "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n")

print("Готово. Результаты сохранены в results_tasks_1_3.txt")