import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

results = []

# =========================
# ЗАДАЧА 2
# =========================

results.append("===== ЗАДАЧА 2 =====\n")

t25 = np.array([0.45, 0.48, 0.42, 0.47, 0.44, 0.46, 0.43, 0.45, 0.44, 0.46])
t37 = np.array([0.78, 0.82, 0.79, 0.81, 0.77, 0.80, 0.79, 0.78, 0.81, 0.80])
t42 = np.array([0.52, 0.55, 0.51, 0.54, 0.53, 0.56, 0.52, 0.51, 0.55, 0.54])

# Средние
results.append(f"Среднее 25°C: {np.mean(t25):.4f}")
results.append(f"Среднее 37°C: {np.mean(t37):.4f}")
results.append(f"Среднее 42°C: {np.mean(t42):.4f}\n")

# Левен
levene = stats.levene(t25, t37, t42)
results.append("Критерий Левена:")
results.append(f"statistic={levene.statistic:.4f}, p={levene.pvalue:.6f}\n")

# ANOVA
anova = stats.f_oneway(t25, t37, t42)
results.append("ANOVA:")
results.append(f"F={anova.statistic:.4f}, p={anova.pvalue:.6f}\n")

# Тьюки
data = np.concatenate([t25, t37, t42])
groups = (["25°C"]*10) + (["37°C"]*10) + (["42°C"]*10)

tukey = pairwise_tukeyhsd(data, groups)

results.append("Пост-хок анализ Тьюки:\n")
results.append(str(tukey))
results.append("\n")

# =========================
# ЗАДАЧА 4
# =========================

results.append("\n===== ЗАДАЧА 4 =====\n")

glucose = np.array([0.5,1,2,3,4,5,6,7,8,9])
density = np.array([2.1e6,3.5e6,6.0e6,7.8e6,8.5e6,8.9e6,9.0e6,8.8e6,8.3e6,7.0e6])

# Спирмен
spearman = stats.spearmanr(glucose, density)
results.append("Корреляция Спирмена:")
results.append(f"rho={spearman.statistic:.4f}, p={spearman.pvalue:.6f}\n")

# Максимум
max_index = np.argmax(density)
results.append(f"Максимальная плотность: {density[max_index]:.2e}")
results.append(f"При концентрации глюкозы: {glucose[max_index]} мг/мл\n")

# =========================
# Сохранение
# =========================

with open("results_tasks_2_4.txt", "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n")

print("Готово. Результаты сохранены в results_tasks_2_4.txt")