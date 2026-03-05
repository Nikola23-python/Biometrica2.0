import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, shapiro, mannwhitneyu, pearsonr, spearmanr
from scipy.stats import t as t_dist
import os
from datetime import datetime

# Создаём папку для результатов
output_dir = "Результаты_задач_1_и_3"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Файл для текстовых результатов
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_file = os.path.join(output_dir, f"результаты_{timestamp}.txt")

with open(result_file, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("СТАТИСТИЧЕСКИЙ АНАЛИЗ ДАННЫХ\n")
    f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")

    # ============= ЗАДАЧА 1 =============
    f.write("ЗАДАЧА 1: Анализ устойчивости бактерий к антибиотикам\n")
    f.write("-" * 60 + "\n\n")

    # Данные
    strain_A = [1.2, 1.5, 1.8, 1.1, 1.6, 1.7, 1.3, 1.5, 1.4, 1.8, 1.9, 1.6, 1.5, 1.4, 1.7]
    strain_B = [2.5, 2.8, 3.0, 2.6, 2.9, 3.2, 3.1, 2.7, 3.0, 2.9, 2.8, 3.1, 3.2, 2.6, 2.9]

    # Описательные статистики
    f.write("Исходные данные:\n")
    f.write(f"Штамм A: {strain_A}\n")
    f.write(f"Штамм B: {strain_B}\n\n")

    # 1. t-критерий Стьюдента
    t_stat, p_value = ttest_ind(strain_A, strain_B, equal_var=False)
    f.write("1. Проверка гипотезы о равенстве средних (t-критерий Стьюдента):\n")
    f.write(f"   t-статистика = {t_stat:.4f}\n")
    f.write(f"   p-значение = {p_value:.4f}\n")
    f.write(f"   Уровень значимости α = 0.05\n")
    if p_value < 0.05:
        f.write("   → Отвергаем H0. Средние значения МПК двух штаммов статистически значимо различаются.\n\n")
    else:
        f.write("   → H0 не отвергается. Средние значения МПК двух штаммов статистически значимо не различаются.\n\n")

    # 2. 95% доверительный интервал для разности средних
    mean_A, mean_B = np.mean(strain_A), np.mean(strain_B)
    std_A, std_B = np.std(strain_A, ddof=1), np.std(strain_B, ddof=1)
    n_A, n_B = len(strain_A), len(strain_B)

    se_diff = np.sqrt(std_A ** 2 / n_A + std_B ** 2 / n_B)
    df = (std_A ** 2 / n_A + std_B ** 2 / n_B) ** 2 / (
            (std_A ** 2 / n_A) ** 2 / (n_A - 1) + (std_B ** 2 / n_B) ** 2 / (n_B - 1)
    )

    t_crit = t_dist.ppf(0.975, df)
    mean_diff = mean_A - mean_B
    ci_low = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff

    f.write("2. 95% доверительный интервал для разности средних:\n")
    f.write(f"   Среднее штамма A = {mean_A:.3f} мкг/мл\n")
    f.write(f"   Среднее штамма B = {mean_B:.3f} мкг/мл\n")
    f.write(f"   Разность средних (A - B) = {mean_diff:.3f} мкг/мл\n")
    f.write(f"   95% ДИ: [{ci_low:.3f}, {ci_high:.3f}] мкг/мл\n\n")

    # 3. Проверка нормальности (Шапиро-Уилк)
    shap_A = shapiro(strain_A)
    shap_B = shapiro(strain_B)

    f.write("3. Проверка данных на нормальность (критерий Шапиро-Уилка):\n")
    f.write(f"   Штамм A: W={shap_A.statistic:.4f}, p={shap_A.pvalue:.4f}\n")
    f.write(f"   Штамм B: W={shap_B.statistic:.4f}, p={shap_B.pvalue:.4f}\n")

    if shap_A.pvalue > 0.05 and shap_B.pvalue > 0.05:
        f.write("   → Данные обеих групп подчиняются нормальному распределению (p > 0.05).\n")
        f.write("   → Условия для t-критерия выполнены.\n\n")
    else:
        f.write("   → Нарушение условия нормальности.\n")
        f.write("   → Используем непараметрический критерий Манна-Уитни:\n")
        u_stat, p_u = mannwhitneyu(strain_A, strain_B, alternative='two-sided')
        f.write(f"      U = {u_stat:.1f}, p = {p_u:.4f}\n")
        if p_u < 0.05:
            f.write("      → Различия статистически значимы.\n\n")
        else:
            f.write("      → Различия статистически не значимы.\n\n")

    f.write("\n" + "=" * 60 + "\n\n")

    # ============= ЗАДАЧА 3 =============
    f.write("ЗАДАЧА 3: Корреляция между pH и биолюминесценцией бактерий\n")
    f.write("-" * 60 + "\n\n")

    # Данные
    ph = np.array([6.0, 6.2, 6.5, 6.8, 7.0, 7.2, 7.5, 7.8, 8.0, 8.2])
    lum = np.array([100, 120, 150, 180, 200, 210, 190, 170, 140, 110])

    f.write("Исходные данные:\n")
    f.write("pH: " + " ".join(f"{x:4.1f}" for x in ph) + "\n")
    f.write("Lum: " + " ".join(f"{x:4d}" for x in lum) + "\n\n")

    # 1. Коэффициент корреляции Пирсона
    r_pearson, p_pearson = pearsonr(ph, lum)

    f.write("1. Коэффициент корреляции Пирсона:\n")
    f.write(f"   r = {r_pearson:.4f}\n")
    f.write(f"   p-значение = {p_pearson:.4f}\n")
    f.write(f"   Уровень значимости α = 0.05\n")
    if p_pearson < 0.05:
        f.write(f"   → Корреляция статистически значима.\n")
    else:
        f.write(f"   → Корреляция статистически не значима.\n\n")


    f.write("\n" + "=" * 60 + "\n")
    f.write(f"АНАЛИЗ ЗАВЕРШЕН. Время сохранения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n")

print(f"✅ Текстовый отчёт сохранён: {result_file}")

# ============= ГРАФИКИ ТОЛЬКО ПО ЗАДАНИЯМ =============

# Задача 3: Диаграмма рассеяния (обязательно по заданию)
plt.figure(figsize=(8, 6))
plt.scatter(ph, lum, s=80, c='blue', alpha=0.7)
plt.title('Зависимость биолюминесценции от pH среды', fontsize=14)
plt.xlabel('pH среды', fontsize=12)
plt.ylabel('Биолюминесценция (усл. ед.)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'задача3_диаграмма_рассеяния.png'), dpi=300)
plt.close()
print(f"✅ График сохранён: задача3_диаграмма_рассеяния.png")

print(f"\n📁 Все результаты в папке: {output_dir}/")