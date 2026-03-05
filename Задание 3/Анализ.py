import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import os

file_path = os.path.expanduser("/home/nikolay/PycharmProjects/Biometrica2.0/Задание 3/Данные по клеверу и одуванчику.csv")
df = pd.read_csv(file_path, sep=';', decimal='.')

clover = df[df['Вид'] == 'Клевер'].copy()
oduv = df[df['Вид'] == 'Одуванчик'].copy()

clover.columns = ['Освещенность', 'Площадь', 'Масса', 'Цветки', 'Вид']
oduv.columns = ['Освещенность', 'Площадь', 'Масса', 'Цветки', 'Вид']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
r_clover_pl_mass = np.corrcoef(clover['Площадь'], clover['Масса'])[0, 1]
ax.scatter(clover['Площадь'], clover['Масса'], color='violet', alpha=0.7)
z = np.polyfit(clover['Площадь'], clover['Масса'], 1)
p = np.poly1d(z)
ax.plot(clover['Площадь'], p(clover['Площадь']), "r-", linewidth=2)
ax.set_title(f"Клевер: Площадь vs Масса (r = {r_clover_pl_mass:.3f})")
ax.set_xlabel("Площадь листьев (см²)")
ax.set_ylabel("Масса (г)")

ax = axes[0, 1]
r_clover_light_flowers = np.corrcoef(clover['Освещенность'], clover['Цветки'])[0, 1]
ax.scatter(clover['Освещенность'], clover['Цветки'], color='violet', alpha=0.7)
z = np.polyfit(clover['Освещенность'], clover['Цветки'], 1)
p = np.poly1d(z)
ax.plot(clover['Освещенность'], p(clover['Освещенность']), "r-", linewidth=2)
ax.set_title(f"Клевер: Освещенность vs Цветки (r = {r_clover_light_flowers:.3f})")
ax.set_xlabel("Освещенность (лк)")
ax.set_ylabel("Число цветков")

ax = axes[1, 0]
r_oduv_pl_mass = np.corrcoef(oduv['Площадь'], oduv['Масса'])[0, 1]
ax.scatter(oduv['Площадь'], oduv['Масса'], color='yellow', alpha=0.7)
z = np.polyfit(oduv['Площадь'], oduv['Масса'], 1)
p = np.poly1d(z)
ax.plot(oduv['Площадь'], p(oduv['Площадь']), "r-", linewidth=2)
ax.set_title(f"Одуванчик: Площадь vs Масса (r = {r_oduv_pl_mass:.3f})")
ax.set_xlabel("Площадь листьев (см²)")
ax.set_ylabel("Масса (г)")

ax = axes[1, 1]
r_oduv_light_flowers = np.corrcoef(oduv['Освещенность'], oduv['Цветки'])[0, 1]
ax.scatter(oduv['Освещенность'], oduv['Цветки'], color='yellow', alpha=0.7)
z = np.polyfit(oduv['Освещенность'], oduv['Цветки'], 1)
p = np.poly1d(z)
ax.plot(oduv['Освещенность'], p(oduv['Освещенность']), "r-", linewidth=2)
ax.set_title(f"Одуванчик: Освещенность vs Цветки (r = {r_oduv_light_flowers:.3f})")
ax.set_xlabel("Освещенность (лк)")
ax.set_ylabel("Число цветков")

plt.tight_layout()
plt.savefig(os.path.expanduser("Графики.png"), dpi=120)
plt.close()

with open(os.path.expanduser("Результаты_анализа.txt"), "w", encoding="utf-8") as f:
    f.write(f"Дата анализа: {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}\n\n")
    f.write("1. КОРРЕЛЯЦИОННЫЕ МАТРИЦЫ ПИРСОНА (3×3)\n")
    f.write("--- КЛЕВЕР ---\n")
    f.write("Матрица корреляций: Площадь, Масса, Число цветков\n\n")

    clover_matrix = np.zeros((3, 3))
    cols = ['Площадь', 'Масса', 'Цветки']
    for i in range(3):
        for j in range(3):
            clover_matrix[i, j] = np.corrcoef(clover[cols[i]], clover[cols[j]])[0, 1]

    f.write("        Площадь  Масса  Цветки\n")
    for i, row_name in enumerate(cols):
        f.write(f"{row_name:8} {clover_matrix[i, 0]:.3f}  {clover_matrix[i, 1]:.3f}  {clover_matrix[i, 2]:.3f}\n")
    f.write("\n")

    f.write("--- ОДУВАНЧИК ---\n")
    f.write("Матрица корреляций: Площадь, Масса, Число цветков\n\n")

    oduv_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            oduv_matrix[i, j] = np.corrcoef(oduv[cols[i]], oduv[cols[j]])[0, 1]

    f.write("        Площадь  Масса  Цветки\n")
    for i, row_name in enumerate(cols):
        f.write(f"{row_name:8} {oduv_matrix[i, 0]:.3f}  {oduv_matrix[i, 1]:.3f}  {oduv_matrix[i, 2]:.3f}\n")
    f.write("\n")

    r1 = clover_matrix[0, 1]  # Клевер Площадь-Масса
    r2 = clover_matrix[0, 2]  # Клевер Площадь-Цветки
    r3 = clover_matrix[1, 2]  # Клевер Масса-Цветки
    r4 = oduv_matrix[0, 1]  # Одуванчик Площадь-Масса
    r5 = oduv_matrix[0, 2]  # Одуванчик Площадь-Цветки
    r6 = oduv_matrix[1, 2]  # Одуванчик Масса-Цветки

    f.write("2. ПРОВЕРКА СТАТИСТИЧЕСКОЙ ЗНАЧИМОСТИ\n")
    critical_r = 0.36
    f.write(f"Критическое значение r при α=0.05 и df=28: {critical_r}\n")
    f.write(f"Если |r| > {critical_r} → связь статистически значима\n\n")

    f.write("--- КЛЕВЕР ---\n")
    f.write(
        f"Площадь-Масса: | {abs(r1):.3f} | > {critical_r} →  {'ЗНАЧИМО' if abs(r1) > critical_r else 'НЕ ЗНАЧИМО'}\n")
    f.write(
        f"Площадь-Цветки: | {abs(r2):.3f} | > {critical_r} →  {'ЗНАЧИМО' if abs(r2) > critical_r else 'НЕ ЗНАЧИМО'}\n")
    f.write(
        f"Масса-Цветки: | {abs(r3):.3f} | > {critical_r} →  {'ЗНАЧИМО' if abs(r3) > critical_r else 'НЕ ЗНАЧИМО'}\n\n")

    f.write("--- ОДУВАНЧИК ---\n")
    f.write(
        f"Площадь-Масса: | {abs(r4):.3f} | > {critical_r} →  {'ЗНАЧИМО' if abs(r4) > critical_r else 'НЕ ЗНАЧИМО'}\n")
    f.write(
        f"Площадь-Цветки: | {abs(r5):.3f} | > {critical_r} →  {'ЗНАЧИМО' if abs(r5) > critical_r else 'НЕ ЗНАЧИМО'}\n")
    f.write(
        f"Масса-Цветки: | {abs(r6):.3f} | > {critical_r} →  {'ЗНАЧИМО' if abs(r6) > critical_r else 'НЕ ЗНАЧИМО'}\n\n")

    f.write("3. СРАВНЕНИЕ ПИРСОНА И СПИРМЕНА\n")
    f.write("   (Освещенность vs Число цветков)\n")

    def spearman_cor(x, y):
        n = len(x)
        rx = stats.rankdata(x)
        ry = stats.rankdata(y)
        d = rx - ry
        rho = 1 - (6 * sum(d ** 2)) / (n * (n ** 2 - 1))
        return rho

    r_pearson_clover = np.corrcoef(clover['Освещенность'], clover['Цветки'])[0, 1]
    r_pearson_oduv = np.corrcoef(oduv['Освещенность'], oduv['Цветки'])[0, 1]

    r_spearman_clover = spearman_cor(clover['Освещенность'], clover['Цветки'])
    r_spearman_oduv = spearman_cor(oduv['Освещенность'], oduv['Цветки'])

    f.write("--- КЛЕВЕР ---\n")
    f.write(f"Пирсон (r):   {r_pearson_clover:.3f}\n")
    f.write(f"Спирмен (ρ):  {r_spearman_clover:.3f}\n")
    f.write(f"Разница:      {abs(r_pearson_clover - r_spearman_clover):.3f}\n\n")

    f.write("--- ОДУВАНЧИК ---\n")
    f.write(f"Пирсон (r):   {r_pearson_oduv:.3f}\n")
    f.write(f"Спирмен (ρ):  {r_spearman_oduv:.3f}\n")
    f.write(f"Разница:      {abs(r_pearson_oduv - r_spearman_oduv):.3f}\n\n")

    f.write("Почему Спирмен лучше для одуванчика?\n")
    f.write("• Спирмен учитывает монотонную, а не только линейную связь\n")
    f.write("• Менее чувствителен к выбросам\n")
    f.write("• Лучше подходит для дискретных данных (число цветков)\n\n")

    f.write("4. ШКАЛА ЧЕДДОКА\n")
    f.write("0.1 - 0.3: слабая связь\n")
    f.write("0.3 - 0.5: умеренная связь\n")
    f.write("0.5 - 0.7: заметная связь\n")
    f.write("0.7 - 0.9: высокая связь\n")
    f.write("0.9 - 1.0: весьма высокая связь\n\n")

    f.write("5. ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ\n")


    def interpret(r):
        r = abs(r)
        if r < 0.1:
            return "нет связи"
        elif r < 0.3:
            return "слабая"
        elif r < 0.5:
            return "умеренная"
        elif r < 0.7:
            return "заметная"
        elif r < 0.9:
            return "высокая"
        else:
            return "весьма высокая"


    f.write("КЛЕВЕР:\n")
    f.write(f"Площадь-Масса:   {r1:.3f} - {interpret(r1)}\n")
    f.write(f"Площадь-Цветки:  {r2:.3f} - {interpret(r2)}\n")
    f.write(f"Масса-Цветки:    {r3:.3f} - {interpret(r3)}\n\n")

    f.write("ОДУВАНЧИК:\n")
    f.write(f"Площадь-Масса:   {r4:.3f} - {interpret(r4)}\n")
    f.write(f"Площадь-Цветки:  {r5:.3f} - {interpret(r5)}\n")
    f.write(f"Масса-Цветки:    {r6:.3f} - {interpret(r6)}\n\n")

    f.write("6. ПРОВЕРКА ГИПОТЕЗЫ ИССЛЕДОВАНИЯ\n")
    f.write("Гипотеза: у клевера (симбиоз с азотфиксирующими бактериями)\n")
    f.write("связь между площадью листьев и массой будет иной, чем у одуванчика\n\n")

    f.write("Результат:\n")
    f.write(f"Клевер: площадь-масса = {r1:.3f} ( {interpret(r1)} )\n")
    f.write(f"Одуванчик: площадь-масса = {r4:.3f} ( {interpret(r4)} )\n\n")

    if abs(r1) > abs(r4):
        f.write(f"✓ Гипотеза подтверждается: у клевера связь теснее\n")
        f.write(f"  ( {r1:.3f} против {r4:.3f} )\n")
    else:
        f.write(f"✗ Гипотеза не подтверждается: у одуванчика связь теснее или одинакова\n")

