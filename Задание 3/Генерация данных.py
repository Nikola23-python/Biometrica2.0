import numpy as np
import pandas as pd

np.random.seed(42)

n = 30

light_clover = np.random.normal(loc=1200, scale=250, size=n).clip(500, 2000)

mean_clover = [45.0, 3.5]
cov_clover = [[100.0, 0.95 * 10 * 1.2],[0.95 * 10 * 1.2, 1.44]]
data_clover = np.random.multivariate_normal(mean_clover, cov_clover, n)
area_clover = data_clover[:, 0].clip(20, 80)
mass_clover = data_clover[:, 1].clip(1.0, 8.0)

flowers_clover = (light_clover - 500) * 0.006 + 1.5
flowers_clover = np.random.poisson(lam=flowers_clover)
flowers_clover = flowers_clover.clip(0, 10).astype(int)

light_oduvanc = np.random.normal(loc=1150, scale=280, size=n).clip(500, 2000)

mean_oduvanc = [40.0, 3.0]
cov_oduvanc = [[90.0, 0.85 * 9.5 * 1.1],[0.85 * 9.5 * 1.1, 1.21]]
data_oduvanc = np.random.multivariate_normal(mean_oduvanc, cov_oduvanc, n)
area_oduvanc = data_oduvanc[:, 0].clip(20, 80)
mass_oduvanc = data_oduvanc[:, 1].clip(1.0, 7.0)
flowers_oduvanc = 2.5 * np.log(light_oduvanc / 400) + 1
flowers_oduvanc = np.random.poisson(lam=flowers_oduvanc)
flowers_oduvanc = flowers_oduvanc.clip(0, 8).astype(int)
df_clover = pd.DataFrame({
'Освещенность (лк)': light_clover.round(),
'Площадь листьев (см²)': area_clover.round(1),
'Масса (г)': mass_clover.round(2),
'Число цветков': flowers_clover,
'Вид': 'Клевер'
})

df_oduvanc = pd.DataFrame({
'Освещенность (лк)': light_oduvanc.round(),
'Площадь листьев (см²)': area_oduvanc.round(1),
'Масса (г)': mass_oduvanc.round(2),
'Число цветков': flowers_oduvanc,
'Вид': 'Одуванчик'
})

df = pd.concat([df_clover, df_oduvanc], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('Данные по клеверу и одуванчику.csv',
 sep=';',
 encoding='utf-8-sig',
 index=False)
print(df.head(10))
