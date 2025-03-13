import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "132-serednia-nominalna-zarobitna-plata-za-misiats-grn.csv"
df = pd.read_csv(path, delimiter=";", encoding="1251")
rates = df[["period", "data"]]
rates["period"] = pd.to_datetime(rates["period"], dayfirst= True)
rates["data"] = pd.to_numeric(rates["data"], errors="coerce")
rates = rates.sort_values("period")

aggregated_df = rates.groupby("period", as_index=False).agg({"data": "sum"})

x = (aggregated_df["period"] - aggregated_df["period"].min()).dt.days

coeffs = np.polyfit(x, rates["data"], 3)
trend = np.polyval(coeffs, x)
trend_equation = f"y = {coeffs[0]:.5f}x^2 + {coeffs[1]:.5f}x + {coeffs[2]:.5f}"
stohastic = rates["data"] - trend

# 4. Статистичні характеристики
variance = np.var(stohastic)
std_dev = np.std(stohastic)
mS = np.median(stohastic)
mean_value = np.mean(stohastic)

print('-------- статистичні характеристики ХІ КВАДРАТ закону розподілу ВВ ---------')
print(f"математичне сподівання (медіана): {mS}")
print(f"Математичне очікування (середня): {mean_value}")
print(f"Дисперсія: {variance}")
print(f"Середньоквадратичне відхилення: {std_dev}")
print(f"Рівняння квадратичного тренду: {trend_equation}")
print('----------------------------------------------------------------------------')

# 5. Візуалізація
df_plot = aggregated_df.copy()
df_plot["Тренд"] = trend

trend_plot = trend.copy()

plt.figure(figsize=(12, 6))
plt.plot(df_plot["period"], df_plot["data"], label="Реальні дані")
plt.plot(df_plot["period"], df_plot["Тренд"], label="Тренд")

plt.legend()
plt.xlabel("Дата")
plt.ylabel("Середня заробітна плата")
plt.title("Моделювання")
plt.grid()
plt.show()

# Гістограма випадкової складової
plt.figure(figsize=(8, 4))
plt.hist(stohastic, bins=20, density=True, alpha=0.6, color='g')
plt.title("Гістограма випадкової складової")
plt.xlabel("Значення")
plt.ylabel("Щільність")
plt.grid()
plt.show()
