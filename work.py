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

coeffs = np.polyfit(x, aggregated_df["data"], 3)
trend = np.polyval(coeffs, x)
trend_equation = f"y = {coeffs[0]:.5f}x^2 + {coeffs[1]:.5f}x + {coeffs[2]:.5f}"
stohastic = aggregated_df["data"] - trend

# Статистичні характеристики стохастичної складової
mean_value = np.mean(stohastic)
variance = np.var(stohastic)
std_dev = np.std(stohastic)
mS = np.median(stohastic)

# Міжквартильний розмах (IQR) для виявлення аномалій
q1 = np.percentile(stohastic, 25)
q3 = np.percentile(stohastic, 75)
iqr = q3 - q1
lower_bound_iqr = q1 - 1.5 * iqr  # Нижня межа для нормальних значень
upper_bound_iqr = q3 + 1.5 * iqr  # Верхня межа для нормальних значень

# Правило трьох сигм
lower_bound_3sigma = mean_value - 3 * std_dev
upper_bound_3sigma = mean_value + 3 * std_dev

print('-------- Статистичні характеристики стохастичної складової ---------')
print(f"Математичне сподівання (середня): {mean_value:.2f}")
print(f"Математичне сподівання (медіана): {mS:.2f}")
print(f"Дисперсія: {variance:.2f}")
print(f"Середньоквадратичне відхилення (σ): {std_dev:.2f}")
print(f"Ширина розподілу (±1σ): [{mean_value - std_dev:.2f}, {mean_value + std_dev:.2f}]")
print(f"Діапазон нормальних значень (3σ): [{lower_bound_3sigma:.2f}, {upper_bound_3sigma:.2f}]")
print(f"Діапазон нормальних значень (IQR): [{lower_bound_iqr:.2f}, {upper_bound_iqr:.2f}]")
print(f"Рівняння кубічного тренду: {trend_equation}")
print('-----------------------------------------------------------------')

# Виявлення аномалій
anomalies_3sigma = stohastic[(stohastic < lower_bound_3sigma) | (stohastic > upper_bound_3sigma)]
anomalies_iqr = stohastic[(stohastic < lower_bound_iqr) | (stohastic > upper_bound_iqr)]

print(f"Кількість аномалій (правило 3σ): {len(anomalies_3sigma)}")
print(f"Аномальні значення (3σ): {anomalies_3sigma.values}")
print(f"Кількість аномалій (IQR): {len(anomalies_iqr)}")
print(f"Аномальні значення (IQR): {anomalies_iqr.values}")

# Візуалізація
df_plot = aggregated_df.copy()
df_plot["Тренд"] = trend

plt.figure(figsize=(12, 6))
plt.plot(df_plot["period"], df_plot["data"], label="Реальні дані")
plt.plot(df_plot["period"], df_plot["Тренд"], label="Тренд", linestyle="--")
# Додаємо аномалії на графік
anomaly_dates_3sigma = df_plot["period"][(stohastic < lower_bound_3sigma) | (stohastic > upper_bound_3sigma)]
anomaly_values_3sigma = df_plot["data"][(stohastic < lower_bound_3sigma) | (stohastic > upper_bound_3sigma)]
anomaly_dates_iqr = df_plot["period"][(stohastic < lower_bound_iqr) | (stohastic > upper_bound_iqr)]
anomaly_values_iqr = df_plot["data"][(stohastic < lower_bound_iqr) | (stohastic > upper_bound_iqr)]
plt.scatter(anomaly_dates_3sigma, anomaly_values_3sigma, color='red', label="Аномалії (3σ)", zorder=5)
plt.scatter(anomaly_dates_iqr, anomaly_values_iqr, color='green', label="Аномалії (IQR)", zorder=5)
plt.legend()
plt.xlabel("Дата")
plt.ylabel("Середня заробітна плата")
plt.title("Моделювання середньої заробітної плати з аномаліями")
plt.grid()
plt.show()

# Гістограма стохастичної складової
plt.figure(figsize=(8, 4))
plt.hist(stohastic, bins=20, density=True, alpha=0.6, color='g', label="Стохастична складова")
plt.axvline(lower_bound_3sigma, color='r', linestyle='--', label="Межі 3σ")
plt.axvline(upper_bound_3sigma, color='r', linestyle='--')
plt.axvline(lower_bound_iqr, color='b', linestyle='--', label="Межі IQR")
plt.axvline(upper_bound_iqr, color='b', linestyle='--')
plt.title("Гістограма стохастичної складової з межами аномалій")
plt.xlabel("Значення")
plt.ylabel("Щільність")
plt.legend()
plt.grid()
plt.show()
