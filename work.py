import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Завантаження даних
path = "132-serednia-nominalna-zarobitna-plata-za-misiats-grn.csv"
df = pd.read_csv(path, delimiter=";", encoding="1251")

# Підготовка даних
rates = df[["period", "data"]].copy()
rates["period"] = pd.to_datetime(rates["period"], dayfirst=True)
rates["data"] = pd.to_numeric(rates["data"], errors="coerce")
rates = rates.sort_values("period").dropna()

# Агрегація
aggregated_df = rates.groupby("period", as_index=False).agg({"data": "sum"})

# Початкові дані для аналізу
x = (aggregated_df["period"] - aggregated_df["period"].min()).dt.days
y = aggregated_df["data"]

# Попередній тренд для виявлення аномалій (оригінальний)
degree = 3
coeffs_initial = np.polyfit(x, y, degree)
trend_initial = np.polyval(coeffs_initial, x)
stohastic_initial = y - trend_initial

# Виявлення аномалій за IQR
q1, q3 = np.percentile(stohastic_initial, [25, 75])
iqr = q3 - q1
lower_bound_iqr = q1 - 1.5 * iqr
upper_bound_iqr = q3 + 1.5 * iqr
mask = (stohastic_initial >= lower_bound_iqr) & (stohastic_initial <= upper_bound_iqr)

# Дані без аномалій
x_clean = x[mask]
y_clean = y[mask]
aggregated_df_clean = aggregated_df[mask].copy()

# Навчання моделі на очищених даних
coeffs_clean = np.polyfit(x_clean, y_clean, degree)
trend_clean = np.polyval(coeffs_clean, x_clean)
trend_equation_clean = f"y = {coeffs_clean[0]:.5f}x³ + {coeffs_clean[1]:.5f}x² + {coeffs_clean[2]:.5f}x + {coeffs_clean[3]:.5f}"

# Обчислення R² для очищеної моделі
residuals_clean = y_clean - trend_clean
ss_tot_clean = np.sum((y_clean - np.mean(y_clean))**2)
ss_res_clean = np.sum(residuals_clean**2)
r_squared_clean = 1 - (ss_res_clean / ss_tot_clean)

# Обчислення R² для оригінальної моделі
residuals_initial = y - trend_initial
ss_tot_initial = np.sum((y - np.mean(y))**2)
ss_res_initial = np.sum(residuals_initial**2)
r_squared_initial = 1 - (ss_res_initial / ss_tot_initial)

# Стохастична складова на очищених даних
stohastic_clean = y_clean - trend_clean

# Статистичні характеристики стохастичної складової (очищені дані)
mean_value = np.mean(stohastic_clean)
std_dev = np.std(stohastic_clean)
variance = np.var(stohastic_clean)

print('-------- Статистичні характеристики стохастичної складової (очищені дані) ---------')
print(f"Математичне сподівання (середня): {mean_value:.2f}")
print(f"Середньоквадратичне відхилення (σ): {std_dev:.2f}")
print(f"Дисперсія: {variance:.2f}")
print(f"Рівняння полінома (очищені дані): {trend_equation_clean}")
print(f"R² (очищені дані): {r_squared_clean:.4f}")
print(f"R² (оригінальні дані): {r_squared_initial:.4f}")
print(f"Кількість видалених аномалій: {len(y) - len(y_clean)}")
print('----------------------------------------------------------------------------')

# Прогнозування (на основі очищеної моделі)
last_date = aggregated_df["period"].max()
days_range = (last_date - aggregated_df["period"].min()).days
forecast_days = days_range // 2
forecast_x = np.arange(x.max() + 1, x.max() + forecast_days + 1)
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
forecast_trend = np.polyval(coeffs_clean, forecast_x)

# Об'єднання даних для візуалізації
df_plot = aggregated_df.copy()
df_plot["Тренд (оригінальний)"] = trend_initial
df_plot["Тренд (очищений)"] = np.polyval(coeffs_clean, x)  # Тренд для всіх x
forecast_df = pd.DataFrame({"period": forecast_dates, "Тренд (очищений)": forecast_trend})

# Візуалізація
plt.figure(figsize=(14, 7))
# Оригінальні дані та тренд
plt.plot(df_plot["period"], df_plot["data"], label="Оригінальні дані", color='blue', alpha=0.5)
plt.plot(df_plot["period"], df_plot["Тренд (оригінальний)"], label="Тренд (оригінальний)", color='orange', linestyle='--', alpha=0.7)
# Очищені дані та тренд
plt.scatter(aggregated_df_clean["period"], aggregated_df_clean["data"], label="Очищені дані", color='blue')
plt.plot(aggregated_df_clean["period"], trend_clean, label="Тренд (очищений)", color='green', linestyle='--')
# Прогноз
plt.plot(forecast_df["period"], forecast_df["Тренд (очищений)"], label="Прогноз (очищений)", color='green', linestyle='-.')
# Аномалії
anomaly_dates = df_plot["period"][~mask]
anomaly_values = df_plot["data"][~mask]
plt.scatter(anomaly_dates, anomaly_values, color='red', label="Аномалії (IQR)", zorder=5)

plt.legend()
plt.xlabel("Дата")
plt.ylabel("Середня заробітна плата")
plt.title("Порівняння оригінального та очищеного тренду з прогнозуванням")
plt.grid()
plt.show()

# Гістограма стохастичної складової (очищені дані)
plt.figure(figsize=(8, 4))
plt.hist(stohastic_clean, bins=20, density=True, alpha=0.6, color='g', label="Стохастична складова (очищені)")
plt.axvline(lower_bound_iqr, color='b', linestyle='--', label="Межі IQR (початкові)")
plt.axvline(upper_bound_iqr, color='b', linestyle='--')
plt.title("Гістограма стохастичної складової (очищені дані)")
plt.xlabel("Значення")
plt.ylabel("Щільність")
plt.legend()
plt.grid()
plt.show()