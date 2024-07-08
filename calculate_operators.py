import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import gamma
import openpyxl
import os

def read_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, usecols="A:C", skiprows=1, nrows=25)
    df.columns = ["Время", "call volume", "СВО"]
    return df

def calculate_traffic(data, forecast):
    data = data.copy()
    data["call volume"] = data["call volume"].apply(
        lambda x: float(str(x).replace('%', '')) / 100 if isinstance(x, str) else x
    )
    data["Количество звонков"] = data["call volume"] * forecast
    data["СВО"] = pd.to_numeric(data["СВО"], errors='coerce')
    data["Нагрузка (Erlang)"] = data["Количество звонков"] * data["СВО"] / 3600
    data["Нагрузка (Erlang)"].fillna(0, inplace=True)
    return data

def erlang_c_approx(traffic, service_level):
    if pd.isna(traffic) or traffic <= 0:
        return 0
    n = math.ceil(traffic)
    while True:
        try:
            L = (traffic**n / math.factorial(n)) * (n / max(0.1, n - traffic))
            sum_L = sum((traffic**i) / math.factorial(i) for i in range(n))
            Pw = L / (sum_L + L)
            if 1 - Pw >= service_level:
                return n
        except OverflowError:
            return n
        n += 1

def calculate_agents(data, service_level=0.8):
    data["Количество операторов"] = data["Нагрузка (Erlang)"].apply(
        lambda x: erlang_c_approx(x, service_level)
    )
    return data

def adjust_for_attrition(data, attrition_rate=0.25):
    data['Количество операторов по расписанию'] = data['Количество операторов'] / (1 - attrition_rate)
    data['Количество операторов по расписанию'] = data['Количество операторов по расписанию'].apply(math.ceil)
    return data

def process_all_days(file_list, days_of_week, forecasts, attrition_rate=0.25):
    results = []
    scheduled_operators = []
    for file_path, file_forecasts in zip(file_list, forecasts):
        file_results = []
        file_scheduled_results = []
        for day, forecast in zip(days_of_week, file_forecasts):
            data = read_data(file_path, day)
            data = calculate_traffic(data, forecast)
            data = calculate_agents(data)
            data = adjust_for_attrition(data, attrition_rate)
            day_operators = data["Количество операторов"].tolist()
            day_scheduled_operators = data["Количество операторов по расписанию"].tolist()
            file_results.append(day_operators)
            file_scheduled_results.append(day_scheduled_operators)
        results.append(file_results)
        scheduled_operators.append(file_scheduled_results)
    return results, scheduled_operators

def plot_results(results, scheduled_results):
    plt.figure(figsize=(24, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(pd.DataFrame(results), annot=True, fmt="d", cmap="YlGnBu")
    plt.title('Распределение операторов по часам')

    plt.subplot(1, 2, 2)
    sns.heatmap(pd.DataFrame(scheduled_results), annot=True, fmt="d", cmap="YlOrBr")
    plt.title('Количество операторов по расписанию по часам')

    plt.show()

def save_data_to_excel(data, file_path):
    output_path = f"scheduled_{os.path.basename(file_path)}"
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        pd.DataFrame(data).to_excel(writer, index=False)
    print(f"Results saved to {output_path}")

def main(file_list, days_of_week, forecasts):
    results, scheduled_operators = process_all_days(file_list, days_of_week, forecasts, 0.25)
    flat_results = [item for sublist in results for item in sublist]
    flat_scheduled_operators = [item for sublist in scheduled_operators for item in sublist]
    plot_results(flat_results, flat_scheduled_operators)
    for idx, file_path in enumerate(file_list):
        save_data_to_excel(scheduled_operators[idx], file_path)

if __name__ == "__main__":
    file_list = ['2.xlsx', '3.xlsx', '4.xlsx']
    days_of_week = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]
    forecasts = [
        [11522, 8465, 7768, 11540, 10543, 10659, 14727, 11801, 8960, 8547, 11605, 11832, 12416, 12481, 14148, 11786, 9923, 14285, 14112, 13238, 13815, 15224, 9891, 9083, 18385, 19288, 18504, 11328, 16387, 12020],
        [16280, 13770, 12589, 16984, 15818, 16669, 13155, 16519, 15589, 14227, 19496, 19032, 19422, 19639, 22971, 17348, 15365, 21489, 21071, 22051, 20781, 22651, 17427, 14613, 22606, 21415, 24124, 27497, 41041, 29255],
        [4156, 2285, 2225, 3136, 2769, 2823, 5152, 7430, 1704, 1658, 1868, 1999, 2088, 2160, 2390, 2060, 1670, 2181, 2212, 2334, 2334, 2745, 2082, 1828, 2572, 3258, 4556, 5933, 7337, 5112]
    ]
    main(file_list, days_of_week, forecasts)
