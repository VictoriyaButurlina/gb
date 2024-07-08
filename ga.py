import random
import pandas as pd
import datetime
from deap import base, creator, tools, algorithms
import statistics

# Создаем классы FitnessMin и Individual один раз
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)
# --- Константы ---
S = 3  # Количество скиллов
group_skills = {
    1: [1],
    2: [2],
    3: [3],
    4: [1, 2],
    5: [2, 3],
    6: [1, 2, 3]
}

# --- Параметры смен ---
SHIFT_TYPES = {
    "5x2": {"duration": 9, "breaks": 3, "lunch": True, "lunch_duration": 40,
             "break_durations": [10, 10, 10, 0], "earliest_start": 7, "latest_end": 24,
             "overtime_allowed": True, "40h_week": True, "sum_accounting": True},
    "2x2 Day": {"duration": 12, "breaks": 4, "lunch": True, "lunch_duration": 40,
               "break_durations": [10, 10, 10, 15], "earliest_start": 7, "latest_end": 24,
               "overtime_allowed": True, "40h_week": False, "sum_accounting": True},
    "2x2 Night": {"duration": 12, "breaks": 4, "lunch": True, "lunch_duration": 40,
                "break_durations": [10, 10, 10, 15], "earliest_start": 19, "latest_end": 7,
                "overtime_allowed": True, "40h_week": False, "sum_accounting": True},
    "5x2_part_time": {"duration": 4, "breaks": 1, "lunch": False, "lunch_duration": 0,
                       "break_durations": [15, 0, 0, 0], "earliest_start": 7, "latest_end": 24,
                       "overtime_allowed": False, "40h_week": False, "sum_accounting": True},
    "2x2 Day_part_time": {"duration": 5.5, "breaks": 2, "lunch": False, "lunch_duration": 0,
                          "break_durations": [10, 10, 0, 0], "earliest_start": 7, "latest_end": 24,
                          "overtime_allowed": False, "40h_week": False, "sum_accounting": True},
    "2x2 Night_part_time": {"duration": 5.5, "breaks": 2, "lunch": False, "lunch_duration": 0,
                            "break_durations": [10, 10, 0, 0], "earliest_start": 19, "latest_end": 7,
                            "overtime_allowed": False, "40h_week": False, "sum_accounting": True},
}
df_cost = pd.DataFrame({
    "shift_type": SHIFT_TYPES.keys(),
    "cost": [4500.0, 6000.0, 10800.0, 2250.0, 3000.0, 5400.0]  # Стоимость для 5x2, 2x2 Day, 2x2 Night, 5x2_part_time, 2x2 Day_part_time, 2x2 Night_part_time
})

# --- Ограничения по количеству сотрудников в каждой группе ---
group_limits = {
    1: 50,  # Группа 1: навык 1 (максимум 50)
    2: 60,  # Группа 2: навык 2 (максимум 60)
    3: 30,  # Группа 3: навык 3 (максимум 30)
    4: 60,  # Группа 4: навыки 1 и 2 (максимум 60 по навыку 2)
    5: 60,  # Группа 5: навыки 2 и 3 (максимум 60 по навыку 2)
    6: 60   # Группа 6: навыки 1, 2 и 3 (максимум 60 по навыку 2)
}

# --- Загрузка данных ---
def load_skill_data(filename):
    """Загружает данные о нагрузке из файла Excel с листами для дней недели."""
    xls = pd.ExcelFile(filename)
    skill_data = {}
    for day in range(7):  # Читаем данные для всех дней недели
        sheet_name = xls.sheet_names[day]
        df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=1, usecols="A:E").copy()
        df['Время'] = df['Время'].replace(':00', '00:00')
        skill_data[day] = df.set_index("Время")
    return skill_data

skill1_data = load_skill_data("2.xlsx")
skill2_data = load_skill_data("3.xlsx")
skill3_data = load_skill_data("4.xlsx")

# --- Класс сотрудника ---
class Employee:
    def __init__(self, id, skills, contract_type=None, min_start=None, max_end=None):
        self.id = id
        self.skills = skills
        self.contract_type = contract_type
        self.min_start = min_start
        self.max_end = max_end
        self.group = None

# --- Функции ---
def get_shift_type(group, day, hour, shift_start):
    """Определяет тип смены с учетом начала смены."""
    if hour >= 19 or hour < 7:
        return "2x2 Night"
    elif group in [4, 5, 6] and shift_start >= 19:
        return "2x2 Night"
    elif group in [4, 5, 6]:
        return "2x2 Day"
    elif group in [1, 2, 3]:
        return "5x2"
    else:
        return "5x2"

def calculate_fitness(individual, skill_data, df_cost, group_limits):
    # Вычисляет пригодность особи (расписания) на основе затрат, штрафов и ограничений.
    total_cost = 0
    total_unserved_calls = 0

    #  individual в словарь
    decoded_individual = {}
    for i, num_employees in enumerate(individual):
        day = i // 24
        hour = i % 24
        decoded_individual[day * 24 + hour] = {group: num_employees % (group_limits[group] + 1) for group in group_skills}

    for day in range(7):  # Все дни недели
        for hour in range(24):
            operators_per_skill = {1: 0, 2: 0, 3: 0}
            for group, num_employees in decoded_individual[day * 24 + hour].items():
                # Учет перерывов
                shift_type = get_shift_type(group, day, hour, hour)
                break_duration = SHIFT_TYPES[shift_type]['break_durations']
                if hour in break_duration:
                    num_employees = 0

                # Проверка, достаточно ли сотрудников с нужными навыками
                for skill in group_skills[group]:
                    operators_per_skill[skill] += num_employees

            # Проверка удовлетворения нагрузки
            for skill in range(1, S + 1):
                time_index = datetime.time(hour, 0)
                operators_needed = skill_data[skill - 1][day].loc[
                    time_index, 'Количество операторов по расписанию'
                ]
                unserved_calls = max(0, operators_needed - operators_per_skill[skill])
                total_unserved_calls += unserved_calls

            # Расчет стоимости смен
            for group, num_employees in decoded_individual[day * 24 + hour].items():
                if num_employees > 0:
                    shift_type = get_shift_type(group, day, hour, hour)
                    # Проверка, существует ли тип смены в df_cost
                    if shift_type in df_cost["shift_type"].values:
                        shift_cost = df_cost[df_cost["shift_type"] == shift_type]["cost"].values[0]
                        total_cost += shift_cost * num_employees
                    else:
                        print(f"Warning: Неизвестный тип смены: {shift_type}")  # Выводим предупреждение

    # Штраф за необслуженные звонки
    total_cost += total_unserved_calls * 1000  # Можно менять стоимость штрафа

    # Штраф за превышение лимита по группам
    for group in group_skills:
        total_employees_in_group = sum(decoded_individual[day * 24 + hour].get(group, 0) for day in range(7) for hour in range(24))
        if total_employees_in_group > group_limits[group]:
            total_cost += 10000 * (total_employees_in_group - group_limits[group]) # Штраф за превышение

    return (total_cost,)

def decode_schedule(individual):
   #Декодирует генотип особи в расписание с указанием времени начала смен.
    schedule = {}
    for i, num_employees in enumerate(individual):
        day = i // 24
        hour = i % 24
        schedule.setdefault(day, {}).setdefault(hour, {})
        for group in group_skills:
            if num_employees % (group_limits[group] + 1) > 0:
                shift_type = get_shift_type(group, day, hour, hour)
                schedule[day][hour][shift_type] = {
                    'start': hour,
                    'num_employees': num_employees % (group_limits[group] + 1),
                    'group': group
                }
    return schedule

def print_schedule(individual):
    #Выводит расписание в виде таблицы для всех дней недели.
    schedule = decode_schedule(individual)

    for day in range(7):
        print(f"\n===== День {day + 1} =====")
        shifts_by_hour = {}
        for hour in range(24):
            shifts_by_hour.setdefault(hour, [])
            for shift_type, shift_info in schedule[day].get(hour, {}).items():
                shifts_by_hour[hour].append(f"{shift_type} - начало: {shift_info['start']}:00, агентов: {shift_info['num_employees']}")

        for hour, shifts in shifts_by_hour.items():
            if shifts:
                print(f"  {hour:02}:00 - {hour+1:02}:00")
                for shift in shifts:
                    print(f"    {shift}")
                lunch_hour = hour + SHIFT_TYPES[shift_type]['duration'] - 1  # Определяем час обеда
                if lunch_hour in SHIFT_TYPES[shift_type]['break_durations']:
                    lunch_start = lunch_hour
                    lunch_end = lunch_hour + 1
                    print(f"    - Обед: {lunch_start:02}:00 - {lunch_end:02}:00")

        # Подсчет количества смен
        total_5x2 = sum(len(shifts_by_hour[hour]) for hour in range(7, 19) if shifts_by_hour[hour])
        total_2x2 = sum(len(shifts_by_hour[hour]) for hour in range(19, 24) if shifts_by_hour[hour]) + \
                    sum(len(shifts_by_hour[hour]) for hour in range(0, 7) if shifts_by_hour[hour])
        print(f"Всего 5x2: {total_5x2}")
        print(f"Всего 2x2: {total_2x2}")
        print(f"Всего: {total_5x2 + total_2x2}")

# --- Создание списка сотрудников ---
def create_employee_list(best_schedule):
    #Создает список сотрудников на основе оптимального расписания.
    employees = []
    employee_id = 1
    for day in range(7):
        for hour in range(24):
            for shift_type, shift_info in best_schedule[day].get(hour, {}).items():
                for _ in range(shift_info['num_employees']):
                    skills = group_skills[shift_info['group']]
                    employee = Employee(employee_id, skills, shift_type)
                    employee.group = shift_info['group']
                    # Определение start и end на основе типа смены (упрощенно)
                    if shift_type == "2x2 Night" or shift_type == "2x2 Night_part_time":
                        employee.min_start = 19
                        employee.max_end = 7
                    else:
                        employee.min_start = 7
                        employee.max_end = 16 if shift_type == "5x2" or shift_type == "5x2_part_time" else 24
                    employees.append(employee)
                    employee_id += 1
    return employees
def save_employees_to_excel(employees, filename):
    #Сохраняет список сотрудников в файл Excel.
    data = {
        "ID": [employee.id for employee in employees],
        "Skills": [", ".join(map(str, employee.skills)) for employee in employees],
        "Shift": [employee.contract_type for employee in employees],
        "Group": [employee.group for employee in employees],
        "Start": [employee.min_start for employee in employees],
        "End": [employee.max_end for employee in employees]
    }
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
    print(f"Данные сохранены в файл {filename}")
# --- Генетический алгоритм ---
def genetic_algorithm_deap(population_size, generations, mutation_rate, skill_data, df_cost, group_limits):
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]  # Define days_of_week
    toolbox = base.Toolbox()

    # Вычисляем суммарное количество операторов по 3 скиллам на каждый час
    toolbox = base.Toolbox()

    # Вычисляем суммарное количество операторов по 3 скиллам на каждый час
    total_operators_per_hour = {}
    for day in range(7):
        total_operators_per_hour[day] = []
        for hour in range(24):
            total_operators_per_hour[day].append(sum(skill_data[0][day].iloc[hour, 3:6]) +
                                              sum(skill_data[1][day].iloc[hour, 3:6]) +
                                              sum(skill_data[2][day].iloc[hour, 3:6]))
        else:
                total_operators_per_hour[day].append(0)
    def generate_individual():
        #Создает особь (расписание) с учетом ограничений по группам.
        individual = []
        for day in range(7):
            for hour in range(24):
                for group in group_skills:
                    # Учитываем ограничение по группе
                    max_employees = min(total_operators_per_hour[day][hour], group_limits[group])
                    num_employees = random.randint(0, max_employees)
                    individual.append(num_employees)
        return individual

    toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", calculate_fitness, skill_data=skill_data, df_cost=df_cost, group_limits=group_limits)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=10, indpb=mutation_rate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", statistics.mean)
    stats.register("min", min)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, stats=stats, halloffame=hof, verbose=True)

    # Форматируем результаты
    best_fitness = hof[0].fitness.values[0]
    best_fitness_formatted = f"{best_fitness:,.0f}"  # Форматирование с разделителями для тысяч

    # Вывод результатов с форматированием
    print("\nРезультаты генетического алгоритма:")
    for record in logbook:
        gen = record['gen']
        nevals = record['nevals']
        avg = record['avg']
        min_ = record['min']
        try:
            avg_num = float(avg)  # Приведение к числовому типу
            min_num = float(min_)  # Приведение к числовому типу
            print(f"gen\t{gen}\tnevals\t{nevals}\tavg\t{avg_num:,.0f}\tmin\t{min_num:,.0f}")  # Форматируем avg и min
        except ValueError:
            print(f"gen\t{gen}\tnevals\t{nevals}\tavg\t{avg}\tmin\t{min_}")  # Без форматирования в случае ошибки

    return hof[0], best_fitness_formatted  # Возвращаем особь и отформатированное значение

# --- Запуск генетического алгоритма ---
if __name__ == "__main__":
    best_individual, best_fitness = genetic_algorithm_deap(
        population_size=100, generations=10, mutation_rate=0.1,
        skill_data=[skill1_data, skill2_data, skill3_data], df_cost=df_cost,
        group_limits=group_limits
    )

    # --- Создание списка сотрудников ---
    best_schedule = decode_schedule(best_individual)
    employees = create_employee_list(best_schedule)

    # --- Сохранение списка сотрудников в файл Excel ---
    save_employees_to_excel(employees, "employees.xlsx")
    # --- Вывод списка сотрудников ---
    print("\nСписок сотрудников:")
    for employee in employees:
        print(
            f"ID: {employee.id}, Skills: {employee.skills}, Shift: {employee.contract_type}, Group: {employee.group}, Start: {employee.min_start}, End: {employee.max_end}"
        )