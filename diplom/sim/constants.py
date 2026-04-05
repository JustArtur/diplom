# ──────────────────── Физика ────────────────────
import numpy as np

DEFAULT_AIR_WEIGHT = 0.0        # Изначальная масса воздуха в баллоне Кг
AIR_MOLAR_MASS = 0.029          # Молярная масса воздуха Кг/Моль
GRAVITY_ACCELERATION = 9.98     # Ускорение свободного падения М/С
GAS_CONSTANT = 8.314_462        # Газовая постоянная Дж/(моль·K)
BALLOON_WEIGHT = 150.0          # Масса старостата в КГ + Гелия
BALLOON_VOLUME = 2000.0         # Объем старостата М3
AIR_DRAG_COEFFICIENT = 0.47     # Коэффициент сопротивления близкий к сфере
BALLOON_CROSS_SECTION = 110.0   # Площадь миделя старостата М2

# ──────────────────── Мир ────────────────────
MIN_HEIGHT = 60.0                       # минимальная допустимая высота (м)
INITIAL_POSITION = np.array([0.0, 0.0, 100.0], dtype=float)              # начальная высота аэростата (м)
TARGET_POSITION = np.array([350.0, 260.0, 1000.0], dtype=float)
SIM_TIME = np.datetime64('2024-07-01')



