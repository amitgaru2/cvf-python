import os

import numpy as np

learning_rate = 0.0001
stop_threshold = 0.0001
slope_step = np.float64(0.025)
slope_step_decimals = 3
min_slope = np.float64(0.500)
max_slope = np.float64(1.900)

data_path = os.path.join(
    "linear_regression", "SOCR-HeightWeight-XY.csv"
)  # from code root

iterations = 100

doubly_stochastic_matrix = [
    [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
    [1 / 6, 5 / 6, 0, 0, 0, 0],
    [1 / 6, 0, 5 / 6, 0, 0, 0],
    [1 / 6, 0, 0, 5 / 6, 0, 0],
    [1 / 6, 0, 0, 0, 5 / 6, 0],
    [1 / 6, 0, 0, 0, 0, 5 / 6],
]


df_random_state = 36
