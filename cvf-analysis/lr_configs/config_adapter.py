import os
import importlib

import numpy as np
import pandas as pd


from custom_logger import logger


class LRConfig:

    def __init__(
        self,
        learning_rate,
        stop_threshold,
        slope_step,
        slope_step_decimals,
        min_slope,
        max_slope,
        data_path,
        doubly_stochastic_matrix,
        iterations,
        matrix_id,
        df_random_state,
    ) -> None:
        self.learning_rate = learning_rate
        self.stop_threshold = stop_threshold
        self.slope_step = slope_step
        self.slope_step_decimals = slope_step_decimals
        self.min_slope = min_slope
        self.max_slope = max_slope
        self.doubly_stochastic_matrix = doubly_stochastic_matrix
        self.no_of_nodes = len(self.doubly_stochastic_matrix)
        self.df = pd.read_csv(
            os.path.join(
                os.getenv("CVF_CODE_ROOT", "/"),
                data_path,
            )
        )

        self.iterations = iterations
        self.matrix_id = matrix_id
        self.df = self.df.sample(frac=1, random_state=df_random_state).reset_index(
            drop=True
        )
        self.node_data_partitions = np.array_split(self.df, self.no_of_nodes)
        for i, node_data in enumerate(self.node_data_partitions):
            self.df.loc[node_data.index, "node"] = i
        self._preprocessing()

    @classmethod
    def generate_config(cls, config_file):
        config_module = importlib.import_module(f"lr_configs.{config_file}")
        cls._check_doubly_stochastic_matrix(config_module.doubly_stochastic_matrix)
        return LRConfig(
            learning_rate=config_module.learning_rate,
            stop_threshold=config_module.stop_threshold,
            min_slope=config_module.min_slope,
            max_slope=config_module.max_slope,
            slope_step=config_module.slope_step,
            slope_step_decimals=config_module.slope_step_decimals,
            data_path=config_module.data_path,
            doubly_stochastic_matrix=config_module.doubly_stochastic_matrix,
            iterations=config_module.iterations,
            matrix_id=config_file,
            df_random_state=config_module.df_random_state,
        )

    def _preprocessing(self):
        self.df["X_2"] = self.df["X"].apply(lambda x: np.square(x))
        self.df["Xy"] = self.df[["X", "y"]].apply(lambda row: row.X * row.y, axis=1)

    @classmethod
    def _check_doubly_stochastic_matrix(cls, doubly_stochastic_matrix):
        arr = np.array(doubly_stochastic_matrix)
        try:
            for i, row in enumerate(arr):
                assert np.sum(row) == 1 or np.isclose(np.sum(row), 1), f"Row index {i}"
                assert (
                    np.sum(arr[:, i]) or np.isclose(np.sum(arr[:, i], 1)) == 1
                ), f"Column index {i}"
        except Exception as e:
            logger.exception("Invalid doubly stochastic matrix!")
            exit(1)

        logger.info("Doubly stochastic matrix validated!")
