from typing import List

import numpy as np
import pandas as pd

from loader import NOISE_NAME, NOISE_COLOR


def my_cyclic_gen(sequence):
    """
    Создаёт цикличный генератор последовательности
    :param sequence: итерируемая последовательность
    :return: элементы последовательности по порядку и цикличны,
    когда последовательность заканчивается
    """
    assert len(sequence) != 0, 'Пустая последовательность'
    j = 0
    while 1:
        yield sequence[j]
        j += 1
        if j >= len(sequence):
            j = 0


def get_color_map(clusters: np.array, colors: List[str]) -> dict:
    """
    Создаёт словарь цветов для кластеров
    :param clusters: последовательность кластеров
    :param colors: последовательность цветов
    :return: словарь цветов
    """
    gen = my_cyclic_gen(colors)
    color_map = {str(k): v for k, v, in zip(clusters, [next(gen) for _ in range(len(clusters))])}
    # отдельно создаем цвет для шума
    color_map[NOISE_NAME] = NOISE_COLOR
    return color_map


def remove_outlier_sigma(df: pd.DataFrame, column: str, sigma: int = 3) -> pd.DataFrame:
    """
    Очистка от выбросов по значению стандартного отклонения
    :param df: исходный фрейм
    :param column: имя колонки, у которой произвести очистку
    :param sigma: стандартное отклонение, в пределах которого фильтруются данные
    :return: фрейм в пределах sigma
    """
    df_clean = df.copy()
    df_clean['std'] = (df[column] - df[column].mean()) / df[column].std()
    return df_clean[df_clean['std'].abs() <= sigma]
