from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

from loader import NOISE, NOISE_NAME, NOISE_COLOR


# кеш не ставить на plot_wc!!!!!!
# бесконечно считает функцию

def create_wc_for_one_cluster(cloud: np.array) -> plt.Figure:
    """
    Создаёт облако слов
    :param cloud: данные для отрисовки картинки
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(cloud, interpolation='bilinear')
    ax.axis("off")
    return fig


def create_wc_for_all_clusters(text_corpus: np.array, labels: np.array,
                               unique_clusters: np.array, wc: WordCloud) -> List[plt.Figure]:
    """
    Создаёт список из графиков для каждого кластера
    :param text_corpus: очищенный корпус текстов
    :param labels: метки кластеров
    :param unique_clusters: уникальноые кластеры
    :param wc: объект WordCloud
    :return:
    """
    figs = []
    for cluster in unique_clusters:
        try:
            text_sample = ' '.join(text_corpus[labels == cluster])
            cloud = wc.generate(text_sample)
            fig = create_wc_for_one_cluster(cloud)
            figs.append(fig)
        except Exception:
            # ошибка что нет слов для облака возникает возможно из-за того что модель FastText
            # не рассматривает слова с указанной малой частотой
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.axis("off")
            figs.append(fig)
    return figs


@st.cache(max_entries=30, ttl=3600)
def plot_3d(df: pd.DataFrame,
            x='x',
            y='y',
            z='z',
            color: Optional[str] = None,
            color_discrete_map: Optional[dict] = None) -> go.Figure:
    """
    Создаёт трехмерную визуализацию
    :param df: таблица с данными с обязательными колонками x, y, z
    :param x: размерность x
    :param y: размерность y
    :param z: размерность z
    :param color: имя колонки кластеров
    :param color_discrete_map: словарь с цветами для кластеров
    :return: figure plotly
    """
    # если передан цвет (кластеры)

    if color is not None:
        # строковый тип нужен для дискретного отображения кластеров plotly
        df_ = df.copy()
        df_.loc[:, color] = df_[color].astype(str)
        # color_discrete_map = {str(k): v for k, v in color_discrete_map.items()}
        # Разделяем на два графика чтобы прозрачность кластера Шума регулироовать, другое решение?
        fig_outlier = None
        df_outlier = df_[df_[color] == str(NOISE)]
        df_norm = df_[df_[color] != str(NOISE)]
        # отрисовываем выбросы если есть
        if df_outlier.shape[0]:
            fig_outlier = px.scatter_3d(df_outlier,
                                        x=x,
                                        y=y,
                                        z=z,
                                        opacity=.4,
                                        color_discrete_sequence=[NOISE_COLOR],
                                        hover_name=[NOISE_NAME] * len(df_outlier),
                                        )

        # отрисовываем кластеры
        fig = px.scatter_3d(df_norm,
                            x=x,
                            y=y,
                            z=z,
                            color=color,
                            labels={color: 'Кластер'},
                            color_discrete_map=color_discrete_map,
                            )

        # если есть выбросы объединяем данные
        data = fig.data
        if fig_outlier is not None:
            data = data + fig_outlier.data

        fig = go.Figure(data=data)
    else:
        fig = px.scatter_3d(df, x=x, y=y, z=z)

    fig.update_layout(legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="right",
        x=1,
        title_text='Кластеры:',
    ))
    return fig


@st.cache(max_entries=30, ttl=3600)
def pie_clusters_size(data: pd.DataFrame,
                      algorithm: str,
                      color_discrete_map: Optional[dict] = None
                      ) -> go.Figure:
    """
    Отрисовывает круговую диаграму размеров кластеров
    :param data: таблица с данными
    :param algorithm: имя колонки кластеров
    :param color_discrete_map: словарь с цветами для кластеров
    :return: figure plotly
    """
    # Считаем количество комментариев в каждом кластере
    df_pie = data[algorithm].value_counts().reset_index()
    # Меняем имя кластера шума
    df_pie['index'] = df_pie['index'].where(df_pie['index'] != NOISE, NOISE_NAME)
    # приводим к строковому типу для идентификации с color_discrete_map
    df_pie['index'] = df_pie['index'].astype(str)
    fig = px.pie(df_pie,
                 values=algorithm,
                 color='index',  # для испоьзования color_discrete_map
                 names='index',  # для отображения легенды
                 hole=0.4,
                 labels={'index': 'Кластер', algorithm: 'Количество комментариев'},
                 color_discrete_map=color_discrete_map,
                 )

    fig.update_layout(legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="right",
        x=1.1,
        title_text='Кластеры:',
    ))
    return fig
