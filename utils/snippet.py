
import numpy as np
import plotly.express as px
import streamlit as st

from loader import VALID_ALGORITHM, NOISE, NOISE_NAME
from topics import Topics
from utils.cluster import plot_3d, pie_clusters_size, create_wc_for_all_clusters
from utils.tool import get_color_map


@st.cache(max_entries=20, ttl=3600)
def get_topics(url: str, max_comments: int, n_videos: int, yotube_api_key: str) -> Topics:
    """
    Запускаем весь процесс моделирования топиков
    :param url: url-видео
    :param max_comments: максимум комментов
    :param n_videos: макисмум доп. видео
    :param yotube_api_key: ключ API YouTube
    :return: экземпляр Topics с результатами моделирования
    """
    top = Topics(yotube_api_key)
    top.generate_topics(url, max_comments, n_videos)
    return top


def plot_streamlit_clusters(result: Topics, algorithm: str) -> None:
    """
    Часть страницы для отображения информации о кластерах
    :param result: объект Topics с результатами кластеризации
    :param algorithm: название алгоритма
    """
    st.header(algorithm)
    # вкладки для алгоритмов
    if algorithm in VALID_ALGORITHM:
        # метки кластеров для каждого примера
        labels = result.data[algorithm]
        # Уникальные кластеры
        unique_clusters = np.unique(labels)
        # фильтруем от шума, чтобы он не занимал цвет и не учитывался в облаке слов
        unique_clusters = unique_clusters[unique_clusters != NOISE]
        # создаём словарь цветов для кластеров
        color_discrete_map = get_color_map(unique_clusters,
                                           px.colors.qualitative.Dark24)
        # создаем 3D график
        fig_3d = plot_3d(result.data[['x', 'y', 'z', algorithm]],
                         color=algorithm,
                         color_discrete_map=color_discrete_map)
        # создаем график пончик
        fig_cluster_size = pie_clusters_size(result.data,
                                             algorithm,
                                             color_discrete_map=color_discrete_map)

        # соединяем токены каждого комментария в одну строку
        # обертываем в np.array для удобства фильтрации по кластерам
        text_corpus = np.array([' '.join(tokens) for tokens in result.cleaned_corpus])

        # создаем графики облаков слов
        figs_of_wordcloud = create_wc_for_all_clusters(text_corpus=text_corpus,
                                                       labels=labels,
                                                       unique_clusters=unique_clusters,
                                                       wc=result.wordcloud)

        # выводим количество кластеров
        st.metric('Количество кластеров', len(unique_clusters),
                  help=f'Без кластера: {NOISE_NAME}')

        # рисуем 3D проекцию
        st.subheader('Проекция комментариев в 3-мерном пространстве')
        st.plotly_chart(fig_3d, use_container_width=True)

        # рисуем пончик
        st.subheader('Диаграмма размера кластеров')
        st.plotly_chart(fig_cluster_size, use_container_width=True)

        # Рисуем облако слов
        st.subheader('Облако слов')
        for fig_wordcloud, cluster in zip(figs_of_wordcloud, unique_clusters):
            st.markdown(f'**Кластер: {cluster}**')
            st.pyplot(fig_wordcloud)
    else:
        # вкладка Данные
        fig_3d = plot_3d(result.data[['x', 'y', 'z']], color=None)
        st.subheader('Проекция комментариев в 3-мерном пространстве')
        st.plotly_chart(fig_3d, use_container_width=True)
