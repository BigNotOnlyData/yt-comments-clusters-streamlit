import logging
from collections import Counter

import numpy as np
import pandas as pd
from wordcloud import WordCloud

from loader import config
from topics import (YouTubeApi, TextPreprocessor, Reducer, Emdedder, Cluster)


class Topics:
    def __init__(self, youtube_api_key):
        """
        :param yotube_api_key: ключ API YouTube
        """
        self.youtube_api_key = youtube_api_key
        self.len_comments_corpus = None
        self.cleaned_corpus = None
        self.data = pd.DataFrame()
        self.wordcloud = None

    def generate_topics(self, url: str, max_comments: int, n_videos: int) -> None:
        """
        Главная функция, запускающая процесс кластеризации
        :param url: url видео-ролика
        :param max_comments: максимальное число комментариев
        :param n_videos: число дополнительных видео
        """
        # Парсинг комментариев с ютуба
        youtube = YouTubeApi(self.youtube_api_key)
        comments_corpus = youtube.get_comments(url,
                                               max_comments=max_comments,
                                               n_videos=n_videos)

        assert len(comments_corpus) != 0, 'Количество комментариев равно 0. ' \
                                          'Возможно они скрыты или отсутствуют, или ' \
                                          'видео недоступно'

        self.len_comments_corpus = len(comments_corpus)
        logging.info(f"Получено комментариев: {len(comments_corpus)}")

        # предобработка текста
        prep = TextPreprocessor(**config['preprocessing'])
        cleaned_corpus = prep.get_clean_text(text_corpus=comments_corpus)

        assert len(cleaned_corpus) != 0, 'Количество валидных данных после очистки текста равна 0.' \
                                         'Возможно все комментарии на иностранном языке или короткие'

        self.cleaned_corpus = cleaned_corpus
        logging.info(f"Количество комментариев после предобработки: {len(cleaned_corpus)}")

        # векторизация текста
        emb = Emdedder()
        embeddings = emb.get_embeddings_ft(corpus=cleaned_corpus,
                                           ft_params=config['fasttext']['init'],
                                           epochs=config['fasttext']['train']['epochs'])
        logging.info(f"Размерность эмбеддинга комментариев: {embeddings.shape}")

        # понижение размерности
        red = Reducer()
        X_umap = red.umap_transform(data=embeddings, umap_params=config['umap'])
        # изменим тип данных для сокращения памяти
        self.data[['x', 'y', 'z']] = X_umap.astype(np.float16)
        logging.info(f"Размерность данных после снижения размерности: {X_umap.shape}")

        # кластеризация
        cluster = Cluster()
        self.data['HDBSCAN'] = cluster.hdbscan_clusters(X_umap, **config['clusters']['hdbscan'])
        logging.info("HDBSCAN OK")
        self.data['DBSCAN'] = cluster.dbscan_clusters(X_umap, **config['clusters']['dbscan'])
        logging.info("DBSCAN OK")
        self.data['Agglomerative'] = cluster.agglomerative_clusters(X_umap,
                                                                    **config['clusters']['agglomerative'])
        logging.info("Agglomerative OK")
        self.data['Spectral'] = cluster.spectral_clusters(X_umap,
                                                          n_minmax=tuple(config['clusters']['n_minmax']),
                                                          **config['clusters']['spectral'])
        logging.info("Spectral OK")
        self.data['KMeans'] = cluster.kmeans_clusters(X_umap,
                                                      n_minmax=tuple(config['clusters']['n_minmax']),
                                                      **config['clusters']['kmeans'])

        logging.info("KMeans OK")
        logging.info("Кластеризация выполнена успешно")

        # облако слов
        # выделяем токены корпуса которых нет в токенах модели (для стопслов wordcloud)
        diff_words = set(Counter(sum(cleaned_corpus, [])).keys()).difference(set(emb.model_tokens))
        wordcloud = WordCloud(**config['wordcloud'],
                              stopwords=diff_words)
        self.wordcloud = wordcloud
        logging.info("Успешно завершено моделирование топиков")
