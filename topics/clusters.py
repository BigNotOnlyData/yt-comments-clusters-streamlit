# Возникает ошибка при импорте hdbscan. Решение - установка библиотеки joblib версии 1.1.0
# https://stackoverflow.com/questions/73830225/init-got-an-unexpected-keyword-argument-cachedir-when-importing-top2vec
from typing import Tuple

import hdbscan
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans, DBSCAN
from sklearn.metrics import (calinski_harabasz_score,
                             davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import MinMaxScaler


class Cluster:
    def __init__(self):
        pass

    def hdbscan_clusters(self, X, **hdbscan_params) -> np.array:
        """
        Кластеры алгоритма HDBSCAN
        """
        model = hdbscan.HDBSCAN(**hdbscan_params)
        model.fit(X)
        return model.labels_.astype(np.int8)

    def dbscan_clusters(self, X, **dbscan_params) -> np.array:
        """
        Кластеры алгоритма DBSCAN
        """
        model = DBSCAN(**dbscan_params)
        model.fit(X)
        return model.labels_.astype(np.int8)

    def agglomerative_clusters(self, X, **aglomerat_params) -> np.array:
        """
        Кластеры алгоритма AgglomerativeClustering
        """
        model = AgglomerativeClustering(**aglomerat_params)
        model.fit(X)
        return model.labels_.astype(np.int8)

    def spectral_clusters(self, X, n_minmax: Tuple[int, int], **spectral_params) -> np.array:
        """
        Кластеры алгоритма SpectralClustering
        :param n_minmax: диапазон количества кластеров для поиска лучшего числа
        """
        model = SpectralClustering(**spectral_params)
        best_clusters = self.get_best_n_clusters(X, model, n_minmax)
        model = SpectralClustering(n_clusters=best_clusters, **spectral_params)
        model.fit(X)
        return model.labels_.astype(np.int8)

    def kmeans_clusters(self, X, n_minmax: Tuple[int, int], **kmeans_params) -> np.array:
        """
        Кластеры алгоритма KMeans
        :param n_minmax: диапазон количества кластеров для поиска лучшего числа
        """
        model = KMeans(**kmeans_params)
        best_clusters = self.get_best_n_clusters(X, model, n_minmax)
        model = KMeans(n_clusters=best_clusters, **kmeans_params)
        model.fit(X)
        return model.labels_.astype(np.int8)

    def get_best_n_clusters(self, X_train, model, n_minmax: Tuple[int, int]) -> int:
        """
        Вычисляет лучшее количество кластеров ориентируясь на метрики
        :param X_train: обучающаяся выборка
        :param model: модель для которой определить лучшие кластеры
        :param n_minmax: интервал в котором ищем
        """
        n_min, n_max = n_minmax
        # считаем метрики
        calinski_harabasz_scores = []
        davies_bouldin_scores = []
        silhouette_scores = []
        for n in range(n_min, n_max + 1):
            model.set_params(n_clusters=n)
            model.fit(X_train)
            calinski_harabasz_scores.append(calinski_harabasz_score(X_train, model.labels_))
            davies_bouldin_scores.append(davies_bouldin_score(X_train, model.labels_))
            silhouette_scores.append(silhouette_score(X_train, model.labels_))

        # вычисляем лучшее количество кластеров
        df_metrics = pd.DataFrame(data={'cal_har': calinski_harabasz_scores,
                                        'dav_bould': davies_bouldin_scores,
                                        'silhouette': silhouette_scores
                                        },
                                  index=range(n_min, n_max + 1)
                                  )

        # инвертируем для удобства сравнивания метрик по максимуму
        # (так как у этой метрики лучшие значения у минимума)
        df_metrics['dav_bould'] = 1 / df_metrics['dav_bould']

        # Приводим к одному маштабу для дальнейшего объединения метрик
        scaler = MinMaxScaler()
        df_metrics[df_metrics.columns] = scaler.fit_transform(df_metrics)

        # объединенная метрика
        union_metric = 1 / 3 * df_metrics['cal_har'] \
                       + 1 / 3 * df_metrics['dav_bould'] \
                       + 1 / 3 * df_metrics['silhouette']

        return union_metric.idxmax()
