from typing import List, Optional

import numpy as np
from gensim.models import FastText


class Emdedder:
    def __init__(self):
        self.model_tokens = None

    def text2embedding(self, corpus: List[List[str]], model: FastText) -> np.array:
        """
        Трансформация сырого токенизированного текста в эмбединг
        :param corpus: список токенизированных тестов
        :param model: FastText обученная модель
        :return: матрица эмбедингов текстов
        """
        n_col, n_row = model.vector_size, len(corpus)
        embeddings = np.zeros(shape=(n_row, n_col))
        for i, sent in enumerate(corpus):
            vectors = []
            for word in sent:
                try:
                    vectors.append(model.wv[word])
                except KeyError:
                    continue
            if vectors:
                embeddings[i] = np.mean(np.array(vectors), axis=0)
        return embeddings

    def get_embeddings_ft(self,
                          corpus: List[List[str]],
                          model: Optional[FastText] = None,
                          ft_params: Optional[dict] = None,
                          epochs: int = 40) -> np.array:
        """
        Обучение модели FastText и преобразование корпуса
        в векторное представление на обученных эмбедингах слов
        :param corpus: список токенизированных тестов
        :param model: предобученная модель FastText
        :param ft_params: парметры для создания модели FastText
        :param epochs: количество эпох обучения
        :return: матрица эмбедингов текстов
        """

        # создаем или выбираем сохраненную модель
        if model is not None:
            update = True
        else:
            update = False
            model = FastText(**ft_params)

        # тренеруем модель
        model.build_vocab(corpus, update=update)
        model.train(corpus, total_examples=model.corpus_count, epochs=epochs)

        # сохраняем список уникальных токенов модели
        self.model_tokens = model.wv.index_to_key
        return self.text2embedding(corpus, model)
