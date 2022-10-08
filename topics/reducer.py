import numpy as np
import umap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Reducer:
    def __init__(self):
        pass

    def umap_transform(self, data: np.array, umap_params: dict) -> np.array:
        """
        Снижение размерности алгоритмом UMAP
        :param data: данные, у которых понизить размерность
        :param umap_params: параметры модели UMAP
        :return: преобразованные данные
        """
        pipe_umap = make_pipeline(StandardScaler(),
                                  umap.UMAP(**umap_params)
                                  )
        return pipe_umap.fit_transform(data)
