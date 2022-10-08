# этот файл должен находиться в одном месте с 'config.yaml'
from pathlib import Path

import matplotlib as plt
import yaml

CONFIG_FILE = Path(__file__).parent / 'config.yaml'
with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

VALID_ALGORITHM = ['HDBSCAN', 'DBSCAN', 'Agglomerative', 'Spectral', 'KMeans']
NOISE = -1
NOISE_NAME = 'Шум'
NOISE_COLOR = '#777676'

plt.rcParams.update({'figure.max_open_warning': 0})
