from collections import Counter
from typing import List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.tool import remove_outlier_sigma


@st.cache(max_entries=10, ttl=3600)
def plot_token_count(filtered_tokenized_text: List[List[str]]) -> Tuple[int, go.Figure]:
    """
    Создаёт график количества популярных токенов
    :param filtered_tokenized_text: очищенный, токенизированный текст
    :return: figure plotly
    """
    count = Counter(sum(filtered_tokenized_text, []))
    df_count = pd.DataFrame(count.most_common(50), columns=['Токен', 'Частота']).reset_index()
    fig = px.bar(df_count.sort_values('Частота'),
                 y='Токен',
                 x='Частота',
                 color='index',
                 color_continuous_scale='jet_r',
                 hover_name='Токен',
                 hover_data={'Токен': False, 'index': False}
                 )
    fig.update_layout(title_x=0.5,
                      # title_text='Топ 50 частых токенов',
                      height=900)
    fig.update_coloraxes(showscale=False)
    return len(count), fig


@st.cache(max_entries=10, ttl=3600)
def plot_tokens_distribution(filtered_tokenized_text: List[List[str]]) -> go.Figure:
    """
    Создаёт график распределения комментариев по длине
    :param filtered_tokenized_text: очищенный, токенизированный текст
    :return: figure plotly
    """
    df_tokens = pd.DataFrame({'tokens': filtered_tokenized_text})
    df_tokens['len'] = df_tokens['tokens'].apply(len)

    df_sigma = remove_outlier_sigma(df_tokens, 'len')

    fig = px.histogram(df_sigma.sort_values('len'), x="len",
                       marginal="box")
    fig.update_layout(xaxis_title='Длина комментария (количество токенов)',
                      yaxis_title='Количество комментариев',
                      title_x=0.5)
    fig.update_xaxes(type='category')
    return fig
