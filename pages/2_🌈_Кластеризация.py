import streamlit as st

from loader import VALID_ALGORITHM
from utils.snippet import plot_streamlit_clusters

header = st.container()
body = st.container()

with header:
    st.title('Кластеризация')


with body:
    if "result" not in st.session_state:
        st.write('Сначала получите данные')
    else:
        result = st.session_state.get('result')
        st.markdown('Во вкладке *Данные* представлены исходные данные для кластеризации. '
                    'Далее следуют результаты кластеризации по каждому из алгоритмов:\n'
                    '+ Трёхмерная визуализация (Точка=Комментарий);\n'
                    '+ Диаграмма размера кластеров (Показывает количество комментариев '
                    'для каждого кластерв);\n'
                    '+ Облако слов для каждого кластера (Показывает наиболее частые '
                    'слова кластера).\n'
                    )

        # Результаты кластеризаци
        tabs_names = ['Данные'] + VALID_ALGORITHM
        tabs = st.tabs(tabs_names)
        with st.spinner("Формируются результаты..."):
            for tab, tab_name in zip(tabs, tabs_names):
                with tab:
                    try:
                        plot_streamlit_clusters(result, tab_name)
                    except Exception as e:
                        st.error(f'Ошибка! {e.args}')
            st.balloons()
