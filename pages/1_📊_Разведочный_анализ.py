import streamlit as st

from utils.eda import plot_tokens_distribution, plot_token_count

header = st.container()
body = st.container()

with header:
    st.title('Разведочный анализ данных')

with body:
    if "result" not in st.session_state:
        st.write('Сначала получите данные')
    else:
        result = st.session_state.get('result')
        st.markdown('Краткая статистика по очищенным данным.')
        # Рассчитываем метрики
        count_tokens, count_tokens_fig = plot_token_count(result.cleaned_corpus)
        tokens_dist_fig = plot_tokens_distribution(result.cleaned_corpus)

        # изображаем метрики
        col1, col2 = st.columns(2)
        col1.metric("Отфильтрованных комментариев",
                    len(result.cleaned_corpus),
                    delta=len(result.cleaned_corpus) - result.len_comments_corpus,
                    help="Разница отсчитывается от необработанных комментариев")
        col2.metric("Уникальных токенов", count_tokens)

        # рисуем графики
        st.header("Топ 50 частых токенов")
        st.plotly_chart(count_tokens_fig, use_container_width=True)

        st.header("Распределение комментариев по длине")
        st.plotly_chart(tokens_dist_fig, use_container_width=True)
