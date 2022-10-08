import streamlit as st

from utils.snippet import get_topics

header = st.container()
body = st.container()
status = st.empty()

with header:
    st.title("Главная")
    st.markdown('Проект по кластеризации комментариев из YouTube. Комментарии извлекаются '
                'из видео-ролика, переданного по URL, проходят предобработку, анализируются и подаются '
                'в модели кластеризации. Для каждой модели выводятся результаты в виде '
                'трёхмерной визуализации, количественного распределения комментариев по кластерам, '
                'облака слов.')
    st.header("Поиск комментариев")
    with st.expander("Нюансы:"):
        st.markdown("+ Комментарии на иностранном языке игнорируются;\n"
                    "+ Извлекаются только открытые комментарии на YouTube;\n"
                    "+ Ответы на комментарии не учитываются;\n"
                    "+ Для обогащения данных, если комментариев мало укажите "
                    "дополнительное количество видео. Видео будут извлекаться с того же канала, "
                    "что и переданное по URL видео;\n"
                    "+ Для уменьшения данных, если комментариев много укажите макимальное "
                    "число комментариев;\n"
                    "+ Качество и производительность алгоритмов напрямую зависят от указаннго количества "
                    "видео и комментариев;\n"
                    "+ *Данные нигде не сохраняются, поэтому при обновлении страниц, данные теряются.")

    # форма для поиска
    with st.form(key='youtube-url'):
        st.markdown("Параметры для поиска")
        url = st.text_input("URL видеоролика с YouTube",
                            placeholder='https://www.youtube.com/watch?v=TpXVcVnR3vo',
                            )

        n_videos = st.number_input("Количество дополнительных видео",
                                   min_value=0,
                                   help="0 - означает использовать только переданное по URL видео")
        max_comments = st.number_input("Максимальное количество комментариев ",
                                       min_value=1000,
                                       help="Устанавливаем предел по количеству комментариев под каждым видео, "
                                            "которые будут в дальнейшем обрабатываться")

        submitted = st.form_submit_button("Поехали")

    # обработка нажатия кнопки
    if submitted:
        # st.markdown(f"URL: {url}\n\nn_videos={n_videos}\n\nmax_comments={max_comments}")
        try:
            status.warning('Обработка может занять несколько минут... Не уходите с этой страницы...')
            result = get_topics(url, max_comments, n_videos, st.secrets["youtube-api-key"])
            st.balloons()
            st.session_state['result'] = result
            status.success('Моделирование успешно завершено!')
        except Exception as e:
            status.error(f'Ошибка! {e.args}')
