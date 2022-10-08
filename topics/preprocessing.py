
import re
import string
from typing import List

from nltk.corpus import stopwords
from pymystem3 import Mystem
from stop_words import get_stop_words


class TextPreprocessor:
    def __init__(self, stopwords: bool = True, min_symbols: int = 2, min_words: int = 3) -> None:
        """
        :param stopwords: флаг - учитывать или нет стоп-слова
        :param min_symbols: минимум символов для валидного токена
        :param min_words:  минимум токенов для валидного текста
        """
        self.min_symbols = min_symbols
        self.min_words = min_words
        # self.morph = pymorphy2.MorphAnalyzer()
        self.mystem = Mystem()

        if stopwords:
            self.set_stopwords()
        else:
            self.russian_stopwords = set()

    def set_stopwords(self) -> None:
        """
        Устанавливает стоп-слова
        """
        stop_words1 = set(get_stop_words('ru'))
        stop_words2 = set(stopwords.words('russian'))
        stop_words3 = set(stopwords.words('english'))
        self.russian_stopwords = stop_words1.union(stop_words2).union(stop_words3)

    def lowercase(self, text: str) -> str:
        """
        Приводит все символы в нижний регистр
        """
        return text.lower()

    def remove_emojis(self, text: str) -> str:
        """
        Удаляет смайлы по коду юникода
        """
        emoj = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          u"\U00002500-\U00002BEF"  # chinese char
                          u"\U00002702-\U000027B0"
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          u"\U0001f926-\U0001f937"
                          u"\U00010000-\U0010ffff"
                          u"\u2640-\u2642"
                          u"\u2600-\u2B55"
                          u"\u200d"
                          u"\u23cf"
                          u"\u23e9"
                          u"\u231a"
                          u"\ufe0f"  # dingbats
                          u"\u3030"
                          "]+", re.UNICODE)
        return re.sub(emoj, ' ', text)

    def remove_html(self, text: str) -> str:
        """
        Удаление html кода
        """
        pattern = re.compile('<.*?>')
        return pattern.sub(' ', text)

    def remove_urls(self, text: str) -> str:
        """
        Удаление некоторых ссылок
        """
        re_urls = ['https?://\S+', 'www\.\S+', 'bit.ly/\S+']
        url_pattern = re.compile(r'|'.join(re_urls))
        return url_pattern.sub(' ', text)

    def remove_spec_symbols(self, text: str) -> str:
        """
        Удаление спецсимволов
        """
        pattern = re.compile(r'&\S+?;')
        return pattern.sub(' ', text)

    def remove_files(self, text: str) -> str:
        """
        Удаление некоторых файлов
        """
        pattern = re.compile(r'\S+\.(?:csv|xlsx|py|ipynb|pdf|zip|rar|docx)')
        return pattern.sub(' ', text)

    def remove_punctuation(self, text: str) -> str:
        """
        Удаление знаков пунктуации
        """
        pattern = re.compile(fr'[{string.punctuation}]')
        return pattern.sub(' ', text)

    def remove_digits(self, text: str) -> str:
        """
        Удаление цифр
        """
        pattern = re.compile(r'\d+')
        return pattern.sub(' ', text)

    def only_ru_en_chars(self, text: str) -> str:
        """
        Дополнительная очистка от небуквенных символов.
        Казалось бы почему сразу не применить эту функцию,
        но тогда останутся внутренности html, ссылок и подобные вещи
        """
        pattern = re.compile(r'[^a-zа-яё ]')
        return pattern.sub(' ', text)

    def remove_all_non_russian(self, text: str) -> str:
        """
        Обнуляет строки, состоящие полностью на не русском языке
        """
        pattern = re.compile(r'[а-яё]')
        return text if re.search(pattern, text) else ''

    def remove_whitespace(self, text: str) -> str:
        """
        Удаляет лишние пробелы
        """
        pattern = re.compile(r'\s+')
        return pattern.sub(' ', text)

    def tokenizer(self, text: str) -> List[str]:
        """
        Токенизация теста
        :param text: текст
        :return: список токенов
        """
        return text.split()

    def lemmatization(self, list_text: List[str]) -> List[str]:
        """
        Лемматизирует слова
        :param list_text: список текстов
        :return: список текстов со словами в нормальной форме
        """
        # соединяем все тексты в один большой
        text = " # ".join(list_text)
        # лемматизируем с pymystem
        lemmas = self.mystem.lemmatize(text)
        # возвращаем текст в исходный вид
        return ''.join(lemmas).split(' # ')

    def remove_stopwords(self, token_list: List[str]) -> List[str]:
        """
        Удаляет стоп-слова из списка токенов.
        :param token_list: список токенов
        :return: список токенов без стоп-слов
        """
        return [word for word in token_list if word not in self.russian_stopwords]

    def text_filter(self, tokenized_text: List[List[str]]) -> List[List[str]]:
        """
        Фильтрует тексты по минимальному количеству токенов и символов и удаляет дубликаты.
        :param tokenized_text: список токенизированных текстов
        :return: писок токенизированных, отфильтрованных текстов
        """
        # фильтрация по минимальному количеству символов (букв)
        filtered_tokenized_text = [[word for word in text if len(word) >= self.min_symbols]
                                   for text in tokenized_text]

        # фильтрация по минимальному количеству слов в комментарие
        filtered_tokenized_text = [text for text in filtered_tokenized_text if len(text) >= self.min_words]

        # Удаление дубликатов строк
        filtered_tokenized_text = set([' '.join(text) for text in filtered_tokenized_text])
        filtered_tokenized_text = list(map(str.split, filtered_tokenized_text))
        return filtered_tokenized_text

    def preprocessing(self, text: str) -> str:
        """
        Выполняеет основную предобработку строк
        """
        text = self.lowercase(text)
        text = self.remove_emojis(text)
        text = self.remove_html(text)
        text = self.remove_urls(text)
        text = self.remove_spec_symbols(text)
        text = self.remove_files(text)
        text = self.remove_punctuation(text)
        text = self.remove_digits(text)
        text = self.only_ru_en_chars(text)
        text = self.remove_all_non_russian(text)
        text = self.remove_whitespace(text)
        return text

    def get_clean_text(self, text_corpus: List[str]) -> List[List[str]]:
        """
        Главная функция для очистки и предобработки текста
        :param text_corpus: список сырых текстов
        :return: список токенизированных, очищенных текстов
        """
        # Основная предобработка тектса
        preproces_text = [self.preprocessing(text) for text in text_corpus]
        # лемматизируем
        lemma_text = self.lemmatization(preproces_text)
        # токенизируем
        tokenized_text = [self.tokenizer(text) for text in lemma_text]
        # удаляем стоп слова
        tokenized_text = [self.remove_stopwords(tokens) for tokens in tokenized_text]
        # фильтруем по количеству символов\слов, дубликаты
        clean_text = self.text_filter(tokenized_text)
        return clean_text
