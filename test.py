import pymorphy2
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

morph = pymorphy2.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))

def prepare_text(text):
    return [
        morph.parse(word)[0].normal_form
        for word in simple_preprocess(text)
        if word not in stop_words and not word.isdigit()
    ]

# Основная функция
def main():
    input_text = """стажировки рисунок парсер таблица редактор результаты объявление archimate личный кабинет стажёра учебная практика такой образ стажёр работа парсинга объявления инструмент задача доступ вакансия функциональные требования сбор данных наша команда база данных управление страница стажировка ссылка система"""

    corpus = prepare_text(input_text)
    print("Массив слов:", corpus)

    model = Word2Vec([corpus], vector_size=100, min_count=1)

    target_word = input("Введите слово для поиска схожих слов: ").strip().lower()
    target_word_normalized = morph.parse(target_word)[0].normal_form

    if target_word_normalized in model.wv.key_to_index:
        print(f"Найдены слова, схожие со словом '{target_word}':")
        similar_words = model.wv.most_similar(target_word_normalized, topn=100)
        for i, (word, similarity) in enumerate(similar_words, start=1):
            print(f"{i}) {word} (сходство: {similarity:.2f})")
    else:
        print(f"Слово '{target_word}' не найдено в тексте.")

if __name__ == '__main__':
    main()
