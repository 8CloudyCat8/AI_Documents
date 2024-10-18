from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

data = [
    ("Я люблю собак и кошек", "Животные"),
    ("Птицы летают высоко", "Животные"),
    ("Я люблю изучать программирование", "Программирование"),
    ("Python - это отличный язык", "Программирование"),
    ("Коты очень игривые", "Животные"),
    ("Технологии меняют мир", "Технологии"),
]

X_train, y_train = zip(*data)

model = make_pipeline(CountVectorizer(), DecisionTreeClassifier())
model.fit(X_train, y_train)


def extract_keywords(theme, sentence):
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)

    keywords = []
    for word, tag in pos_tags:
        if tag in ["NN", "NNS"]:
            if theme.lower() in word.lower():
                keywords.append(word)

    return keywords


theme = input("Введите тему (например, Животные): ")
sentence = input("Введите предложение: ")

keywords = extract_keywords(theme, sentence)

if keywords:
    print(f"Извлеченные слова по теме '{theme}': {', '.join(keywords)}")
else:
    print(f"Нет извлеченных слов по теме '{theme}'.")

predicted_class = model.predict([sentence])
print(f"Предсказанный класс для предложения '{sentence}': {predicted_class[0]}")
