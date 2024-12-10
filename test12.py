import re
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from gensim.models import FastText
from nltk.tokenize import word_tokenize
import numpy as np
import pymorphy2
from nltk.corpus import stopwords
from sklearn.tree import DecisionTreeClassifier, plot_tree

morph = pymorphy2.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))

params = {
    'fasttext': {
        'vector_size': 100,  # Размер векторов
        'window': 10,        # Размер окна
        'min_count': 3,      # Минимальное количество вхождений слова
        'workers': 4         # Количество потоков
    },
    'kmeans': {
        'max_clusters': 10    # Максимальное количество кластеров
    }
}

def prepare_data(text):
    text = re.sub(r'[^\w\s]', '', text)  # Удаляем знаки препинания
    tokens = word_tokenize(text.lower())  # Токенизация текста
    nouns = [token for token in tokens if morph.parse(token)[0].tag.POS == 'NOUN' and token not in stop_words]
    return nouns

def train_fasttext_model(tokens):
    model = FastText(sentences=[tokens],
                     vector_size=params['fasttext']['vector_size'],
                     window=params['fasttext']['window'],
                     min_count=params['fasttext']['min_count'],
                     workers=params['fasttext']['workers'])
    return model

def classify_words(model, clusters, tokens):
    word_vectors = np.array([model.wv[word] for word in tokens])
    clf = DecisionTreeClassifier()
    clf.fit(word_vectors, clusters)
    return clf, tokens

def visualize_tree_with_words(clf, words):
    plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=words, class_names=[str(i) for i in range(np.max(labels) + 1)])
    plt.title("Decision Tree Visualization with Words")
    plt.savefig('decision_tree_words.png', dpi=900)
    plt.close()

# Основной код
if __name__ == "__main__":
    option = input("Введите '1' для загрузки текста из файла или '2' для ввода текста вручную: ")

    if option == '1':
        file_path = input("Введите путь к текстовому файлу: ")
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                input_text = file.read()
        else:
            print("Файл не найден.")
            exit()
    elif option == '2':
        input_text = input("Введите текст: ")
    else:
        print("Неверный выбор.")
        exit()

    tokens = prepare_data(input_text)
    model = train_fasttext_model(tokens)

    num_clusters = min(params['kmeans']['max_clusters'], len(tokens))
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(np.array([model.wv[word] for word in tokens]))
    labels = kmeans.labels_

    clf, words = classify_words(model, labels, tokens)

    visualize_tree_with_words(clf, words)

    print("Дерево решений сохранено в 'decision_tree_words.png'.")
