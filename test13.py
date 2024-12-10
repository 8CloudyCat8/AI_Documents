import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
import re
import pymorphy2
from nltk.corpus import stopwords
from collections import Counter

morph = pymorphy2.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_text(text):
    words = re.sub(r'[^\w\s]', '', text).lower().split()
    nouns = [morph.parse(word)[0].normal_form for word in words if
             len(word) >= 3 and 'NOUN' in morph.parse(word)[0].tag and morph.parse(word)[0].normal_form not in stop_words]

    return ' '.join(word for word, count in Counter(nouns).items() if count >= 2)

def vectorize_text(text):
    sentences = [preprocess_text(sentence) for sentence in text.split(".") if sentence]
    print("Обработанные предложения:", *sentences, sep='\n')
    vectorizer = TfidfVectorizer(max_features=1000)
    return vectorizer.fit_transform(sentences), sentences, vectorizer

def cluster_text(tfidf_matrix, n_clusters=10):
    return AgglomerativeClustering(n_clusters=min(n_clusters, tfidf_matrix.shape[0]), metric='euclidean',
                                   linkage='ward').fit_predict(tfidf_matrix.toarray())


def build_and_visualize_decision_tree(sentences, clusters, vectorizer):
    X_train, X_test, y_train, y_test = train_test_split(sentences, clusters, test_size=0.3, random_state=42)
    X_train_tfidf = vectorizer.transform(X_train)

    tree = DecisionTreeClassifier(random_state=42).fit(X_train_tfidf.toarray(), y_train)

    plt.figure(figsize=(20, 10))
    plot_tree(tree, feature_names=vectorizer.get_feature_names_out(),
              class_names=[f"Cluster {i}" for i in np.unique(clusters)], filled=True, rounded=True)
    plt.title("Дерево решений для кластеров")
    plt.show()

    rules = export_text(tree, feature_names=vectorizer.get_feature_names_out())
    print("Правила, извлеченные из дерева решений:")
    print(rules)

def print_clusters(clusters, sentences):
    cluster_words = {i: [] for i in np.unique(clusters)}
    for cluster_id, sentence in zip(clusters, sentences):
        cluster_words[cluster_id].extend(sentence.split())
    for cluster_id, words in cluster_words.items():
        print(f"Класс {cluster_id}: {', '.join(set(words))}")

file_path = "text1.txt"
text = load_text(file_path)
tfidf_matrix, sentences, vectorizer = vectorize_text(text)
clusters = cluster_text(tfidf_matrix)
build_and_visualize_decision_tree(sentences, clusters, vectorizer)
print_clusters(clusters, sentences)
