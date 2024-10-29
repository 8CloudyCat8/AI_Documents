import sys
import time
import torch
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QTextEdit, QPushButton, QLabel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline
from rutermextract import TermExtractor
from sklearn.decomposition import PCA


models = {
    "DistilBERT (Geotrend)": {
        "tokenizer": AutoTokenizer.from_pretrained("Geotrend/distilbert-base-ru-cased"),
        "model": AutoModel.from_pretrained("Geotrend/distilbert-base-ru-cased"),
        "type": "embedding"
    },
    "DistilBERT ZeroShot (AyoubChLin)": {
        "tokenizer": AutoTokenizer.from_pretrained("AyoubChLin/DistilBERT_ZeroShot"),
        "model": AutoModelForSequenceClassification.from_pretrained("AyoubChLin/DistilBERT_ZeroShot"),
        "type": "zero-shot"
    },
    "mDeBERTa-v3 (MoritzLaurer)": {
        "tokenizer": None,
        "model": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "type": "zero-shot"
    }
}

def extract_keywords(text):
    """

    :param text: Входной текст
    :return: Список кортежей (нормализованное слово, количество)
    """
    term_extractor = TermExtractor()
    keywords = [(term.normalized, term.count) for term in term_extractor(text)]
    return keywords

def get_embedding(model_name, text):
    """
    Получение эмбеддинга для заданного текста с использованием указанной модели.
    :param model_name: Название модели
    :param text: Входной текст
    :return: Эмбеддинг текста
    """
    model_info = models[model_name]

    if model_info["type"] == "embedding":
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]

        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[0][0].numpy()
    else:
        return None

def calculate_similarity(model_name, keyword, topics):
    """
    Вычисление семантического сходства между ключевым словом и темами.
    :param model_name: Название модели
    :param keyword: Ключевое слово
    :param topics: Список тем
    :return: Словарь с темами и их сходствами
    """
    keyword_embedding = get_embedding(model_name, keyword)
    similarities = {}

    if model_name in ["DistilBERT (Geotrend)"]:
        for topic in topics:
            topic_embedding = get_embedding(model_name, topic)
            similarity = cosine_similarity([keyword_embedding], [topic_embedding])[0][0]
            similarities[topic] = similarity
    else:
        classifier = pipeline("zero-shot-classification", model=models[model_name]["model"],
                              tokenizer=models[model_name]["tokenizer"])
        result = classifier(keyword, candidate_labels=topics)
        for topic, score in zip(result['labels'], result['scores']):
            similarities[topic] = score

    return similarities

def cluster_embeddings(embeddings, num_clusters):
    """
    Кластеризация эмбеддингов с использованием KMeans.
    :param embeddings: Эмбеддинги для кластеризации
    :param num_clusters: Количество кластеров
    :return: Метки кластеров
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(embeddings)
    return kmeans.labels_

def visualize_embeddings(embeddings):
    """
    Визуализация эмбеддингов с использованием PCA.
    :param embeddings: Эмбеддинги для визуализации
    """
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], marker='o')
    plt.title('2D Visualization of Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid()
    plt.show()

def classify_texts(embeddings, labels):
    """
    Классификация текстов с использованием логистической регрессии.
    :param embeddings: Эмбеддинги для классификации
    :param labels: Метки классов
    :return: Обученная модель
    """
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(embeddings, labels)
    return model

class SimilarityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Keyword Similarity and Analysis Tool')

        layout = QVBoxLayout()

        self.model_selection = QComboBox(self)
        self.model_selection.addItems(models.keys())
        layout.addWidget(self.model_selection)

        self.text_input = QTextEdit(self)
        self.text_input.setPlaceholderText('Введите текст для извлечения ключевых слов')
        layout.addWidget(self.text_input)

        self.topics_input = QTextEdit(self)
        self.topics_input.setPlaceholderText('Введите классы (каждый класс на новой строке)')
        layout.addWidget(self.topics_input)

        self.calculate_button = QPushButton('Вычислить сходство и кластеризацию', self)
        self.calculate_button.clicked.connect(self.calculate_similarity_and_clustering)
        layout.addWidget(self.calculate_button)

        self.result_label = QLabel('Результаты:')
        layout.addWidget(self.result_label)

        self.final_result_label = QLabel('')
        layout.addWidget(self.final_result_label)

        self.setLayout(layout)

    def calculate_similarity_and_clustering(self):
        text = self.text_input.toPlainText()
        topics = self.topics_input.toPlainText().splitlines()

        if not text or not topics:
            self.result_label.setText("Пожалуйста, введите текст и классы.")
            return

        start_time = time.time()

        selected_model = self.model_selection.currentText()
        keywords = extract_keywords(text)

        embeddings = []
        for keyword, _ in keywords:
            embedding = get_embedding(selected_model, keyword)
            if embedding is not None:
                embeddings.append(embedding)

        num_clusters = min(5, len(embeddings))
        labels = cluster_embeddings(embeddings, num_clusters)

        clustered_keywords = {}
        for idx, (keyword, _) in enumerate(keywords):
            if labels[idx] not in clustered_keywords:
                clustered_keywords[labels[idx]] = []
            clustered_keywords[labels[idx]].append(keyword)

        visualize_embeddings(embeddings)

        example_labels = [1 if "кошка" in k else 0 for k, _ in keywords]
        classifier = classify_texts(embeddings, example_labels)

        end_time = time.time()
        elapsed_time = end_time - start_time

        result_strings = []
        for cluster, words in clustered_keywords.items():
            result_strings.append(f"Кластер {cluster}: {', '.join(words)}")

        results_text = '\n'.join(result_strings)

        self.result_label.setText(f"Результаты кластеризации:\n{results_text}")
        self.final_result_label.setText(f"\nВремя вычисления: {elapsed_time:.4f} секунд")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SimilarityApp()
    ex.resize(600, 600)
    ex.show()
    sys.exit(app.exec_())