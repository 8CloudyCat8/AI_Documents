import sys
import time

import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, QComboBox
)
from rutermextract import TermExtractor
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline

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
    term_extractor = TermExtractor()
    keywords = [(term.normalized, term.count) for term in term_extractor(text)]
    return keywords


def get_embedding(model_name, text):
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
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(embeddings)
    return kmeans.labels_


class SimilarityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Keyword Similarity Calculator')

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
        clustered_embeddings = {}
        for idx, (keyword, _) in enumerate(keywords):
            if labels[idx] not in clustered_keywords:
                clustered_keywords[labels[idx]] = []
                clustered_embeddings[labels[idx]] = []
            clustered_keywords[labels[idx]].append(keyword)
            clustered_embeddings[labels[idx]].append(embeddings[idx])

        end_time = time.time()
        elapsed_time = end_time - start_time

        result_strings = []
        for cluster, words in clustered_keywords.items():
            result_strings.append(f"Кластер {cluster}: {', '.join(words)}")
            result_strings.append(f"Эмбеддинги: {clustered_embeddings[cluster]}")

        results_text = '\n'.join(result_strings)

        self.result_label.setText(f"Результаты кластеризации:\n{results_text}")
        self.final_result_label.setText(f"\nВремя вычисления: {elapsed_time:.4f} секунд")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SimilarityApp()
    ex.resize(400, 400)
    ex.show()
    sys.exit(app.exec_())
