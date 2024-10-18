import json
import sys
import pymorphy2
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QLineEdit
from rutermextract import TermExtractor
from docx import Document

morph = pymorphy2.MorphAnalyzer()


def extract_terms(text):
    term_extractor = TermExtractor()
    terms = term_extractor(text)

    keyword_counts = {}
    for term in terms:
        normalized_term = term.normalized
        keyword_counts[normalized_term] = keyword_counts.get(normalized_term, 0) + 1
        for word in normalized_term.split():
            keyword_counts[word] = keyword_counts.get(word, 0) + 1

    return keyword_counts


def save_keywords_to_file(keywords, filename="topic_keywords.json"):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    for topic, counts in keywords.items():
        if topic in data:
            for keyword, count in counts.items():
                data[topic][keyword] = data[topic].get(keyword, 0) + count
        else:
            data[topic] = counts

    for topic in data:
        data[topic] = dict(sorted(data[topic].items(), key=lambda item: item[1], reverse=True))

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)


def load_keywords(filename="topic_keywords.json"):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def compare_with_topics(sentence, keywords_data):
    sentence_keywords = extract_terms(sentence)
    results = {}

    for topic, topic_keywords in keywords_data.items():
        matches = set(sentence_keywords.keys()).intersection(set(topic_keywords))
        match_percentage = (len(matches) / len(set(topic_keywords))) * 100 if len(set(topic_keywords)) > 0 else 0
        results[topic] = match_percentage

    return results


class TextClassifierApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Классификация текста")
        self.setGeometry(100, 100, 600, 500)

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        self.setPalette(palette)

        layout = QVBoxLayout()

        self.label = QLabel("Выберите файл .docx, введите тему или предложение:")
        layout.addWidget(self.label)

        self.topic_input = QLineEdit(self)
        self.topic_input.setPlaceholderText("Введите тему или предложение")
        self.topic_input.setStyleSheet("background-color: #2A2A2A; color: white;")
        layout.addWidget(self.topic_input)

        self.file_button = QPushButton("Выбрать файл")
        self.file_button.setStyleSheet("background-color: #3E3E3E; color: white;")
        self.file_button.clicked.connect(self.select_file)
        layout.addWidget(self.file_button)

        self.file_label = QLabel("Файл не выбран")
        layout.addWidget(self.file_label)

        self.save_button = QPushButton("Сохранить ключевые слова")
        self.save_button.setStyleSheet("background-color: #3E3E3E; color: white;")
        self.save_button.clicked.connect(self.save_keywords)
        layout.addWidget(self.save_button)

        self.compare_button = QPushButton("Сравнить предложение с темами")
        self.compare_button.setStyleSheet("background-color: #3E3E3E; color: white;")
        self.compare_button.clicked.connect(self.compare_sentence)
        layout.addWidget(self.compare_button)

        self.result_label = QLabel("")
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def select_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "", "Документы Word (*.docx)",
                                                   options=options)
        if file_path:
            self.file_path = file_path
            self.file_label.setText(f"Выбран файл: {file_path.split('/')[-1]}")
        else:
            self.file_label.setText("Файл не выбран")
            self.file_path = None

    def save_keywords(self):
        topic = self.topic_input.text()
        if not topic:
            self.result_label.setText("Введите тему!")
            return

        if not hasattr(self, 'file_path') or not self.file_path:
            self.result_label.setText("Выберите файл!")
            return

        text = extract_text_from_docx(self.file_path)

        keywords = extract_terms(text)

        save_keywords_to_file({topic: keywords})

        self.result_label.setText(f"Ключевые слова для темы '{topic}' сохранены в topic_keywords.json")

    def compare_sentence(self):
        sentence = self.topic_input.text()
        if not sentence:
            self.result_label.setText("Введите предложение для сравнения!")
            return
        keywords_data = load_keywords()

        if not keywords_data:
            self.result_label.setText("Нет данных для сравнения. Сохраните ключевые слова для тем.")
            return

        comparison_results = compare_with_topics(sentence, keywords_data)

        result_text = "Результаты сравнения:\n"
        for topic, percentage in comparison_results.items():
            result_text += f"{topic}: {percentage:.2f}% совпадений\n"

        self.result_label.setText(result_text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    classifier = TextClassifierApp()
    classifier.show()
    sys.exit(app.exec_())
