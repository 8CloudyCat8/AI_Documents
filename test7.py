import sys
import re
import pandas as pd
import nltk
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from nltk.corpus import stopwords
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton

habrParse_df = pd.read_csv('Parse.csv')

nltk.download("stopwords")
russian_stopwords = stopwords.words("russian")


def remove_punct(text):
    return re.sub(r'[^\w\s]', '', text)


habrParse_df['Post_clean'] = habrParse_df['Post'].map(lambda x: x.lower())
habrParse_df['Post_clean'] = habrParse_df['Post_clean'].map(remove_punct)
habrParse_df['Post_clean'] = habrParse_df['Post_clean'].map(
    lambda x: ' '.join([word for word in x.split() if word not in russian_stopwords]))

X_train = habrParse_df['Post_clean']
y_train = habrParse_df['hubs']

sgd_ppl_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('sgd_clf', SGDClassifier(random_state=42))
])

sgd_ppl_clf.fit(X_train, y_train)


class TextClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel("Введите текст для классификации:")
        layout.addWidget(self.label)

        self.text_input = QLineEdit(self)
        layout.addWidget(self.text_input)

        self.button = QPushButton("Предсказать класс", self)
        self.button.clicked.connect(self.predict_class)
        layout.addWidget(self.button)

        self.result_label = QLabel("Результат будет здесь")
        layout.addWidget(self.result_label)

        self.label_class = QLabel("Введите класс для извлечения слов:")
        layout.addWidget(self.label_class)

        self.class_input = QLineEdit(self)
        layout.addWidget(self.class_input)

        self.button_extract = QPushButton("Извлечь слова", self)
        self.button_extract.clicked.connect(self.extract_words)
        layout.addWidget(self.button_extract)

        self.extract_result_label = QLabel("Извлеченные слова будут здесь")
        layout.addWidget(self.extract_result_label)

        self.setLayout(layout)
        self.setWindowTitle('Классификатор текстов')
        self.show()

    def predict_class(self):
        text = self.text_input.text()
        if text:
            cleaned_text = remove_punct(text.lower())
            cleaned_text = ' '.join([word for word in cleaned_text.split() if word not in russian_stopwords])
            prediction = sgd_ppl_clf.predict([cleaned_text])
            self.result_label.setText(f"Вероятный класс: {prediction[0]}")
        else:
            self.result_label.setText("Введите текст для классификации.")

    def extract_words(self):
        class_name = self.class_input.text()
        if class_name:
            relevant_posts = habrParse_df[habrParse_df['hubs'] == class_name]['Post_clean']
            words = ' '.join(relevant_posts).split()
            unique_words = set(words)
            self.extract_result_label.setText(f"Слова для класса '{class_name}': {', '.join(unique_words)}")
        else:
            self.extract_result_label.setText("Введите класс для извлечения слов.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TextClassifierApp()
    sys.exit(app.exec_())
