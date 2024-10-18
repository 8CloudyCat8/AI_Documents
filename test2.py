import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel

data = {
    'text': [
        'Я люблю готовить пасту',
        'Футбол - это захватывающий вид спорта',
        'Инвестирование в акции - это умный шаг',
        'Пицца - это моя любимая еда',
        'На олимпиаде много спортсменов',
        'Создание стартапа может быть прибыльным',
        'Я пью кофе и ем торт',
        'Баскетбол - моя страсть',
        'Как запустить интернет-магазин',
        'Салат с овощами полезен',
        'У нас будет футбольный матч',
        'Секреты успешного бизнеса',
        'Я готовлю десерт',
        'Мы играем в теннис каждый уикенд',
        'Как открыть успешный бизнес',
    ],
    'class': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
}

df = pd.DataFrame(data)

X = df['text']
y = df['class']
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X, y)


class TextClassifierApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Классификатор текстов')
        self.setGeometry(100, 100, 400, 200)

        self.layout = QVBoxLayout()

        self.input_line = QLineEdit(self)
        self.input_line.setPlaceholderText('Введите текст...')
        self.layout.addWidget(self.input_line)

        self.predict_button = QPushButton('Классифицировать', self)
        self.predict_button.clicked.connect(self.predict)
        self.layout.addWidget(self.predict_button)

        self.result_label = QLabel('Результат: ', self)
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)

    def predict(self):
        input_text = self.input_line.text()
        if input_text:
            prediction = model.predict([input_text])[0]
            self.result_label.setText(f'Результат: класс {prediction}')
        else:
            self.result_label.setText('Введите текст для классификации.')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TextClassifierApp()
    window.show()
    sys.exit(app.exec_())
