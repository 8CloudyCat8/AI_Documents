import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import re
import matplotlib.pyplot as plt

data = pd.read_csv('topics.csv', header=None, names=['text', 'label'])


def preprocess_text(text):
    text = re.sub(r'[^а-яА-ЯёЁ\s]', '', text).lower()
    return text


data['text'] = data['text'].apply(preprocess_text)

X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(TfidfVectorizer(), DecisionTreeClassifier())

model.fit(X_train, y_train)


def predict_class(text):
    text = preprocess_text(text)
    predicted_class = model.predict([text])
    return predicted_class[0]


user_input = input("Введите текст: ")
predicted_class = predict_class(user_input)
print(f"Предсказанный класс: {predicted_class}")

decision_tree = model.named_steps['decisiontreeclassifier']

plt.figure(figsize=(40, 20))
plot_tree(decision_tree, filled=True, feature_names=model.named_steps['tfidfvectorizer'].get_feature_names_out(),
          class_names=['1', '2', '3'])
