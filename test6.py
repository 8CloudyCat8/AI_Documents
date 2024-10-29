import sys
import numpy as np
import re
from Stemmer import Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def text_cleaner(text):
    text = text.lower()
    stemmer = Stemmer('russian')
    text = ' '.join(stemmer.stemWords(text.split()))
    text = re.sub(r'\b\d+\b', ' digit ', text)
    return text

def load_data():
    data = {'text': [], 'tag': []}
    for line in open('model.txt', encoding='utf-8'):
        if not ('#' in line):
            row = line.strip().split("@")
            data['text'].append(row[0].strip())
            data['tag'].append(row[1].strip())
    return data

def train_test_split(data, validation_split=0.1):
    sz = len(data['text'])
    indices = np.arange(sz)
    np.random.shuffle(indices)

    X = [data['text'][i] for i in indices]
    Y = [data['tag'][i] for i in indices]
    nb_validation_samples = int(validation_split * sz)

    return {
        'train': {'x': X[:-nb_validation_samples], 'y': Y[:-nb_validation_samples]},
        'test': {'x': X[-nb_validation_samples:], 'y': Y[-nb_validation_samples:]}
    }

def openai():
    data = load_data()
    D = train_test_split(data)
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=text_cleaner)),
        ('clf', LogisticRegression()),
    ])
    text_clf.fit(D['train']['x'], D['train']['y'])

    z = input("Введите предложение без знака вопроса на конце: ")
    zz = [z]
    predicted = text_clf.predict(zz)
    predicted_proba = text_clf.predict_proba(zz)

    classes = text_clf.classes_

    print(f"Предсказанная категория: {predicted[0]}")
    print("Вероятности для каждой категории:")
    for i, label in enumerate(classes):
        print(f"{label}: {predicted_proba[0][i] * 100:.2f}%")

if __name__ == '__main__':
    sys.exit(openai())
