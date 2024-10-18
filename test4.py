import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

data = pd.read_csv('topics.csv', header=None, names=['sentence', 'label'])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['sentence'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, zero_division=1))

new_sentence = ["салат"]
new_vector = vectorizer.transform(new_sentence)
prediction = model.predict(new_vector)
print(f"Предложение: {new_sentence[0]}, Тематика: {prediction[0]}")
