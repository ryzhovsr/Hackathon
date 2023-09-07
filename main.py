import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Загрузка данных
data = pd.read_csv('data.csv')

# Предобработка данных
# ...

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['rating'], test_size=0.2)

# Векторизация текста
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Обучение модели
model = SVC()
model.fit(X_train_vectorized, y_train)

# Прогнозирование на тестовой выборке
y_pred = model.predict(X_test_vectorized)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
