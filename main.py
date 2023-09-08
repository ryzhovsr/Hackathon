import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stopwords(text):
    # Разделяем текст на слова
    words = word_tokenize(text)

    # Загружаем список стоп-слов на английском языке
    stop_words = set(stopwords.words('english'))

    # Удаляем стоп-слова из списка слов
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Объединяем отфильтрованные слова обратно в текст
    filtered_text = ' '.join(filtered_words)

    # Возвращаем результат
    return filtered_text

# Загрузка данных
data = pd.read_excel('CRA_train_1200.xlsx')

# Применяем функцию remove_stopwords к столбцу pr_txt
data['pr_txt'] = data['pr_txt'].apply(remove_stopwords)

# Сохраняем изменения в датасете
data.to_csv('processed_dataset.csv', index=False)



# Предобработка данных
# ...

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(data['pr_txt'], data['Категория'], test_size=0.2)

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
