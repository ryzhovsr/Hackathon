import pandas as pd
import numpy as np
import random
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from transformers import BertForSequenceClassification, BertTokenizer
from transformers import TrainingArguments
from transformers import Trainer

import transformers
import accelerate

var = transformers.__version__, accelerate.__version__

train_df = pd.read_excel('/content/CRA_train_1200.xlsx', engine='openpyxl', index_col=0)

train_text, test_text, train_labels, test_labels = train_test_split(train_df['pr_txt'].astype('str'),
                                                                    train_df['Категория'].astype('str'), test_size=0.1,
                                                                    random_state=36)


def seed_all(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


seed_all(42)

# Модель 'DeepPavlov/rubert-base-cased'  Внимание: 7 или 17 выбрать
model_name = 'DeepPavlov/rubert-base-cased'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=7)  # 7!!!!
tokenizer = BertTokenizer.from_pretrained(model_name)

tokens_train = tokenizer.batch_encode_plus(
    train_text.values,
    max_length=512,
    padding='max_length',
    truncation=True
)

tokens_test = tokenizer.batch_encode_plus(
    test_text.values,
    max_length=512,
    padding='max_length',
    truncation=True
)

# Создание и обучение кодировщика на тренировочных метках
label_encoder = LabelEncoder()
label_encoder.fit(train_labels)

# Преобразование тренировочных и тестовых меток в целочисленные значения
train_labels_encoded = label_encoder.transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Получение соответствия между исходными метками и их целочисленными значениями
label_mapping = {label: value for label, value in
                 zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}


# Оборачиваем токенизированные текстовые данные в torch Dataset:
class Data(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


# train_dataset = Data(tokens_train, train_labels)
# test_dataset = Data(tokens_test, test_labels)

train_dataset = Data(tokens_train, train_labels_encoded)
test_dataset = Data(tokens_test, test_labels_encoded)


# Расчет метрики - F1

def compute_metrics(predict):
    labels = predict.label_ids
    preds = predict.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='micro')
    # one of [None, 'micro', 'macro', 'weighted']
    return {'F1': f1}


# Параметры для обучения:
training_args = TrainingArguments(
    output_dir='./results',  # Выходной каталог
    num_train_epochs=20,  # Кол-во эпох для обучения
    per_device_train_batch_size=12,  # Размер пакета для каждого устройства во время обучения
    per_device_eval_batch_size=12,  # Размер пакета для каждого устройства во время валидации
    weight_decay=0.01,  # Понижение весов
    logging_dir='./logs',  # Каталог для хранения журналов
    load_best_model_at_end=True,  # Загружать ли лучшую модель после обучения
    learning_rate=1e-5,  # Скорость обучения
    evaluation_strategy='epoch',  # Валидация после каждой эпохи (можно сделать после конкретного кол-ва шагов)
    logging_strategy='epoch',  # Логирование после каждой эпохи
    save_strategy='epoch',  # Сохранение после каждой эпохи
    save_total_limit=1,
    seed=42)

# Передача в trainer предообученной модели, tokenizer, данных для обучения, для валидации и способа расчета метрики
trainer = Trainer(model=model,
                  tokenizer=tokenizer,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=test_dataset,
                  compute_metrics=compute_metrics)

# Запуск обучения модели
trainer.train()

# Сохранение обученной модели
model_path = "fine-tune-bert"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)


# Функция для получения предсказания
def get_prediction():
    test_predict = trainer.predict(test_dataset)
    labels = np.argmax(test_predict.predictions, axis=-1)
    return labels


predict = get_prediction()

# Проверка полученного результата

# Оценки качества модели
print(classification_report(test_labels_encoded, predict))
print(f1_score(test_labels_encoded, predict, average='micro'))
