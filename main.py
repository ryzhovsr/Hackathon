from transformers import BertTokenizer, BertForSequenceClassification
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import random
import torch
import seaborn as sns
import re
import nltk
import pymorphy2


def delete_symbols(release):
    new_release = re.sub(r'[,.;:»«_()@+=><%#–!"-]', ' ', release)
    # new_release = re.sub(r"[^а-яА-Я]+"," ",release, flags=re.UNICODE)
    return new_release


def delete_stop_words(release_words):
    # print('Удаляю стоп слова...')
    # print('Перевожу в нижний регистр...')
    stop_words = list(stopwords.words('russian'))
    word_tokens = word_tokenize(release_words)
    new_release = []
    for w in word_tokens:
        word = w.lower()
        if word not in stop_words and word != "\n":
            new_release.append(word)
    return new_release


def seed_all(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


train_df = pd.read_excel('CRA_train_1200.xlsx', engine='openpyxl', index_col=0)
test_df = pd.read_excel('CRA_train_1200.xlsx', engine='openpyxl', index_col=0)

sns.countplot(data=train_df, x="Категория")
sns.countplot(data=train_df, x="Уровень рейтинга")

train_text = train_df['pr_txt'].astype('str')
train_labels7 = train_df['Категория']
train_labels17 = train_df['Уровень рейтинга']

test_text = train_df['pr_txt'].astype('str')
test_labels7 = train_df['Категория']
test_labels17 = train_df['Уровень рейтинга']

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

morph = pymorphy2.MorphAnalyzer()

train_text1 = train_text.apply(lambda x: delete_symbols(x))
train_text1 = train_text1.apply(lambda x: nltk.word_tokenize(x, language="russian"))

for word in train_text1[1]:
    # p = morph.parse(word)[0]
    print(morph.parse(word)[0].normal_form)

train_text2 = train_text1.apply(lambda x: [morph.parse(word)[0].normal_form for word in x])

train_text2[1]

seed_all(42)

model = BertForSequenceClassification.from_pretrained('rubert_base_cased_sentence/', num_labels=2).to("cuda")
tokenizer = BertTokenizer.from_pretrained('rubert_base_cased_sentence/')

seq_len_train = [len(str(i).split()) for i in train_df['pr_txt']]
seq_len_test = [len(str(i).split()) for i in test_df['pr_txt']]
max_seq_len = max(max(seq_len_test), max(seq_len_train))
