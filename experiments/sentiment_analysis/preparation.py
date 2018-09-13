from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def data_preparation(dataset):
    #dataset = 'D:/data/sentiment classification simple/finegrained.txt'
    reviews = []
    all_txt = ''

    with open(dataset, 'r', encoding='utf-8') as r:
        for line in r:
            all_txt += line + '\n'
    all_txt = all_txt.replace('\r', '\n')

    reviews = all_txt.split('\n\n\n')
    all_labs = set()
    all_sents = []

    for r in reviews:
        sents = [l for l in r.split('\n') if len(l.strip()) > 0]
        sents = sents[2:]
        all_sents.extend(sents)
        labs = [s.split('\t')[0] for s in sents]

        for l in labs:
            all_labs.add(l)

    all_sents = [{'lab': l.split('\t')[0], 'sent': l.split('\t')[1]} for l in all_sents]
    print('all labels ', all_labs)

    df = pd.DataFrame(all_sents)
    df = df[df.lab.isin(['neu', 'neg', 'pos'])]
    print('whole data size {}'.format(len(df)))

    # train / test split
    data_x = df.sent.values
    data_y = df.lab.values

    tx, tsx, ty, tsy = train_test_split(data_x, data_y, test_size=0.1)

    df_train = pd.DataFrame()
    df_train['sent'] = tx
    df_train['lab'] = ty

    df_test = pd.DataFrame()
    df_test['sent'] = tsx
    df_test['lab'] = tsy

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    all_words = set()

    for sent in df_train.sent:
        for w in sent.split(' '):
            all_words.add(w)

    print('#of words: {}'.format(len(all_words)))

    return df_train, df_test


def make_embedding_weight(df_train, glove_vector_path):
    from keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_train.sent)

    word_index = tokenizer.word_index

    EMBEDDING_DIM = 50

    # load glove weight
    #glove_vector_path = 'D:/data/glove/glove.6B.{}d.txt'.format(EMBEDDING_DIM)

    w = {}

    with open(glove_vector_path, 'r', encoding='utf-8') as r:
        for l in r:
            s = l.split(' ')
            k, v = s[0], s[1:]

            w[k] = v

    embedding_weight = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        v = w.get(word)
        if v is not None:
            embedding_weight[i] = v
    print('embedding weight shape: {}'.format(embedding_weight.shape))

    return embedding_weight, tokenizer


def make_label_df(df):
    labels = pd.get_dummies(df.lab)
    print(labels.sum())

    return labels

