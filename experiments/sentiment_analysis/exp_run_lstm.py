from models import *
from preparation import *

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


def training(df_train, model, tokenizer, verbose=0):
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=1E-3))

    #tokenizer = Tokenizer()
    sequences = tokenizer.texts_to_sequences(df_train.sent)

    maxlen = max([len(s) for s in sequences])
    word_sequences = sequence.pad_sequences(sequences, maxlen=maxlen)

    lab_stats = list(labels.sum())
    # lab_rat = [sum(lab_stats)/x for x in lab_stats]
    lab_rat = [x / sum(lab_stats) for x in lab_stats]

    history = model.fit(batch_size=200, epochs=50,
              x=word_sequences, y=labels.values,
              validation_split=0.1,
              class_weight=lab_rat,
              verbose=verbose)

    return history


def testing(df_test, model, tokenizer, maxlen):
    # test
    test_seq = tokenizer.texts_to_sequences(df_test.sent)
    test_word_seq = sequence.pad_sequences(test_seq, maxlen=maxlen)

    test_lab = pd.get_dummies(df_test.lab).values
    model.evaluate(x=test_word_seq, y=test_lab, verbose=1)
    res = model.predict(x=test_word_seq)

    df_test['predict'] = res.argmax(axis=1)
    df_test['answer'] = test_lab.argmax(axis=1)
    df_test['matched'] = df_test.apply(lambda r: r.predict == r.answer, axis=1)

    conf_mat = pd.crosstab(df_test.answer, df_test.predict, rownames=['answer'], colnames=['predict'])

    print(conf_mat)
    pass


if __name__ == '__main__':
    # training
    EMBEDDING_DIM = 50
    dataset = 'D:/data/sentiment classification simple/finegrained.txt'
    glove_vector_path = 'D:/data/glove/glove.6B.{}d.txt'.format(EMBEDDING_DIM)

    df_train, df_test = data_preparation(dataset)
    embedding_weight, tokenizer = make_embedding_weight(df_train, glove_vector_path)
    labels = make_label_df(df_train)

    a_funcs = ['affine', 'cosine']
    m_funcs = ['add', 'mul', 'concat']

    print('Single_direction LSTM with attention')
    for af in a_funcs:
        for mf in m_funcs:
            model = Attended_LSTM(a_func=af, m_func=mf,
                                  vocab_size=len(tokenizer.word_index),
                                  out_dim=labels.values.shape[1],
                                  seq_dim=50,
                                  embedding_weight=embedding_weight)
            history = training(df_train, model, tokenizer=tokenizer, verbose=0)

            df = pd.DataFrame(history.history)

            print('attending: {}, merge: {}, min_loss: {} at {}'.format(af, mf, min(df.val_loss), df.idxmin()['val_loss']))
