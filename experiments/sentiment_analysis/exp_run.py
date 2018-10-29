# run all experiments by giving arguments to model Class
from models import *
from preparation import *

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


def training(df_train, model, tokenizer, verbose=0):
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=1E-3))

    #tokenizer = Tokenizer()
    sequences = tokenizer.texts_to_sequences(df_train.sent)

    maxlen = max([len(s) for s in sequences])
    fw_word_sequences = sequence.pad_sequences(sequences, maxlen=maxlen)
    inputs = fw_word_sequences

    if model.is_bidirectional:
        rev_sequences = [list(reversed(s)) for s in sequences]
        bw_word_sequences = sequence.pad_sequences(rev_sequences, maxlen=maxlen)
        inputs = [fw_word_sequences, bw_word_sequences]

    lab_stats = list(labels.sum())
    # lab_rat = [sum(lab_stats)/x for x in lab_stats]
    lab_rat = [x / sum(lab_stats) for x in lab_stats]

    history = model.fit(batch_size=200, epochs=50,
                        x=inputs, y=labels.values,
                        validation_split=0.1,
                        class_weight=lab_rat,
                        verbose=verbose)

    return history


def testing(df_test, model, tokenizer, maxlen):
    # test
    test_seq = tokenizer.texts_to_sequences(df_test.sent)
    fw_test_word_seq = sequence.pad_sequences(test_seq, maxlen=maxlen)
    inputs = fw_test_word_seq

    if model.is_bidirectional:
        rev_test_seq = [list(reversed(s)) for s in test_seq]
        bw_test_word_seq = sequence.pad_sequences(rev_test_seq, maxlen=maxlen)
        inputs = [fw_test_word_seq, bw_test_word_seq]

    test_lab = pd.get_dummies(df_test.lab).values

    model.evaluate(x=inputs, y=test_lab, verbose=1)

    res = model.predict(x=inputs)

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

    units = ['BiLSTM', 'GRU', 'LSTM', 'BiGRU']
    a_funcs = ['affine', 'cosine']
    m_funcs = ['add', 'mul', 'concat']
    contexts = ['word', 'state']
    num_layers = [1, 2, 3]

    df_train, df_test = data_preparation(dataset)
    embedding_weight, tokenizer = make_embedding_weight(df_train, glove_vector_path)
    labels = make_label_df(df_train)

    n_layer = 1
    ctx = 'state'
    units = ['BiLSTM']
    for unit in units:
        print('{} with attention'.format(unit))
        for af in a_funcs:
            for mf in m_funcs:
                model = Attended_RNN_ematt_tmp(a_func=af, m_func=mf, rnn_unit=unit,
                                           vocab_size=len(tokenizer.word_index),
                                           out_dim=labels.values.shape[1],
                                           attending_context=ctx,
                                           seq_dim=EMBEDDING_DIM,
                                           embedding_weight=embedding_weight,
                                           embedding_train=False,
                                           layers=n_layer
                                           )
                history = training(df_train, model, tokenizer=tokenizer, verbose=0)

                df = pd.DataFrame(history.history)
                string_template = 'unit: {}, layers: {}, context: {}, attending: {}, merge: {}, min_loss: {} at {}'
                print(string_template.format(unit,
                                             n_layer,
                                             ctx,
                                             af, mf,
                                             min(df.val_loss),
                                             df.idxmin()['val_loss']))

                model = None
