from clayers import Attention
import keras.backend as K
import keras


class Attended_LSTM(keras.Model):
    def __init__(self, seq_dim, out_dim, vocab_size, embedding_weight, a_func, m_func):
        super(Attended_LSTM, self).__init__()
        EMBEDDING_DIM = embedding_weight.shape[1]

        self.embedding_lookup = keras.layers.Embedding(vocab_size + 1, EMBEDDING_DIM, weights=[embedding_weight],
                                                       trainable=False)
        self.seq_model = keras.layers.LSTM(seq_dim,
                                           return_sequences=True, return_state=True,
                                           recurrent_initializer=keras.initializers.he_uniform())

        self.attention_model = Attention(attend_func=a_func, merge_func=m_func)

        self.simple_dense = keras.layers.Dense(seq_dim, kernel_initializer=keras.initializers.he_uniform())
        self.relu_activation = keras.layers.Activation('relu')
        self.dropout = keras.layers.Dropout(rate=0.3)
        self.simple_dense2 = keras.layers.Dense(out_dim, kernel_initializer=keras.initializers.he_uniform())
        self.sm_activation = keras.layers.Activation('softmax')

    def call(self, inputs, **kwargs):
        embedding = self.embedding_lookup(inputs)

        context, hs, cs = self.seq_model(embedding)
        attended_state = self.attention_model([hs, context])
        h = self.simple_dense(attended_state)
        d1 = self.relu_activation(h)
        do = self.dropout(d1)
        h2 = self.simple_dense2(do)
        out = self.sm_activation(h2)

        return out

class Attended_GRU(keras.Model):
    def __init__(self, seq_dim, out_dim, vocab_size, embedding_weight, a_func, m_func):
        super(Attended_GRU, self).__init__()
        EMBEDDING_DIM = embedding_weight.shape[1]

        self.embedding_lookup = keras.layers.Embedding(vocab_size + 1, EMBEDDING_DIM, weights=[embedding_weight],
                                                       trainable=False)
        self.seq_model = keras.layers.GRU(seq_dim,
                                           return_sequences=True, return_state=True,
                                           recurrent_initializer=keras.initializers.he_uniform())

        self.attention_model = Attention(attend_func=a_func, merge_func=m_func)

        self.simple_dense = keras.layers.Dense(seq_dim, kernel_initializer=keras.initializers.he_uniform())
        self.relu_activation = keras.layers.Activation('relu')
        self.dropout = keras.layers.Dropout(rate=0.3)
        self.simple_dense2 = keras.layers.Dense(out_dim, kernel_initializer=keras.initializers.he_uniform())
        self.sm_activation = keras.layers.Activation('softmax')

    def call(self, inputs, **kwargs):
        embedding = self.embedding_lookup(inputs)

        context, hs = self.seq_model(embedding)
        attended_state = self.attention_model([hs, context])
        h = self.simple_dense(attended_state)
        d1 = self.relu_activation(h)
        do = self.dropout(d1)
        h2 = self.simple_dense2(do)
        out = self.sm_activation(h2)

        return out


class Attended_BiGRU(keras.Model):
    def __init__(self, seq_dim, vocab_size, out_dim, embedding_weight, a_func, m_func):
        super(Attended_BiGRU, self).__init__()
        EMBEDDING_DIM = embedding_weight.shape[1]

        self.embedding_lookup = keras.layers.Embedding(vocab_size + 1, EMBEDDING_DIM, weights=[embedding_weight],
                                                       trainable=False)
        self.masking = keras.layers.Masking(mask_value=0)

        self.seq_model = keras.layers.GRU(seq_dim,
                                          return_sequences=True, return_state=True,
                                          recurrent_initializer=keras.initializers.he_uniform())
        self.concat = keras.layers.Concatenate()

        self.attention_model = Attention(attend_func=a_func, merge_func=m_func)

        self.simple_dense = keras.layers.Dense(seq_dim, kernel_initializer=keras.initializers.he_uniform())
        self.relu_activation = keras.layers.Activation('relu')
        self.dropout = keras.layers.Dropout(rate=0.3)
        self.simple_dense2 = keras.layers.Dense(out_dim, kernel_initializer=keras.initializers.he_uniform())
        self.sm_activation = keras.layers.Activation('softmax')

    def call(self, inputs, **kwargs):
        fw_seq, bw_seq = inputs

        fw_embedding = self.embedding_lookup(fw_seq)
        bw_embedding = self.embedding_lookup(bw_seq)

        # fw_masked_embedding = self.masking(fw_embedding)
        # bw_masked_embedding = self.masking(bw_embedding)

        f_ctx, f_hs = self.seq_model(fw_embedding)
        b_ctx, b_hs = self.seq_model(bw_embedding)

        # context = K.concatenate([f_ctx, b_ctx])
        context = self.concat([f_ctx, b_ctx])
        hs = K.concatenate([f_hs, b_hs])

        attended_state = self.attention_model([hs, context])
        h = self.simple_dense(attended_state)
        d1 = self.relu_activation(h)
        do = self.dropout(d1)
        h2 = self.simple_dense2(do)
        out = self.sm_activation(h2)

        return out


class Attended_BiLSTM(keras.Model):
    def __init__(self, seq_dim, out_dim, vocab_size, embedding_weight, a_func, m_func):
        super(Attended_BiLSTM, self).__init__()
        EMBEDDING_DIM = embedding_weight.shape[1]

        self.embedding_lookup = keras.layers.Embedding(vocab_size + 1, EMBEDDING_DIM, weights=[embedding_weight],
                                                       trainable=False)
        self.masking = keras.layers.Masking(mask_value=0)

        self.seq_model = keras.layers.LSTM(seq_dim,
                                           return_sequences=True, return_state=True,
                                           recurrent_initializer=keras.initializers.he_uniform())
        self.concat = keras.layers.Concatenate()

        self.attention_model = Attention(attend_func=a_func, merge_func=m_func)

        self.simple_dense = keras.layers.Dense(seq_dim, kernel_initializer=keras.initializers.he_uniform())
        self.relu_activation = keras.layers.Activation('relu')
        self.dropout = keras.layers.Dropout(rate=0.3)
        self.simple_dense2 = keras.layers.Dense(out_dim, kernel_initializer=keras.initializers.he_uniform())
        self.sm_activation = keras.layers.Activation('softmax')

    def call(self, inputs, **kwargs):
        fw_seq, bw_seq = inputs

        fw_embedding = self.embedding_lookup(fw_seq)
        bw_embedding = self.embedding_lookup(bw_seq)

        # fw_masked_embedding = self.masking(fw_embedding)
        # bw_masked_embedding = self.masking(bw_embedding)

        f_ctx, f_hs, f_cs = self.seq_model(fw_embedding)
        b_ctx, b_hs, b_cs = self.seq_model(bw_embedding)

        # context = K.concatenate([f_ctx, b_ctx])
        context = self.concat([f_ctx, b_ctx])
        hs = K.concatenate([f_hs, b_hs])
        cs = K.concatenate([f_cs, b_cs])

        attended_state = self.attention_model([hs, context])
        h = self.simple_dense(attended_state)
        d1 = self.relu_activation(h)
        do = self.dropout(d1)
        h2 = self.simple_dense2(do)
        out = self.sm_activation(h2)

        return out