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


class Attended_RNN_ematt(keras.Model):
    def __init__(self,
                 seq_dim, out_dim, vocab_size,
                 rnn_unit,
                 embedding_weight, embedding_train,
                 attending_context,
                 a_func, m_func,
                 layers=1):
        super(Attended_RNN_ematt, self).__init__()
        EMBEDDING_DIM = embedding_weight.shape[1]

        self.rnn_layers = layers
        self.rnn_unit = rnn_unit
        self.attending_context = attending_context

        self.embedding_lookup = keras.layers.Embedding(vocab_size + 1, EMBEDDING_DIM, weights=[embedding_weight],
                                                       trainable=embedding_train)
        self.seq_model = []
        for i in range(self.rnn_layers):
            if self.rnn_unit == 'LSTM' or self.rnn_unit == 'BiLSTM':
                self.seq_model.append(keras.layers.LSTM(seq_dim,
                                                        return_sequences=True, return_state=True,
                                                        recurrent_initializer=keras.initializers.he_uniform()))
            elif self.rnn_unit == 'GRU' or self.rnn_unit == 'BiGRU':
                self.seq_model.append(keras.layers.GRU(seq_dim,
                                                       return_sequences=True, return_state=True,
                                                       recurrent_initializer=keras.initializers.he_uniform()))
            else:
                raise Exception('{} not in [LSTM, GRU, BiLSTM, BiGRU]'.format(rnn_unit))
        self.is_bidirectional = (self.rnn_unit[0:2] == 'Bi')
        if self.is_bidirectional:
            self.concat = keras.layers.Concatenate()
        elif self.rnn_unit not in ('LSTM', 'GRU'):
            raise Exception('{} not in [LSTM, GRU, BiLSTM, BiGRU]'.format(rnn_unit))

        self.attention_model = Attention(attend_func=a_func, merge_func=m_func)

        self.simple_dense = keras.layers.Dense(seq_dim, kernel_initializer=keras.initializers.he_uniform())
        self.relu_activation = keras.layers.Activation('relu')
        self.dropout = keras.layers.Dropout(rate=0.3)
        self.simple_dense2 = keras.layers.Dense(out_dim, kernel_initializer=keras.initializers.he_uniform())
        self.sm_activation = keras.layers.Activation('softmax')

    def call(self, inputs, **kwargs):
        if self.is_bidirectional:
            fw_seq, bw_seq = inputs
        else:
            fw_seq = inputs
            bw_seq = None

        fw_embedding = self.embedding_lookup(fw_seq)
        fw_ctx = fw_embedding
        bw_ctx = None

        all_embedding = fw_embedding
        if self.is_bidirectional:
            bw_embedding = self.embedding_lookup(bw_seq)
            bw_ctx = bw_embedding
            # all_embedding = self.concat([fw_embedding, bw_embedding])
            all_embedding = K.concatenate([fw_embedding, bw_embedding])

        ctx = None
        hs = None
        for a_seq_model in self.seq_model:
            if self.rnn_unit == 'LSTM':
                fw_ctx, fw_hs, fw_cs = a_seq_model(fw_ctx)
                hs = fw_hs
                ctx = fw_ctx
            elif self.rnn_unit == 'GRU':
                fw_ctx, fw_hs = a_seq_model(fw_ctx)
                hs = fw_hs
                ctx = fw_ctx
            elif self.rnn_unit == 'BiLSTM':
                fw_ctx, fw_hs, fw_cs = a_seq_model(fw_ctx)
                bw_ctx, bw_hs, bw_cs = a_seq_model(bw_ctx)
                ctx = self.concat([fw_ctx, bw_ctx])
                # ctx = K.concatenate([fw_ctx, bw_ctx])
                fw_ctx = ctx
                bw_ctx = K.reverse(fw_ctx, axes=1)
                # hs = self.concat([fw_hs, bw_hs])
                hs = K.concatenate([fw_hs, bw_hs])
            elif self.rnn_unit == 'BiGRU':
                fw_ctx, fw_hs = a_seq_model(fw_ctx)
                bw_ctx, bw_hs = a_seq_model(bw_ctx)
                ctx = self.concat([fw_ctx, bw_ctx])
                # ctx = K.concatenate([fw_ctx, bw_ctx])
                fw_ctx = ctx
                bw_ctx = K.reverse(fw_ctx, axes=1)
                # hs = self.concat([fw_hs, bw_hs])
                hs = K.concatenate([fw_hs, bw_hs])
            else:
                raise Exception('not initialized!: seq_model {}'.format(self.rnn_unit))

        if self.attending_context == 'word':
            attended_state = self.attention_model([hs, all_embedding])
        elif self.attending_context == 'state':
            attended_state = self.attention_model([hs, ctx])
        else:
            raise Exception('invalid attending context: {}'.format(self.attending_context))

        h = self.simple_dense(attended_state)
        d1 = self.relu_activation(h)
        do = self.dropout(d1)
        h2 = self.simple_dense2(do)
        out = self.sm_activation(h2)

        return out





class Attended_RNN_ematt_tmp(keras.Model):
    def __init__(self,
                 seq_dim, out_dim, vocab_size,
                 rnn_unit,
                 embedding_weight, embedding_train,
                 attending_context,
                 a_func, m_func,
                 layers=1):
        super(Attended_RNN_ematt_tmp, self).__init__()
        EMBEDDING_DIM = embedding_weight.shape[1]

        self.rnn_layers = layers
        self.rnn_unit = rnn_unit
        self.attending_context = attending_context

        self.embedding_lookup = keras.layers.Embedding(vocab_size + 1, EMBEDDING_DIM, weights=[embedding_weight],
                                                       trainable=embedding_train)
        self.seq_model = []
        self.seq_model.append(keras.layers.LSTM(seq_dim,
                                                return_sequences=True, return_state=True,
                                                recurrent_initializer=keras.initializers.he_uniform()))

        self.is_bidirectional = (self.rnn_unit[0:2] == 'Bi')
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
        fw_ctx = fw_embedding

        bw_embedding = self.embedding_lookup(bw_seq)
        bw_ctx = bw_embedding

        ctx = None
        hs = None
        for a_seq_model in self.seq_model:
            fw_ctx, fw_hs, fw_cs = a_seq_model(fw_ctx)
            bw_ctx, bw_hs, bw_cs = a_seq_model(bw_ctx)
            ctx = self.concat([fw_ctx, bw_ctx])
            # ctx = K.concatenate([fw_ctx, bw_ctx])
            fw_ctx = ctx
            bw_ctx = K.reverse(fw_ctx, axes=1)
            hs = self.concat([fw_hs, bw_hs])
            # hs = K.concatenate([fw_hs, bw_hs])

        attended_state = self.attention_model([hs, ctx])

        h = self.simple_dense(attended_state)
        d1 = self.relu_activation(h)
        do = self.dropout(d1)
        h2 = self.simple_dense2(do)
        out = self.sm_activation(h2)

        return out

