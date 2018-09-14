from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer


class Attention(Layer):
    def __init__(self, attend_func, merge_func, **kwargs):
        self.attend_func = attend_func
        self.merge_func = merge_func
        self.out = None
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shapes):
        input_shape, context_shape = input_shapes

        if self.attend_func == 'cosine':
            self.attend_func = self.cosine_attend
        elif self.attend_func == 'affine':
            self.attend_func = self.affine_attend
            self.attend_w = self.add_weight('attned_w', (input_shape[1], context_shape[1]),
                                            initializer=initializers.get('glorot_uniform'))
            self.attend_b = self.add_weight('attned_b', (context_shape[1],),
                                            initializer=initializers.get('glorot_uniform'))
        else:
            raise Exception('Invalid attending function:', self.attend_func)

        if self.merge_func == 'add':
            self.merge_func = self.add_merge
        elif self.merge_func == 'concat':
            self.merge_func = self.concat_merge
        elif self.merge_func == 'mul':
            self.merge_func = self.mul_merge

        pass

    def call(self, inputs, **kwargs):
        i, c = inputs
        attention_weight = self.attend_func(i, c)
        # a = K.permute_dimensions(K.permute_dimensions(attention_weight, (1,0)) * K.permute_dimensions(c, (2,1,0)), (2,1,0))
        # print(attention_weight.shape)
        # print(c.shape)
        a = K.permute_dimensions(attention_weight * K.permute_dimensions(c, (2, 0, 1)), (1, 2, 0))
        a = K.sum(a, axis=1)

        self.out = self.merge_func(i, a)

        return self.out

    def compute_output_shape(self, input_shape):

        return K.int_shape(self.out)

    def cosine_attend(self, x, c):
        xn = K.l2_normalize(x, axis=-1)
        cn = K.l2_normalize(c, axis=-1)
        # print(xn.shape) # (?, input_dim)
        # print(cn.shape) # (?, context_dim, input_dim)

        ip = K.permute_dimensions(xn * K.permute_dimensions(cn, (1, 0, 2)), (1, 0, 2))
        cos = K.sum(ip, axis=-1)

        return 1 - cos

    def affine_attend(self, x, c):
        attention_vec = K.dot(x, self.attend_w) + self.attend_b

        return attention_vec

    def add_merge(self, i, a):
        return i + a

    def mul_merge(self, i, a):
        return K.prod([i, a], axis=0)

    def concat_merge(self, i, a):
        return K.concatenate([i, a], axis=-1)


class SimpleMemoryCell(Layer):
    def __init__(self, mem_size, **kwargs):
        self.mem_size = mem_size

        super(SimpleMemoryCell, self).__init__(**kwargs)

    def build(self, input_shapes):

        pass

    def call(self, inputs, **kwargs):

        pass

    def compute_output_shape(self, input_shape):

        pass