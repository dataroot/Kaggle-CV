import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Embedding, Concatenate
from keras import backend as K


class Encoder(Model):
    def __init__(self, input_dim: int, output_dim: int, emb_input_dims: list, emb_output_dims: list):
        inputs = Input((input_dim, ))
        n_embs = len(emb_input_dims)
        
        if n_embs > 0:
            embs = []
            for i, nunique, dim in zip(range(n_embs), emb_input_dims, emb_output_dims):
                tmp = Lambda(lambda x: x[:, i])(inputs)
                embs.append(Embedding(nunique, dim)(tmp))
            embs.append(Lambda(lambda x: x[:, n_embs:])(inputs))
            self.inter = Concatenate()(embs)
        else:
            self.inter = inputs
        
        x = self.inter
#         x = Dense(15, activation='tanh')(x)
        
        outputs = Dense(output_dim)(x)
        super().__init__(inputs, outputs)


class Categorizer(Model):
    def __init__(self, input_dim: int, emb_input_dims: list, emb_output_dims: list):
        inputs = Input((input_dim, ))
        n_embs = len(emb_input_dims)
        
        if n_embs > 0:
            embs = []
            for i, nunique, dim in zip(range(n_embs), emb_input_dims, emb_output_dims):
                tmp = Lambda(lambda x: x[:, i])(inputs)
                embs.append(Embedding(nunique, dim)(tmp))
            embs.append(Lambda(lambda x: x[:, n_embs:])(inputs))
            outputs = Concatenate()(embs)
        else:
            outputs = inputs
        
        super().__init__(inputs, outputs)


class Mothership_v1(Model):
    """
    The model with Encoder-based architecture
    """
    
    def __init__(self,
                 que_dim: int, que_input_embs: list, que_output_embs: list,
                 pro_dim: int, pro_input_embs: list, pro_output_embs: list,
                 inter_dim: int = 10):
        super().__init__()
        
        self.que_model = Encoder(que_dim, inter_dim, que_input_embs, que_output_embs)
        self.pro_model = Encoder(pro_dim, inter_dim, pro_input_embs, pro_output_embs)
        
        self.merged = Lambda(lambda x: tf.reduce_sum(tf.square(x[0]-x[1]), axis = -1)) \
            ([self.que_model.outputs[0], self.pro_model.outputs[0]])
        
        outputs = Lambda(lambda x: tf.reshape(tf.exp(-self.merged), (-1, 1)))(self.merged)
        
        super().__init__([self.que_model.inputs[0], self.pro_model.inputs[0]], outputs)


class Mothership_v2(Model):
    """
    The model with non-Encoder-based architecture
    """
    
    def __init__(self,
                 que_dim: int, que_input_embs: list, que_output_embs: list,
                 pro_dim: int, pro_input_embs: list, pro_output_embs: list):
        super().__init__()
        
        que_inputs = Input((que_dim, ))
        pro_inputs = Input((pro_dim, ))
        
        x = Concatenate()([que_inputs, pro_inputs])
        
        x = Dense(20, activation='tanh')(x)
        self.latent_vector = x
        
#         x = Dense(10, activation='tanh')(x)
#         wide_part = Concatenate()([que_inputs, pro_inputs])
#         x = Concatenate()([x, wide_part])
        
        outputs = Dense(1, activation='sigmoid')(x)
        
        super().__init__([que_inputs, pro_inputs], outputs)


class Mothership_v3(Model):
    """
    The model which pretrains another model (Mothership_v2) on embeddings
    """
    
    def __init__(self,
                 que_emb_dim: int, que_stat_dim: int, que_input_embs: list, que_output_embs: list,
                 pro_emb_dim: int, pro_stat_dim: int, pro_input_embs: list, pro_output_embs: list,
                 inter_dim: int = 0):
        super().__init__()
        
        que_stat_inputs = Input((que_stat_dim, ))
        pro_stat_inputs = Input((pro_stat_dim, ))
        
        self.embedding_model = Mothership_v2(que_emb_dim, que_input_embs, que_output_embs,
                                             pro_emb_dim, pro_input_embs, pro_output_embs)
        latent_vector = self.embedding_model.latent_vector
        
        que_stat = Categorizer(que_stat_dim, que_input_embs, que_output_embs)(que_stat_inputs)
        pro_stat = Categorizer(pro_stat_dim, pro_input_embs, pro_output_embs)(pro_stat_inputs)
        
        x = Concatenate()([latent_vector, que_stat, pro_stat])
        x = Dense(20, activation='tanh')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        super().__init__([
            self.embedding_model.inputs[0], self.embedding_model.inputs[1],
            que_stat_inputs, pro_stat_inputs
        ], outputs)


class Mothership_v4(Model):
    """
    The model with non-Encoder-based architecture that works with embeddings and statistical features from the beginning
    """
    
    def __init__(self,
                 que_emb_dim: int, que_stat_dim: int, que_input_embs: list, que_output_embs: list,
                 pro_emb_dim: int, pro_stat_dim: int, pro_input_embs: list, pro_output_embs: list):
        super().__init__()
        
        que_emb_inputs = Input((que_emb_dim, ))
        pro_emb_inputs = Input((pro_emb_dim, ))
        que_stat_inputs = Input((que_stat_dim, ))
        pro_stat_inputs = Input((pro_stat_dim, ))
        
        self.que_stat = Categorizer(que_stat_dim, que_input_embs, que_output_embs)(que_stat_inputs)
        self.pro_stat = Categorizer(pro_stat_dim, pro_input_embs, pro_output_embs)(pro_stat_inputs)
        
        x = Concatenate()([que_emb_inputs, pro_emb_inputs, self.que_stat, self.pro_stat])
        
        x = Dense(30, activation='tanh')(x)
        x = Dense(15, activation='tanh')(x)
        
        outputs = Dense(1, activation='sigmoid')(x)
        
        super().__init__([que_emb_inputs, pro_emb_inputs, que_stat_inputs, pro_stat_inputs], outputs)

