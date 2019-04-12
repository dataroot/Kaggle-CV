import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Embedding, Concatenate, Reshape


def Categorizer(x, emb_input_dims: list, emb_output_dims: list):
    """
    Creates multidimensional trainable embeddings for categorical features
    """
    n_embs = len(emb_input_dims)
    
    if n_embs > 0:
        embs = []
        
        for i, nunique, dim in zip(range(n_embs), emb_input_dims, emb_output_dims):
            tmp = Lambda(lambda x: x[:, i])(x)
            embs.append(Embedding(nunique, dim)(tmp))
        
        embs.append(Lambda(lambda x: x[:, n_embs:])(x))
        x = Concatenate()(embs)
    
    return x


def Encoder(x, middle_dim: int, output_dim: int, emb_input_dims: list, emb_output_dims: list):
    """
   Computes latent space representation of x
    """
    x = Categorizer(x, emb_input_dims, emb_output_dims)
    
    x = Dense(middle_dim, activation='tanh')(x)
    x = Dense(output_dim)(x)
    return x


def Extractor(x, mask: np.ndarray):
    """
    Extract columns whose indices are in or outside date_mask
    """
    parts = []
    
    for i, in_mask in enumerate(mask):
        if in_mask:
            parts.append( Lambda(lambda x: x[:, i:(i+1)])(x) )
    
    x = Concatenate()(parts)
    return x


class ContentModel(Model):
    """
    The model with Encoder-based architecture
    """
    
    def __init__(self,
                 que_dim: int, que_date_mask: np.ndarray, que_input_embs: list, que_output_embs: list,
                 pro_dim: int, pro_date_mask: np.ndarray, pro_input_embs: list, pro_output_embs: list,
                 middle_dim: int, latent_dim: int):
        super().__init__()

        que_inputs = Input((que_dim, ))
        pro_inputs = Input((pro_dim, ))

        que_features = Extractor(que_inputs, ~que_date_mask)
        pro_features = Extractor(pro_inputs, ~pro_date_mask)

        que_encoded = Encoder(que_features, middle_dim, latent_dim, que_input_embs, que_output_embs)
        pro_encoded = Encoder(pro_features, middle_dim, latent_dim, pro_input_embs, pro_output_embs)

        dist = Lambda(lambda x: tf.reduce_sum(tf.square(x[0]-x[1]), axis = -1)) \
            ([que_encoded, pro_encoded])

        outputs = Lambda(lambda x: tf.reshape(tf.exp(-x), (-1, 1)))(dist)

        super().__init__([que_inputs, pro_inputs], outputs)


class DateModel(Model):
    """
    The model with Encoder-based architecture
    """
    
    def __init__(self,
                 que_dim: int, que_date_mask: np.ndarray,
                 pro_dim: int, pro_date_mask: np.ndarray,
                 middle_dim: int, latent_dim: int):
        super().__init__()
        
        que_inputs = Input((que_dim, ))
        pro_inputs = Input((pro_dim, ))
        
        que_features = Extractor(que_inputs, que_date_mask)
        pro_features = Extractor(pro_inputs, pro_date_mask)
        
        que_encoded = Encoder(que_features, middle_dim, latent_dim, [], [])
        pro_encoded = Encoder(pro_features, middle_dim, latent_dim, [], [])
        
        dist = Lambda(lambda x: tf.reduce_sum(tf.square(x[0]-x[1]), axis = -1)) \
            ([que_encoded, pro_encoded])
        
        outputs = Lambda(lambda x: tf.reshape(tf.exp(-x), (-1, 1)))(dist)
        
        super().__init__([que_inputs, pro_inputs], outputs)


class DoubleModel(Model):
    """
    The model with Encoder-based architecture
    """
    
    def __init__(self,
                 que_dim: int, que_date_mask: np.ndarray, que_input_embs: list, que_output_embs: list,
                 pro_dim: int, pro_date_mask: np.ndarray, pro_input_embs: list, pro_output_embs: list,
                 content_middle_dim: int, content_latent_dim: int,
                 date_middle_dim: int, date_latent_dim: int):
        super().__init__()
        
        que_inputs = Input((que_dim, ))
        pro_inputs = Input((pro_dim, ))
        
        self.content_model = ContentModel(
            que_dim, que_date_mask, que_input_embs, que_output_embs,
            pro_dim, pro_date_mask, pro_input_embs, pro_output_embs,
            content_middle_dim, content_latent_dim
        )
        self.date_model = DateModel(
            que_dim, que_date_mask,
            pro_dim, pro_date_mask,
            date_middle_dim, date_latent_dim
        )
        
        content_score = self.content_model([que_inputs, pro_inputs])
        date_score = self.date_model([que_inputs, pro_inputs])
        
        score = Concatenate()([content_score, date_score])
        
        # Compute F1 score
        F1_score = Lambda(lambda x: (x[:, 0] + x[:, 1]) / 2)(score)
        F1_score = Reshape((1, ))(F1_score)
        
        super().__init__([que_inputs, pro_inputs], F1_score)


class SimpleModel(Model):
    """
    The model with Simple architecture
    """
    
    def __init__(self,
                 que_dim: int, que_input_embs: list, que_output_embs: list,
                 pro_dim: int, pro_input_embs: list, pro_output_embs: list):
        super().__init__()
        
        que_inputs = Input((que_dim, ))
        pro_inputs = Input((pro_dim, ))
        
        que_features = Categorizer(que_inputs, que_input_embs, que_output_embs)
        pro_features = Categorizer(pro_inputs, pro_input_embs, pro_output_embs)
        
        x = Concatenate()([que_features, pro_features])
        
        x = Dense(30, activation='tanh')(x)
        x = Dense(15, activation='tanh')(x)
        
        outputs = Dense(1, activation='sigmoid')(x)
        
        super().__init__([que_inputs, pro_inputs], outputs)


class EncoderModel(Model):
    """
    The model with Encoder-based architecture that concatenates que and pro latent-space representations
    and passes them through several Dense layers.
    """
    
    def __init__(self,
                 que_dim: int, que_input_embs: list, que_output_embs: list,
                 pro_dim: int, pro_input_embs: list, pro_output_embs: list,
                 middle_dim: int, latent_dim: int):
        super().__init__()
        
        que_inputs = Input((que_dim, ))
        pro_inputs = Input((pro_dim, ))
        
        que_encoded = Encoder(que_inputs, que_dim, middle_dim, latent_dim, que_input_embs, que_output_embs)
        pro_encoded = Encoder(pro_inputs, pro_dim, middle_dim, latent_dim, pro_input_embs, pro_output_embs)
        
        x = Concatenate()([que_encoded, pro_encoded])
        
        x = Dense(10, activation='tanh')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        super().__init__([que_inputs, pro_inputs], outputs)

