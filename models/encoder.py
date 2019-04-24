import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Embedding, Concatenate


def l2_reg_last_n(alpha: float, n: int):
    """
    Adds L2 regularization on weights connected with the last n features with multiplier alpha
    """
    return lambda w: alpha * tf.reduce_mean(tf.square(w[-n:, :]))


def categorize(inputs: tf.Tensor, emb_input_dims: list, emb_output_dims: list):
    """
    Replaces categorical features with trainable embeddings

    :param inputs: tensor with encoded categorical features in first columns
    :param emb_input_dims: number of unique classes in categorical features
    :param emb_output_dims: embedding dimensions of categorical features
    :return: transformed tensor
    """
    n_embs = len(emb_input_dims)

    if n_embs > 0:
        embs = []

        # iterate over categorical features
        for i, nunique, dim in zip(range(n_embs), emb_input_dims, emb_output_dims):
            # separate their values with Lambda layer
            tmp = Lambda(lambda x: x[:, i])(inputs)
            # pass them through Embedding layer
            embs.append(Embedding(nunique, dim)(tmp))

        # pass all the numerical features directly
        embs.append(Lambda(lambda x: x[:, n_embs:])(inputs))
        # and concatenate them
        outputs = Concatenate()(embs)
    else:
        outputs = inputs

    return outputs


class Encoder(Model):
    """
    Model for extraction of high-level feature vector from question or professional
    """

    def __init__(self, input_dim: int, inter_dim: int, output_dim: int, emb_input_dims: list, emb_output_dims: list,
                 reg: float = 0.0):
        """
        :param input_dim: dimension of raw feature vector
        :param inter_dim: dimension of intermediate layer
        :param output_dim: dimension of computed high-level feature vector
        :param emb_input_dims: number of unique classes in categorical features
        :param emb_output_dims: embedding dimensions of categorical features
        :param reg:
        """
        self.inputs = Input((input_dim,))
        self.categorized = categorize(self.inputs, emb_input_dims, emb_output_dims)

        # here goes main dense layers
        self.inter = Dense(inter_dim, activation='tanh',
                           kernel_regularizer=l2_reg_last_n(reg, 10))(self.categorized)

        self.outputs = Dense(output_dim)(self.inter)

        super().__init__(self.inputs, self.outputs)
