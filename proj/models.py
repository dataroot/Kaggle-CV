import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Embedding, Concatenate, Layer
from keras.optimizers import Adam


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
                 reg: bool = False):
        """
        :param input_dim: dimension of raw feature vector
        :param inter_dim: dimension of intermediate layer
        :param output_dim: dimension of computed high-level feature vector
        :param emb_input_dims: number of unique classes in categorical features
        :param emb_output_dims: embedding dimensions of categorical features
        """
        self.inputs = Input((input_dim,))
        self.categorized = categorize(self.inputs, emb_input_dims, emb_output_dims)

        # here goes main dense layers
        if not reg:
            self.inter = Dense(inter_dim, activation='tanh')(self.categorized)
        else:
            # in Question's encoder, constrain the last 10 features, which in our case are tag embeddings
            self.inter = Dense(inter_dim, activation='tanh',
                               kernel_regularizer=l2_reg_last_n(2.0, 10))(self.categorized)

        self.outputs = Dense(output_dim)(self.inter)

        super().__init__(self.inputs, self.outputs)


class DistanceModel(Model):
    """
    Main model which combines two encoders (for questions and professionals),
    calculates distance between high-level feature vectors and applies activation
    """

    def __init__(self, que_dim: int, que_input_embs: list, que_output_embs: list,
                 pro_dim: int, pro_input_embs: list, pro_output_embs: list,
                 inter_dim: int, output_dim: int):
        """
        :param que_dim: dimension of question's raw feature vector
        :param que_input_embs: number of unique classes in question's categorical features
        :param que_output_embs: embedding dimensions of question's categorical features
        :param pro_dim: dimension of professional's raw feature vector
        :param pro_input_embs: number of unique classes in professional's categorical features
        :param pro_output_embs: embedding dimensions of professional's categorical features
        :param inter_dim: dimension of Encoder's intermediate layer
        :param output_dim: dimension of high-level feature vectors
        """
        super().__init__()

        # build an Encoder model for questions
        self.que_model = Encoder(que_dim, inter_dim, output_dim, que_input_embs, que_output_embs, reg=True)
        # same for professionals
        self.pro_model = Encoder(pro_dim, inter_dim, output_dim, pro_input_embs, pro_output_embs)

        # calculate distance between high-level feature vectors
        self.merged = Lambda(lambda x: tf.reduce_sum(tf.square(x[0] - x[1]), axis=-1))(
            [self.que_model.outputs[0], self.pro_model.outputs[0]])
        # and apply activation - e^-x here, actually
        self.outputs = Lambda(lambda x: tf.reshape(tf.exp(-self.merged), (-1, 1)))(self.merged)

        super().__init__([self.que_model.inputs[0], self.pro_model.inputs[0]], self.outputs)


class SimpleModel(Model):
    """
    The model with Simple architecture
    """

    def __init__(self,
                 que_dim: int, que_input_embs: list, que_output_embs: list,
                 pro_dim: int, pro_input_embs: list, pro_output_embs: list,
                 inter_dim: int, output_dim: int):
        super().__init__()

        self.que_inputs = Input((que_dim,))
        self.pro_inputs = Input((pro_dim,))

        self.que_categorized = categorize(self.que_inputs, que_input_embs, que_output_embs)
        self.pro_categorized = categorize(self.pro_inputs, pro_input_embs, pro_output_embs)

        self.merged = Concatenate()([self.que_categorized, self.pro_categorized])

        self.inter = Dense(inter_dim, activation='tanh')(self.merged)
        self.inter = Dense(output_dim, activation='tanh')(self.inter)

        self.outputs = Dense(1, activation='sigmoid')(self.inter)

        super().__init__([self.que_inputs, self.pro_inputs], self.outputs)


class ConcatModel(Model):
    """
    The model with Encoder-based architecture that concatenates que and pro latent-space representations
    and passes them through several Dense layers.
    """

    def __init__(self,
                 que_dim: int, que_input_embs: list, que_output_embs: list,
                 pro_dim: int, pro_input_embs: list, pro_output_embs: list,
                 inter_dim: int, output_dim: int):
        super().__init__()

        self.que_model = Encoder(que_dim, inter_dim, output_dim, que_input_embs, que_output_embs)
        self.pro_model = Encoder(pro_dim, inter_dim, output_dim, pro_input_embs, pro_output_embs)

        self.merged = Concatenate()([self.que_model.outputs[0], self.pro_model.outputs[0]])

        self.inter = Dense(16, activation='tanh')(self.merged)
        self.outputs = Dense(1, activation='sigmoid')(self.inter)

        super().__init__([self.que_model.inputs[0], self.pro_model.inputs[0]], self.outputs)
