import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Embedding, Concatenate
from keras.regularizers import l2


class Encoder(Model):
    """
    Model for extraction of high-level feature vector from question or professional
    """

    def __init__(self, input_dim: int, output_dim: int, emb_input_dims: list, emb_output_dims: list):
        """
        :param input_dim: dimension of raw feature vector
        :param output_dim: dimension of computed high-level feature vector
        :param emb_input_dims: number of unique classes in categorical features
        :param emb_output_dims: embedding dimensions of categorical features
        """
        inputs = Input((input_dim,))
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
            self.inter = Concatenate()(embs)
        else:
            self.inter = inputs

        # it's only a single linear transformation, actually
        outputs = Dense(output_dim, kernel_regularizer=l2(0.5))(self.inter)

        super().__init__(inputs, outputs)


class Mothership(Model):
    """
    Main model which combines two encoders (for questions and professionals),
    calculates distance between high-level feature vectors and applies activation
    """

    def __init__(self, que_dim: int, que_input_embs: list, que_output_embs: list,
                 pro_dim: int, pro_input_embs: list, pro_output_embs: list, inter_dim: int):
        super().__init__()  # idk y

        # build an Encoder model for questions
        self.que_model = Encoder(que_dim, inter_dim, que_input_embs, que_output_embs)
        # same for professionals
        self.pro_model = Encoder(pro_dim, inter_dim, pro_input_embs, pro_output_embs)

        # calculate distance between high-level feature vectors
        self.merged = Lambda(lambda x: tf.reduce_sum(tf.square(x[0] - x[1]), axis=-1))(
            [self.que_model.outputs[0], self.pro_model.outputs[0]])
        # and apply activation - e^-x here, actually
        outputs = Lambda(lambda x: tf.reshape(tf.exp(-self.merged), (-1, 1)))(self.merged)

        super().__init__([self.que_model.inputs[0], self.pro_model.inputs[0]], outputs)
