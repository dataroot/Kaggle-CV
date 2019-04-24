import tensorflow as tf
from keras.models import Model
from keras.layers import Lambda
from keras.optimizers import Adam

from models.encoder import Encoder


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
        self.que_model = Encoder(que_dim, inter_dim, output_dim, que_input_embs, que_output_embs, reg=2.0)
        # same for professionals
        self.pro_model = Encoder(pro_dim, inter_dim, output_dim, pro_input_embs, pro_output_embs, reg=0.2)

        # calculate distance between high-level feature vectors
        self.merged = Lambda(lambda x: tf.reduce_sum(tf.square(x[0] - x[1]), axis=-1))(
            [self.que_model.outputs[0], self.pro_model.outputs[0]])
        # and apply activation - e^-x here, actually
        self.outputs = Lambda(lambda x: tf.reshape(tf.exp(-self.merged), (-1, 1)))(self.merged)

        super().__init__([self.que_model.inputs[0], self.pro_model.inputs[0]], self.outputs)
