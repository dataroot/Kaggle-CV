from keras.models import Model
from keras.layers import Dense, Concatenate
from keras.optimizers import Adam

from models.encoder import Encoder


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
