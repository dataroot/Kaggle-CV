from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.optimizers import Adam

from models.encoder import categorize


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
