import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Embedding, Concatenate

class Encoder(Model):
    def __init__(self, input_dim: int, output_dim: int, emb_input_dims: list, emb_output_dims: list):
        inputs = Input((input_dim, ))
        
        n_embs = len(emb_input_dims)
        embs = []
        for i, nunique, dim in zip(range(n_embs), emb_input_dims, emb_output_dims):
            tmp = Lambda(lambda x: x[:, i])(inputs)
            embs.append(Embedding(nunique, dim)(tmp))
        embs.append(Lambda(lambda x: x[:, n_embs:])(inputs))
        
        self.inter = Concatenate()(embs)
        outputs = Dense(output_dim)(self.inter)
        
        super().__init__(inputs, outputs)
        
class Merger(Model):
    def __init__(self, dim):
        inputs_que = Input((dim, ))
        inputs_pro = Input((dim, ))
        
        # Purely experimental approach for loss function construction
        self.merged = Lambda(lambda x: tf.reduce_sum(tf.square(x[0]-x[1]), axis = -1))([inputs_que, inputs_pro])
        outputs = Lambda(lambda x: tf.exp(-self.merged))(self.merged)
        
        super().__init__([inputs_que, inputs_pro], outputs)
        
class Mothership(Model):
    def __init__(self, que_dim: int, que_input_embs: list, pro_dim: int, pro_input_embs: list, inter_dim: int, \
                que_output_embs: list, pro_output_embs: list):
        super().__init__() # idk y
        
        self.que_model = Encoder(que_dim, inter_dim, que_input_embs, que_output_embs)
        self.pro_model = Encoder(pro_dim, inter_dim, pro_input_embs, pro_output_embs)
        self.merge_model = Merger(inter_dim)
        
        super().__init__([self.que_model.inputs[0], self.pro_model.inputs[0]], 
                         self.merge_model([self.pro_model.outputs[0], self.que_model.outputs[0]]))