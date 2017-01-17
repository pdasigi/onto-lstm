'''
Layers used after RNN with return_sequence to summarize the sentence encoding.
'''

from keras.engine import Layer
from keras import initializations
from keras import backend as K

from keras_extensions import switch

class AveragePooling(Layer):
    '''
    This layer takes sequential output from an RNN and simply computes the average of it.
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(AveragePooling, self).__init__(**kwargs)

    def compute_mask(self, input_, mask=None):
        # pylint: disable=unused-argument
        return None

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None):
        # x: (batch_size, input_length, input_dim)
        if mask is None:
            return K.mean(x, axis=1)  # (batch_size, input_dim)
        else:
            # This is to remove padding from the computational graph.
            if K.ndim(mask) > K.ndim(x):
                # This is due to the bug in Bidirectional that is passing the input mask
                # instead of computing output mask.
                # TODO: Fix the implementation of Bidirectional.
                mask = K.any(mask, axis=(-2, -1))
            if K.ndim(mask) < K.ndim(x):
                mask = K.expand_dims(mask)
            masked_input = switch(mask, x, K.zeros_like(x))
            weights = K.cast(mask / (K.sum(mask) + K.epsilon()), 'float32')
            return K.sum(masked_input * weights, axis=1)  # (batch_size, input_dim)


class IntraAttention(AveragePooling):
    '''
    This layer returns a average of the input, but the average is weighted by how close the vector
    from each timestep is to the mean.
    '''
    def __init__(self, init='uniform', projection_dim=50, weights=None, **kwargs):
        self.intra_attention_weights = weights
        self.init = initializations.get(init)
        self.projection_dim = projection_dim
        super(IntraAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # pylint: disable=attribute-defined-outside-init
        input_dim = input_shape[-1]
        self.vector_projector = self.init((input_dim, self.projection_dim))
        self.mean_projector = self.init((input_dim, self.projection_dim))
        self.scorer = self.init((self.projection_dim,))
        super(IntraAttention, self).build(input_shape)
        self.trainable_weights = [self.vector_projector, self.mean_projector, self.scorer]
        if self.intra_attention_weights is not None:
            self.set_weights(self.intra_attention_weights)
            del self.intra_attention_weights

    def call(self, x, mask=None):
        mean = super(IntraAttention, self).call(x, mask)
        # x: (batch_size, input_length, input_dim)
        # mean: (batch_size, input_dim)
        ones = K.expand_dims(K.mean(K.ones_like(x), axis=(0, 2)), dim=0)  # (1, input_length)
        # (batch_size, input_length, input_dim)
        tiled_mean = K.permute_dimensions(K.dot(K.expand_dims(mean), ones), (0, 2, 1))
        if mask is not None:
            if K.ndim(mask) > K.ndim(x):
                # Assuming this is because of the bug in Bidirectional. Temporary fix follows.
                # TODO: Fix Bidirectional.
                mask = K.any(mask, axis=(-2, -1))
            if K.ndim(mask) < K.ndim(x):
                mask = K.expand_dims(mask)
            x = switch(mask, x, K.zeros_like(x))
        # (batch_size, input_length, proj_dim)
        projected_combination = K.tanh(K.dot(x, self.vector_projector) + K.dot(tiled_mean, self.mean_projector))
        scores = K.dot(projected_combination, self.scorer)  # (batch_size, input_length)
        weights = K.softmax(scores)  # (batch_size, input_length)
        attended_x = K.sum(K.expand_dims(weights) * x, axis=1)  # (batch_size, input_dim)
        return attended_x

    def get_config(self):
        config = {"init": self.init.__name__, "projection_dim": self.projection_dim}
        base_config = super(IntraAttention, self).get_config()
        config.update(base_config)
        return config
