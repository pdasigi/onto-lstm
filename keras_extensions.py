from __future__ import absolute_import
from keras import backend as K

from keras.layers import Embedding



class HigherOrderEmbedding(Embedding):
    '''Turn positive integer index tensors into higher order tensors of fixed size.
    Extenstion of Embedding layer in Keras to arbitrary dimensions.
    eg. [[[4, 5]], [[20, 4]]] -> [[[0.25, 0.1], [-0.62, 0.9]], [[0.6, -0.2], [0.25, 0.1]]]

    This layer can only be used as the first layer in a model.

    # Input shape
        nD tensor with shape: `(nb_samples, ...)`.

    # Output shape
        (n+1)D tensor with shape: `(nb_samples, ..., output_dim)`.

    '''
    def get_output_shape_for(self, input_shape):
        return input_shape + (self.output_dim,)
