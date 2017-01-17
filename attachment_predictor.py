import sys

from keras.engine import Layer
from keras import initializations
from keras import backend as K

from keras_extensions import switch

class AttachmentPredictor(Layer):
    '''
    AttachmentPredictor is a layer that takes an encoded representation of a phrase that ends with a preposition
    phrase (preposition followed by a noun) and predicts which of the words that come before the PP it attaches to.

    This layer takes as input a sequence output from an RNN, and suumes that the last two timesteps correspond to
    the PP.
    '''
    def __init__(self, num_hidden_layers, proj_dim=None, init='uniform', composition_type='HPCT', **kwargs):
        # The composition types are taken from Belinkov et al.'s TACL 2014 paper:
        # HC: Head-Child; HPC: Head-Prep-Child; HPCT: Head-Prep-Child-Ternary.
        assert composition_type in ['HC', 'HPC', 'HPCT'], "Unknown composition type: %s" % composition_type
        self.composition_type = composition_type
        print >>sys.stderr, "Initializing attachment predictor with %s composition" % self.composition_type
        self.supports_masking = True
        self.num_hidden_layers = num_hidden_layers
        self.proj_dim = proj_dim
        self.init = initializations.get(init)
        super(AttachmentPredictor, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if self.proj_dim is None:
            self.proj_dim = int(input_dim / 2)
        self.proj_head = self.init((input_dim, self.proj_dim), name='{}_proj_head'.format(self.name))
        self.proj_prep = self.init((input_dim, self.proj_dim), name='{}_proj_prep'.format(self.name))
        self.proj_child = self.init((input_dim, self.proj_dim), name='{}_proj_child'.format(self.name))
        self.trainable_weights = [self.proj_head, self.proj_prep, self.proj_child]
        self.hidden_layers = []
        if self.num_hidden_layers > 0:
            # This means we have to pass the composed representation through an MLP instead of directly computing
            # scores.
            for i in range(self.num_hidden_layers):
                hidden_layer = self.init((self.proj_dim, self.proj_dim), name='%s_hidden_layer_%d' % (self.name, i))
                self.hidden_layers.append(hidden_layer)
            self.trainable_weights.extend(self.hidden_layers)
        self.scorer = self.init((self.proj_dim,), name='{}_scorer'.format(self.name))
        self.trainable_weights.append(self.scorer)

    def get_output_shape_for(self, input_shape):
        head_size = input_shape[1] - 2
        return (input_shape[0], head_size)

    def compute_mask(self, input, mask=None):
        return None

    def call(self, x, mask=None):
        # x: (batch_size, input_length, input_dim) where input_length = head_size + 2
        head_encoding = x[:, :-2, :]  # (batch_size, head_size, input_dim)
        prep_encoding = x[:, -2, :]  # (batch_size, input_dim)
        child_encoding = x[:, -1, :]  # (batch_size, input_dim)
        head_projection = K.dot(head_encoding, self.proj_head)  # (batch_size, head_size, proj_dim)
        prep_projection = K.expand_dims(K.dot(prep_encoding, self.proj_prep), dim=1)  # (batch_size, 1, proj_dim)
        child_projection = K.expand_dims(K.dot(child_encoding, self.proj_child), dim=1)  # (batch_size, 1, proj_dim)
        #(batch_size, head_size, proj_dim)
        if self.composition_type == 'HPCT':
            composed_projection = K.tanh(head_projection + prep_projection + child_projection)
        elif self.composition_type == 'HPC':
            prep_child_projection = K.tanh(prep_projection + child_projection)  # (batch_size, 1, proj_dim)
            composed_projection = K.tanh(head_projection + prep_child_projection)
        else:
            # Composition type in HC
            composed_projection = K.tanh(head_projection + child_projection)
        for hidden_layer in self.hidden_layers:
            composed_projection = K.tanh(K.dot(composed_projection, hidden_layer))  # (batch_size, head_size, proj_dim)
        head_word_scores = K.dot(composed_projection, self.scorer)  # (batch_size, head_size)
        if mask is None:
            attachment_probabilities = K.softmax(head_word_scores)  # (batch_size, head_size)
        else:
            if K.ndim(mask) > 2:
                # This means this layer came after a Bidirectional layer. Keras has this bug which
                # concatenates input masks instead of output masks.
                # TODO: Fix Bidirectional instead.
                mask = K.any(mask, axis=(-2, -1))
            # We need to do a masked softmax.
            exp_scores = K.exp(head_word_scores)  # (batch_size, head_size)
            head_mask = mask[:, :-2]  # (batch_size, head_size)
            # (batch_size, head_size)
            masked_exp_scores = switch(head_mask, exp_scores, K.zeros_like(head_encoding[:, :, 0]))
            # (batch_size, 1). Adding epsilon to avoid divison by 0. But epsilon is float64.
            exp_sum = K.cast(K.expand_dims(K.sum(masked_exp_scores, axis=1) + K.epsilon()), 'float32')
            attachment_probabilities = masked_exp_scores / exp_sum  # (batch_size, head_size)
        return attachment_probabilities

    def get_config(self):
        config = {"num_hidden_layers": self.num_hidden_layers,
                  "proj_dim": self.proj_dim,
                  "composition_type": self.composition_type,
                  "init": self.init.__name__}
        base_config = super(AttachmentPredictor, self).get_config()
        config.update(base_config)
        return config
