import sys

from keras.engine import Layer
from keras import initializations
from keras import backend as K

from keras_extensions import switch

class PrepositionPredictor(Layer):
    '''
    This is a generic predictor for various preposition tasks (PP attachment prediction, preposition relation
    prediction, etc.). We generally have one projector each for the head phrase, preposition and the child
    phrase, yielding a projection that is passed through an MLP to yield a distribution over the required number
    of classes.
    '''
    def __init__(self, score_dim=1, num_hidden_layers=0, proj_dim=None, init='uniform', composition_type='HPCT',
                 **kwargs):
        self.composition_type = composition_type
        self.supports_masking = True
        self.num_hidden_layers = num_hidden_layers
        self.proj_dim = proj_dim
        self.init = initializations.get(init)
        self.proj_head = None
        self.proj_prep = None
        self.proj_child = None
        self.scorer = None
        self.hidden_layers = []
        self.score_dim = score_dim
        self.allowed_compositions = []
        super(PrepositionPredictor, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        raise NotImplementedError

    def build(self, input_shape):
        # The composition types are taken from Belinkov et al.'s TACL 2014 paper:
        # HC: Head-Child; HPC: Head-Prep-Child; HPCT: Head-Prep-Child-Ternary.
        assert self.composition_type in self.allowed_compositions, "Unknown composition type: %s" % self.composition_type
        if isinstance(input_shape[0], tuple):
            # This layer has multiple inputs (RelationPredictor).
            input_dim = input_shape[0][-1]
            input_length = input_shape[0][1]
        else:
            input_dim = input_shape[-1]
            input_length = input_shape[1]
        if self.proj_dim is None:
            self.proj_dim = int(input_dim / 2)
        if self.composition_type == 'HPCD':
            max_num_heads = input_length - 2
            # Clipping number of distance based projection matrices to 5.
            num_head_projectors = min(max_num_heads, 5)
            self.proj_head = self.init((num_head_projectors, input_dim, self.proj_dim))
            if max_num_heads > num_head_projectors:
                diff = max_num_heads - num_head_projectors
                farthest_head_proj = K.expand_dims(self.proj_head[0, :, :], dim=0)  # (1, input_dim, proj_dim)
                # (diff, input_dim, proj_dim)
                tiled_farthest_head_proj = K.repeat_elements(farthest_head_proj, diff, 0)
                # (head_size, input_dim, proj_dim)
                self.dist_proj_head = K.concatenate([tiled_farthest_head_proj, self.proj_head], axis=0)
            else:
                self.dist_proj_head = self.proj_head
        else:
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
        self.scorer = self.init((self.proj_dim, self.score_dim), name='{}_scorer'.format(self.name))
        self.trainable_weights.append(self.scorer)

    def compute_mask(self, input, mask=None):
        return None

    def call(self, x, mask=None):
        raise NotImplementedError

    def get_config(self):
        config = {"num_hidden_layers": self.num_hidden_layers,
                  "proj_dim": self.proj_dim,
                  "composition_type": self.composition_type,
                  "init": self.init.__name__}
        base_config = super(PrepositionPredictor, self).get_config()
        config.update(base_config)
        return config


class AttachmentPredictor(PrepositionPredictor):
    '''
    AttachmentPredictor is a layer that takes an encoded representation of a phrase that ends with a preposition
    phrase (preposition followed by a noun) and predicts which of the words that come before the PP it attaches to.

    This layer takes as input a sequence output from an RNN, and suumes that the last two timesteps correspond to
    the PP.
    '''
    def __init__(self, **kwargs):
        kwargs["score_dim"] = 1  # Softmax is over head indices, so make output of scorer be of size 1.
        super(AttachmentPredictor, self).__init__(**kwargs)
        self.allowed_compositions = ['HC', 'HPC', 'HPCD', 'HPCT']
        print >>sys.stderr, "Initializing attachment predictor with %s composition" % self.composition_type

    def get_output_shape_for(self, input_shape):
        head_size = input_shape[1] - 2
        return (input_shape[0], head_size)

    def call(self, x, mask=None):
        # x: (batch_size, input_length, input_dim) where input_length = head_size + 2
        head_encoding = x[:, :-2, :]  # (batch_size, head_size, input_dim)
        prep_encoding = x[:, -2, :]  # (batch_size, input_dim)
        child_encoding = x[:, -1, :]  # (batch_size, input_dim)
        if self.composition_type == 'HPCD':
            # TODO: The following line may not work with TF.
            # (batch_size, head_size, input_dim, 1) * (1, head_size, input_dim, proj_dim)
            head_proj_prod = K.expand_dims(head_encoding) * K.expand_dims(self.dist_proj_head, dim=0)
            head_projection = K.sum(head_proj_prod, axis=2)  # (batch_size, head_size, proj_dim)
        else:
            head_projection = K.dot(head_encoding, self.proj_head)  # (batch_size, head_size, proj_dim)
        prep_projection = K.expand_dims(K.dot(prep_encoding, self.proj_prep), dim=1)  # (batch_size, 1, proj_dim)
        child_projection = K.expand_dims(K.dot(child_encoding, self.proj_child), dim=1)  # (batch_size, 1, proj_dim)
        #(batch_size, head_size, proj_dim)
        if self.composition_type == 'HPCT':
            composed_projection = K.tanh(head_projection + prep_projection + child_projection)
        elif self.composition_type == 'HPC' or self.composition_type == "HPCD":
            prep_child_projection = K.tanh(prep_projection + child_projection)  # (batch_size, 1, proj_dim)
            composed_projection = K.tanh(head_projection + prep_child_projection)
        else:
            # Composition type in HC
            composed_projection = K.tanh(head_projection + child_projection)
        for hidden_layer in self.hidden_layers:
            composed_projection = K.tanh(K.dot(composed_projection, hidden_layer))  # (batch_size, head_size, proj_dim)
        # (batch_size, head_size)
        head_word_scores = K.squeeze(K.dot(composed_projection, self.scorer), axis=-1)
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


class RelationPredictor(PrepositionPredictor):
    '''
    RelationPredictor is a layer that takes an encoded sentence, and the index of the preposition
    and predicts the relation expressed by the preposition. Optionally it takes attachment probabilities
    predicted by an attachment model, and uses the predicted head to predict the relation. If this option
    is not used, the layer assumes the last word in the head phrase is the head.

    Note that this layer takes two or three inputs.
    '''
    def __init__(self, output_dim=32, with_attachment_probs=False, **kwargs):
        kwargs["score_dim"] = output_dim
        super(RelationPredictor, self).__init__(**kwargs)
        self.allowed_compositions = ['HC', 'HPC', 'HPCT']
        self.with_attachment_probs = with_attachment_probs
        print >>sys.stderr, "Initializing relation predictor with %s composition" % self.composition_type
        if self.with_attachment_probs:
            print >>sys.stderr, "\tAttachment probabilities given."

    def get_output_shape_for(self, input_shape):
        # input_shape is a list with two or three elements.
        return (input_shape[0][0], self.score_dim)

    def call(self, x, mask=None):
        # x[0]: (batch_size, input_length, input_dim)
        # x[1]: (batch_size, 1) indices of prepositions
        # Optional: x[2]: (batch_size, input_length - 2)
        assert isinstance(x, list) or isinstance(x, tuple)
        encoded_sentence = x[0]
        prep_indices = K.squeeze(x[1], axis=-1)  #(batch_size,)
        batch_indices = K.arange(K.shape(encoded_sentence)[0])  # (batch_size,)
        if self.with_attachment_probs:
            # We're essentially doing K.argmax(x[2]) here, but argmax is not differentiable!
            head_probs = x[2]
            head_probs_padding = K.zeros_like(x[2])[:, :2]  # (batch_size, 2)
            # (batch_size, input_length)
            padded_head_probs = K.concatenate([head_probs, head_probs_padding])
            # (batch_size, 1)
            max_head_probs = K.expand_dims(K.max(padded_head_probs, axis=1))
            # (batch_size, input_length, 1)
            max_head_prob_indices = K.expand_dims(K.equal(padded_head_probs, max_head_probs))
            # (batch_size, input_length, input_dim)
            masked_head_encoding = K.switch(max_head_prob_indices, encoded_sentence, K.zeros_like(encoded_sentence))
            # (batch_size, input_dim)
            head_encoding = K.sum(masked_head_encoding, axis=1)
        else:
            head_indices = prep_indices - 1  # (batch_size,)
            head_encoding = encoded_sentence[batch_indices, head_indices, :]  # (batch_size, input_dim)
        prep_encoding = encoded_sentence[batch_indices, prep_indices, :]  # (batch_size, input_dim)
        child_encoding = encoded_sentence[batch_indices, prep_indices+1, :]  # (batch_size, input_dim)
        '''
        prep_indices = x[1]
        sentence_mask = mask[0]
        if sentence_mask is not None:
            if K.ndim(sentence_mask) > 2:
                # This means this layer came after a Bidirectional layer. Keras has this bug which
                # concatenates input masks instead of output masks.
                # TODO: Fix Bidirectional instead.
                sentence_mask = K.any(sentence_mask, axis=(-2, -1))
        head_encoding, prep_encoding, child_encoding = self.get_split_averages(encoded_sentence, sentence_mask,
                                                                               prep_indices)
        '''
        head_projection = K.dot(head_encoding, self.proj_head)  # (batch_size, proj_dim)
        prep_projection = K.dot(prep_encoding, self.proj_prep)  # (batch_size, proj_dim)
        child_projection = K.dot(child_encoding, self.proj_child)  # (batch_size, proj_dim)
        #(batch_size, proj_dim)
        if self.composition_type == 'HPCT':
            composed_projection = K.tanh(head_projection + prep_projection + child_projection)
        elif self.composition_type == 'HPC':
            prep_child_projection = K.tanh(prep_projection + child_projection)  # (batch_size, proj_dim)
            composed_projection = K.tanh(head_projection + prep_child_projection)
        else:
            # Composition type in HC
            composed_projection = K.tanh(head_projection + child_projection)
        for hidden_layer in self.hidden_layers:
            composed_projection = K.tanh(K.dot(composed_projection, hidden_layer))  # (batch_size, proj_dim)
        # (batch_size, num_classes)
        class_scores = K.dot(composed_projection, self.scorer)
        label_probabilities = K.softmax(class_scores)
        return label_probabilities

    @staticmethod
    def get_split_averages(input_tensor, input_mask, indices):
        # Splits input tensor into three parts based on the indices and
        # returns average of values prior to index, values at the index and
        # average of values after the index.
        # input_tensor: (batch_size, input_length, input_dim)
        # input_mask: (batch_size, input_length)
        # indices: (batch_size, 1)
        # (1, input_length)
        length_range = K.expand_dims(K.arange(K.shape(input_tensor)[1]), dim=0)
        # (batch_size, input_length)
        batched_range = K.repeat_elements(length_range, K.shape(input_tensor)[0], 0)
        tiled_indices = K.repeat_elements(indices, K.shape(input_tensor)[1], 1)  # (batch_size, input_length)
        greater_mask = K.greater(batched_range, tiled_indices)  # (batch_size, input_length)
        lesser_mask = K.lesser(batched_range, tiled_indices)  # (batch_size, input_length)
        equal_mask = K.equal(batched_range, tiled_indices)  # (batch_size, input_length)

        # We also need to mask these masks using the input mask.
        # (batch_size, input_length)
        if input_mask is not None:
            greater_mask = switch(input_mask, greater_mask, K.zeros_like(greater_mask))
            lesser_mask = switch(input_mask, lesser_mask, K.zeros_like(lesser_mask))

        post_sum = K.sum(switch(K.expand_dims(greater_mask), input_tensor, K.zeros_like(input_tensor)), axis=1)  # (batch_size, input_dim)
        pre_sum = K.sum(switch(K.expand_dims(lesser_mask), input_tensor, K.zeros_like(input_tensor)), axis=1)  # (batch_size, input_dim)
        values_at_indices = K.sum(switch(K.expand_dims(equal_mask), input_tensor, K.zeros_like(input_tensor)), axis=1)  # (batch_size, input_dim)

        post_normalizer = K.expand_dims(K.sum(greater_mask, axis=1) + K.epsilon(), dim=1)  # (batch_size, 1)
        pre_normalizer = K.expand_dims(K.sum(lesser_mask, axis=1) + K.epsilon(), dim=1)  # (batch_size, 1)

        return K.cast(pre_sum / pre_normalizer, 'float32'), values_at_indices, K.cast(post_sum / post_normalizer, 'float32')
        

    def get_config(self):
        config = {"output_dim": self.score_dim}
        base_config = super(RelationPredictor, self).get_config()
        config.update(base_config)
        return config
