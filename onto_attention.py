import warnings

from keras.layers import LSTM
from keras.engine import InputSpec
from keras import backend as K
#from keras_extensions import rnn

class OntoAttentionLSTM(LSTM):
    '''
    Modification of LSTM implementation in Keras to take a WordNet subtree instead of a the word at each timestep.
    The WordNet subtree is given as a sense separated hypernym hierarchy i.e., words are represented as tensors
    instead of vectors at each time step. The wows in the tensors are shared, as synsets are shared across words in
    WordNet. We take a weighted average of the tensor using attention mechanism conditioned on the output of the 
    previous timestep to get a vector, and that vector is processed in the same way the input is processed by LSTM.
    '''
    input_ndim = 5
    
    def __init__(self, output_dim, num_senses, num_hyps, use_attention=False, return_attention=False, **kwargs):
        # Set output_dim in kwargs so that we can pass it along to LSTM's init
        kwargs['output_dim'] = output_dim
        self.num_senses = num_senses
        self.num_hyps = num_hyps
        self.use_attention = use_attention
        self.return_attention = return_attention
        super(OntoAttentionLSTM, self).__init__(**kwargs)
        # Recurrent would have set the input shape to cause the input dim to be 3. Change it.
        self.input_spec = [InputSpec(ndim=5)]
        if self.consume_less == "cpu":
            # In the LSTM implementation in Keras, consume_less = cpu causes all gates' inputs to be precomputed
            # and stored in memory. However, this doesn't work with OntoLSTM since the input to the gates is 
            # dependent on the previous timestep's output.
            warnings.warn("OntoLSTM does not support consume_less = cpu. Changing it to mem.")
            self.consume_less = "mem"

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = input_shape[4]
        self.input_dim = input_dim
        # Saving onto-lstm weights to set them later. This way, LSTM's build method won't 
        # delete them.
        initial_ontolstm_weights = self.initial_weights
        self.initial_weights = None
        lstm_input_shape = input_shape[:2] + (input_shape[4],) # removing senses and hyps
        # Now calling LSTM's build to initialize the LSTM weights
        super(OntoAttentionLSTM, self).build(lstm_input_shape)
        # This would have changed the input shape and ndim. Reset it again.
        self.input_spec = [InputSpec(shape=input_shape)]

        if self.use_attention:
            # Following are the attention parameters
            # Sense projection and scoring
            self.input_sense_projector = self.inner_init((input_dim, self.output_dim), 
                name='{}_input_sense_projector'.format(self.name)) # Projection operator for synsets
            self.context_sense_projector = self.inner_init((self.output_dim, self.output_dim),
                name='{}_context_sense_projector'.format(self.name)) # Projection operator for hidden state (context)
            self.sense_scorer = self.init((self.output_dim,), name='{}_sense_scorer'.format(self.name))

            # Generalization projection and scoring
            self.input_hyp_projector = self.inner_init((input_dim, self.output_dim),
                name='{}_input_hyp_projector'.format(self.name)) # Projection operator for synsets
            self.context_hyp_projector = self.inner_init((self.output_dim, self.output_dim),
                name='{}_context_hyp_projector'.format(self.name)) # Projection operator for hidden state (context)
            self.hyp_scorer = self.init((self.output_dim,), name='{}_hyp_scorer'.format(self.name))

            # LSTM's build method would have initialized trainable_weights. Add to it.
            self.trainable_weights.extend([self.input_sense_projector, self.context_sense_projector,
                                           self.sense_scorer, self.input_hyp_projector, 
                                           self.context_hyp_projector, self.hyp_scorer])

        if initial_ontolstm_weights is not None:
            self.set_weights(initial_ontolstm_weights)
            del initial_ontolstm_weights

    def get_initial_states(self, x):
        # Reimplementing because ndim of x is 5. (samples, timesteps, num_senses, num_hyps, input_dim)
        sense_hyp_stripped_x = x[:, :, 0, 0, :]  # (samples, timesteps, input_dim), just like LSTM input.
        # We need the same initial states as regular LSTM
        return super(OntoAttentionLSTM, self).get_initial_states(sense_hyp_stripped_x)

    def _step(self, x_onto_aware, states):
        h_tm1 = states[0]

        # Before the step function is called, the original input is dimshuffled to have (time, samples, senses, hyps, concept_dim)
        # So shape of x_cs is (samples, senses, hyps, concept_dim)
        # TODO: Better definition of attention, and attention weight regularization
        if self.use_attention:
            # Sense attention
            x_hyp_averaged = K.mean(x_onto_aware, axis=2)  # (samples, sense, input_dim)
            input_sense_projection = K.dot(x_hyp_averaged, self.input_sense_projector)  # (samples, senses, proj_dim)
            context_sense_projection = K.dot(h_tm1, self.context_sense_projector) # (samples, proj_dim)
            # TODO: Expose attention activation
            sense_projection = K.sigmoid(input_sense_projection + K.expand_dims(context_sense_projection, dim=1))  # (samples, senses, proj_dim)
            sense_scores = K.dot(sense_projection, self.sense_scorer)  # (samples, senses)
            #sense_attention = K.softmax(sense_scores) # (samples, senses)

            # Generalization attention
            input_hyp_projection = K.dot(x_onto_aware, self.input_hyp_projector) # (samples, senses, hyps, proj_dim)
            context_hyp_projection = K.dot(h_tm1, self.context_hyp_projector) # (samples, proj_dim)
            context_hyp_projection_expanded = K.expand_dims(K.expand_dims(context_hyp_projection,
                                                                          dim=1),
                                                            dim=1)  #(samples, 1, 1, proj_dim)
            hyp_projection = K.sigmoid(input_hyp_projection + context_hyp_projection_expanded) # (samples, senses, hyps, proj_dim)
            hyp_scores = K.dot(hyp_projection, self.hyp_scorer) # (samples, senses, hyps)
            # Now we need to compute softmax of gen_scores. But we need to reshape it first since we
            # cannot perform softmax on tensors. We need \sum_{senses} \sum_{hyps} gen_att[i] = 1.0,
            # for each sample i.
            #hyp_scores_shape = K.shape(hyp_scores)
            #flat_hyp_scores = K.batch_flatten(hyp_scores)  # (samples, senses * hyps)
            #hyp_attention = K.reshape(K.softmax(flat_hyp_scores), hyp_scores_shape)  # (samples, senses, hyps)
            # (samples, senses, hyps) * (samples, senses, 1)
            #sense_hyp_attention = (hyp_attention * K.expand_dims(sense_attention))  # (samples, senses, hyps)

            # Multiply hyp and sense scores and then normalize. Attention values now sum to 1.
            sense_hyp_scores = (hyp_scores * K.expand_dims(sense_scores))  # (samples, senses, hyps)
            scores_shape = K.shape(sense_hyp_scores)
            # We need to flatten this because we cannot perform softmax on tensors.
            flattened_scores = K.batch_flatten(sense_hyp_scores)  # (samples, senses*hyps)
            sense_hyp_attention = K.reshape(K.softmax(flattened_scores), scores_shape)  # (samples, senses, hyps)
            weighted_product = x_onto_aware * K.expand_dims(sense_hyp_attention)  # (samples, senses, hyps, input_dim)
            # Weighted average, summing over senses and hyps
            lstm_input_t = K.sum(weighted_product, axis=(1,2))  # (samples, input_dim)
        else:
            sense_hyp_attention = K.sum(K.zeros_like(x_onto_aware), axis=-1)  # matrix of zeros for attention to be consistent (samples, senses, hyps)
            lstm_input_t = K.mean(x_onto_aware, axis=(1,2))  # shape of x is (samples, concept_dim)
        # Now pass the computed lstm_input to LSTM's step function to get current h and c.
        h, [_, c] = super(OntoAttentionLSTM, self).step(lstm_input_t, states)
        
        return h, c, sense_hyp_attention
        
    def step(self, x_cs, states):
        h, c, att = self._step(x_cs, states)
        if self.return_attention:
            return att, [h, c]
        else:
            return h, [h, c]

    def get_constants(self, x):
        # Reimplementing because ndim of x is 5. (samples, timesteps, num_senses, num_hyps, input_dim)
        sense_hyp_stripped_x = x[:, :, 0, 0, :]  # (samples, timesteps, input_dim), just like LSTM input.
        # We need the same constants as regular LSTM
        return super(OntoAttentionLSTM, self).get_constants(sense_hyp_stripped_x)
    
    # redefining compute mask because the input ndim is different from the output ndim, and 
    # this needs to be handled.
    def compute_mask(self, input, mask):
        if self.return_sequences and mask is not None:
            # Get rid of syn and hyp dimensions for computing loss
            return mask.sum(axis=(-2, -1))
        else:
            return None

    def get_config(self):
        config = {"num_senses": self.num_senses,
                  "num_hyps": self.num_hyps,
                  "use_attention": self.use_attention,
                  "return_attention": self.return_attention}
        base_config = super(OntoAttentionLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
