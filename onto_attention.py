import warnings

from keras.layers import LSTM
from keras.engine import InputSpec
from keras import regularizers
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
        self.initial_ontolstm_weights = self.initial_weights
        self.initial_weights = None
        lstm_input_shape = input_shape[:2] + (input_shape[4],) # removing senses and hyps
        # Now calling LSTM's build to initialize the LSTM weights
        super(OntoAttentionLSTM, self).build(lstm_input_shape)
        # This would have changed the input shape and ndim. Reset it again.
        self.input_spec = [InputSpec(shape=input_shape)]
        
        if self.use_attention:
            # Following are the attention parameters
            # Sense projection and scoring
            self.P_sense_syn_att = self.inner_init((input_dim, self.output_dim), 
                name='{}_P_sense_syn_att'.format(self.name)) # Projection operator for synsets
            self.P_sense_cont_att = self.inner_init((self.output_dim, self.output_dim),
                name='{}_P_sense_cont_att'.format(self.name)) # Projection operator for hidden state (context)
            self.s_sense_att = self.init((self.output_dim,), name='{}_s_senses_att'.format(self.name))

            # Generalization projection and scoring
            self.P_gen_syn_att = self.inner_init((input_dim, self.output_dim),
                name='{}_P_sense_gen_att'.format(self.name)) # Projection operator for synsets
            self.P_gen_cont_att = self.inner_init((self.output_dim, self.output_dim),
                name='{}_P_gen_cont_att'.format(self.name)) # Projection operator for hidden state (context)
            self.s_gen_att = self.init((self.output_dim,), name='{}_s_gen_att'.format(self.name))

            # LSTM's build method would have initialized trainable_weights. Add to it.
            self.trainable_weights.extend([self.P_sense_syn_att, self.P_sense_cont_att, self.s_sense_att,
                                           self.P_gen_syn_att, self.P_gen_cont_att, self.s_gen_att])

        if self.initial_ontolstm_weights is not None:
            self.set_weights(self.initial_ontolstm_weights)
            del self.initial_ontolstm_weights

    def get_initial_states(self, x):
        # Reimplementing because ndim of x is 5. (samples, timesteps, num_senses, num_hyps, input_dim)
        sense_hyp_stripped_x = x[:, :, 0, 0, :]  # (samples, timesteps, input_dim), just like LSTM input.
        # We need the same initial states as regular LSTM
        return super(OntoAttentionLSTM, self).get_initial_states(sense_hyp_stripped_x)

    def _step(self, x_cs, states):
        h_tm1 = states[0]

        # Before the step function is called, the original input is dimshuffled to have (time, samples, senses, hyps, concept_dim)
        # So shape of x_cs is (samples, senses, hyps, concept_dim)
        # TODO: Better definition of attention, and attention weight regularization
        if self.use_attention:
            # Sense attention
            # Consider an average of all syns in each sense
            # TODO: Get rid of tensordot and dependence on theano.tensor
            s_syn_proj = K.T.tensordot(x_cs.mean(axis=2).dimshuffle(1,0,2), self.P_sense_syn_att, axes=(2,0)) # (senses, samples, proj_dim)
            s_cont_proj = K.dot(h_tm1, self.P_sense_cont_att) # (samples, proj_dim)
            # TODO: Expose attention activation
            s_x_proj = K.sigmoid(s_syn_proj + s_cont_proj) # (senses, samples, proj_dim)
            sense_att = K.softmax(K.T.tensordot(s_x_proj.dimshuffle(1,0,2), self.s_sense_att, axes=(2,0))) # (samples, senses)

            # Generalization attention
            g_syn_proj = K.T.tensordot(x_cs.dimshuffle(1,2,0,3), self.P_gen_syn_att, axes=(3,0)) # (senses, hyps, samples, proj_dim)
            g_cont_proj = K.dot(h_tm1, self.P_gen_cont_att) # (samples, proj_dim)
            g_x_proj = K.sigmoid(g_syn_proj + g_cont_proj) # (senses, hyps, samples, proj_dim)
            gen_scores = K.T.tensordot(g_x_proj.dimshuffle(2,0,1,3), self.s_gen_att, axes=(3,0)) # (samples, senses, hyps)
            gs_shape = gen_scores.shape
            gen_scores_rs = gen_scores.reshape((gs_shape[0] * gs_shape[1], gs_shape[2]))
            gen_att_rs = K.softmax(gen_scores_rs).reshape((gs_shape[2], gs_shape[0], gs_shape[1]))

            # We need gen_att dimshuffled to (hyps, samples, senses), so reshaping into the needed order of dims
            att = (gen_att_rs * sense_att).dimshuffle(1,2,0) # (samples, senses, hyps)

            lstm_input_t = (x_cs.dimshuffle(3,0,1,2) * att).sum(axis=(2,3)).T # [\sum_{(senses, hyps)} (in_dim, samples, senses, hyps) X (samples, senses, hyps)].T = (samples, in_dim)
        else:
            att = K.zeros_like(x_cs).sum(axis=-1) # matrix of zeros for attention to be consistent (samples, senses, hyps)
            lstm_input_t = K.mean(x_cs, axis=(1,2)) # shape of x is (samples, concept_dim)
        # Now pass the computed lstm_input to LSTM's step function to get current h and c.
        h, [_, c] = super(OntoAttentionLSTM, self).step(lstm_input_t, states)
        
        return h, c, att
        
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
