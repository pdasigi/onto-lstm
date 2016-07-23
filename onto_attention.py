from keras.layers import LSTM
from keras.engine import InputSpec
from keras import regularizers
from keras import backend as K
from keras_extensions import rnn

class OntoAttentionLSTM(LSTM):
    '''
    Modification of LSTM implementation in Keras to take a hierarchy as input (matrix instead of a vector at each time step), and take a weighted average of it using attention mechanism.
    '''
    input_ndim = 5
    
    def __init__(self, output_dim,
                 num_senses, num_hyps, use_attention=False, 
                 return_attention=False, **kwargs):
        # Set output_dim in kwargs so that we can pass it along to LSTM's init
        kwargs['output_dim'] = output_dim
        self.num_senses = num_senses
        self.num_hyps = num_hyps
        self.use_attention = use_attention
        self.return_attention = return_attention
        super(OntoAttentionLSTM, self).__init__(**kwargs)
        # Recurrent would have set the input shape to cause the input dim to 3. Change it.
        self.input_spec = [InputSpec(ndim=5)]

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
        
        self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.W_f, self.U_f, self.b_f,
                                  self.W_o, self.U_o, self.b_o]

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

    # Reimplementing because ndim of X is 5
    def get_initial_states(self, X):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(X)  # (samples, timesteps, num_senses, num_hyps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2, 3))  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_states = [initial_state, initial_state]
        return initial_states

    # There are two step functions, one for output and another for attention below. Both call this function.
    def _step(self, x_cs, states):
        assert len(states) == 2
 
        h_tm1 = states[0]
        c_tm1 = states[1]
        # TODO: Use attention from previous state for computing current attention?
        #att_tm1 = states[2] 

        # Before the step function is called, the original input is dimshuffled to have (time, samples, senses, hyps, concept_dim)
        # So shape of x_cs is (samples, senses, hyps, concept_dim)
        # TODO: Better definition of attention, and attention weight regularization
        if self.use_attention:
            # Sense attention
            # Consider an average of all syns in each sense
            s_syn_proj = K.T.tensordot(x_cs.mean(axis=2).dimshuffle(1,0,2), self.P_sense_syn_att, axes=(2,0)) # (senses, samples, proj_dim)
            s_cont_proj = K.dot(h_tm1, self.P_sense_cont_att) # (samples, proj_dim)
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

            x = (x_cs.dimshuffle(3,0,1,2) * att).sum(axis=(2,3)).T # [\sum_{(senses, hyps)} (in_dim, samples, senses, hyps) X (samples, senses, hyps)].T = (samples, in_dim)
        else:
            att = K.zeros_like(x_cs).sum(axis=-1) # matrix of zeros for attention to be consistent (samples, senses, hyps)
            x = K.mean(x_cs, axis=(1,2)) # shape of x is (samples, concept_dim)
        
        x_i = K.dot(x, self.W_i) + self.b_i
        x_f = K.dot(x, self.W_f) + self.b_f
        x_c = K.dot(x, self.W_c) + self.b_c
        x_o = K.dot(x, self.W_o) + self.b_o

        i = self.inner_activation(x_i + K.dot(h_tm1, self.U_i))
        f = self.inner_activation(x_f + K.dot(h_tm1, self.U_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1, self.U_c))
        o = self.inner_activation(x_o + K.dot(h_tm1, self.U_o))
        h = o * self.activation(c)
        return h, c, att
        
    def step(self, x_cs, states):
        h, c, att = self._step(x_cs, states)
        return h, [h, c]

    def att_step(self, x_cs, states):
        h, c, att = self._step(x_cs, states)  
        return att, [h, c]

    def get_constants(self, x):
        return []
    
    # redefining compute mask because the input ndim is different from the output ndim, and 
    # this needs to be handled.
    def compute_mask(self, input, mask):
        if self.return_sequences:
            # Get rid of syn and hyp dimensions for computing loss
            return mask.sum(axis=(-2, -1))
        else:
            return None

    # Redefining call because we may want to return attention values instead of the actual output.
    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), num_senses, num_hyps, input_dim)
        input_shape = self.input_spec[0].shape

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)

        if self.return_attention:
            _, attention, _ = rnn(self.att_step, x,
                                initial_states,
                                go_backwards=self.go_backwards,
                                mask=mask, constants=constants,
                                unroll=self.unroll, input_length=input_shape[1], eliminate_mask_dims=[-3, -2])
            return attention
        else:
            last_output, outputs, states = rnn(self.step, x,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask, constants=constants,
                                             unroll=self.unroll, input_length=input_shape[1], eliminate_mask_dims=[-3, -2])
            if self.stateful:
                self.updates = []
                for i in range(len(states)):
                    self.updates.append((self.states[i], states[i]))

            if self.return_sequences:
                return outputs
            else:
                return last_output

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "use_attention": self.use_attention,
                  "return_attention": self.return_attention}
        base_config = super(OntoAttentionLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
