from keras.layers.recurrent import Recurrent
from keras import activations, initializations
from keras import backend as K
from theano.sandbox.cuda.blas import batched_dot as fast_batched_dot
import theano

class OntoAttentionLSTM(Recurrent):
    '''
    Modification of LSTM implementation in Keras to take a hierarchy as input (matrix instead of a vector at each time step), and take a weighted average of it using attention mechanism.
    '''
    input_ndim = 4
    
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid', weights=None,
                 return_sequences=False, go_backwards=False, stateful=False,
                 input_dim=None, num_hyps=None, input_length=None, use_attention=False, 
                 **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        # Initializations from Recurrent to avoid calling its constructor.
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.stateful = stateful

        self.input_dim = input_dim
        self.num_hyps = num_hyps
        self.input_length = input_length
        self.use_attention = use_attention
        self.initial_weights = weights
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.num_hyps, self.input_dim)
        super(Recurrent, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        input_dim = input_shape[3]
        self.input_dim = input_dim
        self.input = K.placeholder(input_shape)

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensor of shape (output_dim), and 1 all-zero tensor of shape (num_hyps)
            self.states = [None, None, None]

        self.W_i = self.init((input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = K.zeros((self.output_dim,))

        self.W_f = self.init((input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.output_dim,))

        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = K.zeros((self.output_dim,))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = K.zeros((self.output_dim,))

        self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.W_f, self.U_f, self.b_f,
                                  self.W_o, self.U_o, self.b_o]

        if self.use_attention:
            # Following are the attention parameters
            self.P_syn_att = self.inner_init((input_dim, self.output_dim)) # Projection operator for synsets
            self.P_cont_att = self.inner_init((self.output_dim, self.output_dim)) # Projection operator for hidden state (context)
            self.s_att = self.init((self.output_dim,))
            self.trainable_weights.extend([self.P_syn_att, self.P_cont_att, self.s_att])

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            #del self.initial_weights

    # Reimplementing because ndim of X is 4
    def get_initial_states(self, X):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(X)  # (samples, timesteps, num_concepts, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples, input_dim)
        reducer1 = K.zeros((self.input_dim, self.output_dim))
        reducer2 = K.zeros((self.input_dim, self.num_hyps))
        initial_state1 = K.dot(initial_state, reducer1)  # (samples, output_dim)
        initial_state2 = K.dot(initial_state, reducer2)  # (samples, num_hyps)
        #initial_states = [initial_state for _ in range(len(self.states))]
        initial_states = [initial_state1, initial_state1, initial_state2]
        return initial_states

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[2],
                        np.zeros((input_shape[0], self.num_hyps)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.num_hyps))]

    # There are two step functions, one for output and another for attention below. Both call this function.
    def _step(self, x_cs, states):
        assert len(states) == 3
 
        h_tm1 = states[0]
        c_tm1 = states[1]
        # TODO: Use attention from previous state for computing current attention?
        att_tm1 = states[2] 

        # Before the step function is called, the original input is dimshuffled to have (time, samples, concepts, concept_dim)
        # So shape of x_cs is (samples, concepts, concept_dim)
        if self.use_attention:
            #project_concept = lambda x_c, st: K.sigmoid(K.dot(K.concatenate([x_c, st], axis=1), self.P_att))
            # TODO: Make the following line not specific to theano
            #x_proj, _ = theano.scan(fn=project_concept, sequences=[x_cs.dimshuffle(1,0,2)], non_sequences=c_tm1)
            syn_proj = K.T.tensordot(x_cs.dimshuffle(1,0,2), self.P_syn_att, axes=(2,0))
            cont_proj = K.dot(c_tm1, self.P_cont_att)
            x_proj = K.sigmoid(syn_proj + cont_proj)
            att = K.softmax(K.T.tensordot(x_proj.dimshuffle(1,0,2), self.s_att, axes=(2,0)))
            if theano.config.device == "gpu":
                att_t3 = att.reshape((a.shape[0], 1, a.shape[1]))
                x = fast_batched_dot(att_t3, x_cs)
            else:
                # Batched dot probably uses scan internally anyway. Sigh!
                x = K.T.batched_dot(att, x_cs)
        else:
            att = att_tm1 # Continue propogating matrix of zeros for attention
            x = K.mean(x_cs, axis=1) # shape of x is (samples, concept_dim)
        
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
        return h, [h, c, att]

    def att_step(self, x_cs, states):
        h, c, att = self._step(x_cs, states)  
        return att, [h, c, att]

    # Had to reimplement the following function only to avoid asserting ndim(X) == 3!
    def get_output(self, train=False):
        # input shape: (nb_samples, time (padded with zeros), num_concepts, input_dim)
        X = self.get_input(train)
        mask = self.get_input_mask(train)

        assert K.ndim(X) == 4
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)

        last_output, outputs, states = K.rnn(self.step, X,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask)
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def get_attention(self):
        # input shape: (nb_samples, time (padded with zeros), num_concepts, input_dim)
        X = self.get_input(False)
        mask = self.get_input_mask(False)

        assert K.ndim(X) == 4
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)

        _, attention, _ = K.rnn(self.att_step, X,
                                initial_states,
                                go_backwards=self.go_backwards,
                                mask=mask)
        return attention

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__}
        base_config = super(OntoAttentionLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
