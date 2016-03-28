from keras.layers.recurrent import Recurrent
from keras import activations, initializations
from keras import backend as K


class OntoAttentionLSTM(Recurrent):
    '''Long-Short Term Memory unit - Hochreiter 1997.

    For a step-by-step description of the algorithm, see
    [this tutorial](http://deeplearning.net/tutorial/lstm.html).

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.

    # References
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
    '''
    input_ndim = 4
    
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid', weights=None,
                 return_sequences=False, go_backwards=False, stateful=False,
                 input_dim=None, num_hyps=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        # Initializations from Recurrent to avoid calling its constructor.
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.go_backwards = go_backwards
        self.stateful = stateful

        self.input_dim = input_dim
        self.num_hyps = num_hyps
        self.input_length = input_length
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
            # initial states: 2 all-zero tensor of shape (output_dim)
            self.states = [None, None]

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

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    # Reimplementing because ndim of X is 4
    def get_initial_states(self, X):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(X)  # (samples, timesteps, num_concepts, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
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
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]

    def step(self, x_cs, states):
        assert len(states) == 2
        # Before the step function is called, the original input is dimshuffled to have (time, samples, concepts, concept_dim)
        x = K.mean(x_cs, axis=1) # shape is (samples, concepts, concept_dim)
        h_tm1 = states[0]
        c_tm1 = states[1]

        x_i = K.dot(x, self.W_i) + self.b_i
        x_f = K.dot(x, self.W_f) + self.b_f
        x_c = K.dot(x, self.W_c) + self.b_c
        x_o = K.dot(x, self.W_o) + self.b_o

        i = self.inner_activation(x_i + K.dot(h_tm1, self.U_i))
        f = self.inner_activation(x_f + K.dot(h_tm1, self.U_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1, self.U_c))
        o = self.inner_activation(x_o + K.dot(h_tm1, self.U_o))
        h = o * self.activation(c)
        return h, [h, c]

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

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__}
        base_config = super(OntoAttentionLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
