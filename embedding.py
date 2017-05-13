from keras import backend as K
from keras.layers import Embedding

class OntoAwareEmbedding(Embedding):
    '''
    This class modifies two aspects of the Embedding class in Keras:
    1. Higher order inputs: Embedding already works with inputs of any shape, except that the output shape
    returned by it assumes that the input it 2D. Changing it.
    2. Sense priors: The expected input shape is (num_samples, num_words, num_senses, num_hyps+1). The +1 at the
    end is the word index appended at the end of each sense indices vector. This is to define an additional real
    value for each word, which will act as the sense prior in OntoLSTM. OntoLSTM is reponsible for handling this
    correctly. The output shape is (num_samples, num_words, num_senses, num_hyps, embedding_dim + 1).
    '''
    input_ndim = 4

    def __init__(self, word_index_size, synset_index_size, embedding_dim, set_sense_priors=True,
                 tune_embedding=True, **kwargs):
        self.embedding_dim = embedding_dim
        self.word_index_size = word_index_size
        self.synset_index_size = synset_index_size
        self.set_sense_priors = set_sense_priors
        # We have a separate "tune_embedding" field instead of using trainable because we have two sets of
        # parameters here: the embedding weights, and sense prior weights. We may want to fix only one of
        # them at a time.
        self.tune_embedding = tune_embedding
        # Convincing Embedding to return an embedding of the right shape. The output_dim of this layer is embedding_dim+1
        kwargs['output_dim'] = self.embedding_dim
        kwargs['input_dim'] = self.synset_index_size
        self.onto_aware_embedding_weights = None
        super(OntoAwareEmbedding, self).__init__(**kwargs)

    @staticmethod
    def _get_initial_sense_priors(shape, rate_range=None, name=None):
        # This returns a Keras variable with the initial values all being 0.5.
        if rate_range is None:
            low, high = 0.01, 0.99
        else:
            low, high = rate_range
        return K.random_uniform_variable(shape, low, high, name=name)

    def build(self, input_shape):
        # input shape is (batch_size, num_words, num_senses, num_hyps)
        self.num_senses = input_shape[-2]
        self.num_hyps = input_shape[-1] - 1  # -1 because the last value is a word index
        # embedding of size 1.
        if self.set_sense_priors:
            self.sense_priors = self._get_initial_sense_priors((self.word_index_size, 1), name='{}_sense_priors'.format(self.name))
        else:
            # OntoLSTM makes sense proabilities uniform if the passed sense parameters are zero.
            self.sense_priors = K.zeros((self.word_index_size, 1))  # uniform sense probs
        # Keeping aside the initial weights to not let Embedding set them. It wouldn't know what sense priors are.
        if self.initial_weights is not None:
            self.onto_aware_embedding_weights = self.initial_weights
            self.initial_weights = None
        # The following method will set self.trainable_weights
        super(OntoAwareEmbedding, self).build(input_shape)  # input_shape will not be used by Embedding's build.
        if not self.tune_embedding:
            # Move embedding to non_trainable_weights
            self._non_trainable_weights.append(self._trainable_weights.pop())

        if self.set_sense_priors:
            self._trainable_weights.append(self.sense_priors)

        if self.onto_aware_embedding_weights is not None:
            self.set_weights(self.onto_aware_embedding_weights)
 
    def call(self, x, mask=None):
        # Remove the word indices at the end before making a call to Embedding.
        x_synsets = x[:, :, :, :-1]  # (num_samples, num_words, num_senses, num_hyps)
        # Taking the last index from the first sense. The last index in all the senses will be the same.
        x_word_index = x[:, :, 0, -1]  # (num_samples, num_words)
        # (num_samples, num_words, num_senses, num_hyps, embedding_dim)
        synset_embedding = super(OntoAwareEmbedding, self).call(x_synsets, mask=None)
        # (num_samples, num_words, 1, 1)
        sense_prior_embedding = K.expand_dims(K.gather(self.sense_priors, x_word_index))
        # Now tile sense_prior_embedding and concatenate it with synset_embedding.
        # (num_samples, num_words, num_senses, num_hyps, 1)
        tiled_sense_prior_embedding = K.expand_dims(K.tile(sense_prior_embedding, (1, 1, self.num_senses, self.num_hyps)))
        synset_embedding_with_priors = K.concatenate([synset_embedding, tiled_sense_prior_embedding])
        return synset_embedding_with_priors

    def compute_mask(self, x, mask=None):
        # Since the output dim is different, we need to change the mask size
        embedding_mask = super(OntoAwareEmbedding, self).compute_mask(x, mask)
        return embedding_mask[:, :, :, :-1] if embedding_mask is not None else None

    def get_output_shape_for(self, input_shape):
        return input_shape[:3] + (self.num_hyps, self.embedding_dim+1)

    def get_config(self):
        config = {"word_index_size": self.word_index_size,
                  "synset_index_size": self.synset_index_size,
                  "embedding_dim": self.embedding_dim,
                  "set_sense_priors": self.set_sense_priors
                 }
        base_config = super(OntoAwareEmbedding, self).get_config()
        config.update(base_config)
        return config
