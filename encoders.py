import sys
import numpy
from overrides import overrides

from keras.layers import Embedding, Dropout, LSTM, Bidirectional

from onto_attention import OntoAttentionLSTM
from embedding import OntoAwareEmbedding


class Encoder(object):
    '''
    Encoder is an abstract class that defines a get_encoded_phrase method. 
    '''
    def __init__(self, data_processor=None, embed_dim=50, bidirectional=False, tune_embedding=False,
                 return_sequences=True, **kwargs):
        self.embed_dim = embed_dim
        self.data_processor = data_processor
        self.bidirectional = bidirectional
        self.tune_embedding = tune_embedding
        self.return_sequences = return_sequences
        self.embedding_layer = None
        self.encoder_layer = None

    def get_encoded_phrase(self, phrase_input_layer, dropout={}, embedding=None):
        '''
        Takes a Keras input layer, dropout and returns the output of the encoder as a Keras object.

        Arguments:
            phrase_input_layer (Input): Keras Input layer of the appropriate shape.
            dropout (dict [str -> float]): Dict containing dropout values applied after
            `embedding` and `encoder`.
            embedding_file (str): Optional gzipped embedding file to use as initialization
            for embedding layer.
        '''
        embedding_layer = self._get_embedding_layer(embedding)
        embedded_phrase = embedding_layer(phrase_input_layer)
        embedding_dropout = dropout.pop("embedding", 0.0)
        if embedding_dropout > 0:
            embedded_phrase = Dropout(embedding_dropout)(embedded_phrase)
        encoder = self._get_encoder_layer()
        encoded_phrase = encoder(embedded_phrase)
        encoder_dropout = dropout.pop("encoder", 0.0)
        if encoder_dropout > 0:
            encoded_phrase = Dropout(encoder_dropout)(encoded_phrase)
        return encoded_phrase

    def _get_embedding_layer(self, embedding_file=None):
        '''
        Checks if an embedding layer is defined. If so, returns it. Or else, makes one.
        '''
        raise NotImplementedError

    def _get_encoder_layer(self):
        '''
        Checks if an encoder layer is defined. If so, returns it. Or else, makes one.
        '''
        raise NotImplementedError

    @staticmethod
    def get_custom_objects():
        return {}


class LSTMEncoder(Encoder):
    @overrides
    def _get_embedding_layer(self, embedding_file=None):
        if self.embedding_layer is None:
            if embedding_file is None:
                if not self.tune_embedding:
                    print >>sys.stderr, "Pretrained embedding is not given. Setting tune_embedding to True."
                    self.tune_embedding = True
                embedding = None
            else:
                # Put the embedding in a list for Keras to treat it as initiali weights of the embedding
                # layer.
                embedding = [self.data_processor.get_embedding_matrix(embedding_file, onto_aware=False)]
            vocab_size = self.data_processor.get_vocab_size(onto_aware=False)
            self.embedding_layer = Embedding(input_dim=vocab_size, output_dim=self.embed_dim,
                                             weights=embedding, trainable=self.tune_embedding,
                                             mask_zero=True, name="embedding")
        return self.embedding_layer

    @overrides
    def _get_encoder_layer(self):
        if self.encoder_layer is None:
            self.encoder_layer = LSTM(input_dim=self.embed_dim, output_dim=self.embed_dim,
                                      return_sequences=self.return_sequences, name="encoder")
            if self.bidirectional:
                self.encoder_layer = Bidirectional(self.encoder_layer, name="encoder")
        return self.encoder_layer


class OntoLSTMEncoder(Encoder):
    def __init__(self, num_senses, num_hyps, use_attention, set_sense_priors, **kwargs):
        self.num_senses = num_senses
        self.num_hyps = num_hyps
        self.use_attention = use_attention
        self.set_sense_priors = set_sense_priors
        super(OntoLSTMEncoder, self).__init__(**kwargs)

    @overrides
    def _get_embedding_layer(self, embedding_file=None):
        if self.embedding_layer is None:
            word_vocab_size = self.data_processor.get_vocab_size(onto_aware=False)
            synset_vocab_size = self.data_processor.get_vocab_size(onto_aware=True)
            if embedding_file is None:
                if not self.tune_embedding:
                    print >>sys.stderr, "Pretrained embedding is not given. Setting tune_embedding to True."
                    self.tune_embedding = True
                embedding_weights = None
            else:
                # TODO: Other sources for prior initialization
                embedding = self.data_processor.get_embedding_matrix(embedding_file, onto_aware=True)
                # Put the embedding in a list for Keras to treat it as weights of the embedding layer.
                embedding_weights = [embedding]
                if self.set_sense_priors:
                    initial_sense_prior_parameters = numpy.random.uniform(low=0.01, high=0.99,
                                                                          size=(word_vocab_size, 1))
                    # While setting weights, Keras wants trainable weights first, and then the non trainable
                    # weights. If we are not tuning the embedding, we need to keep the sense priors first.
                    if not self.tune_embedding:
                        embedding_weights = [initial_sense_prior_parameters] + embedding_weights
                    else:
                        embedding_weights.append(initial_sense_prior_parameters)
            self.embedding_layer = OntoAwareEmbedding(word_vocab_size, synset_vocab_size, self.embed_dim,
                                                      weights=embedding_weights, mask_zero=True,
                                                      set_sense_priors=self.set_sense_priors,
                                                      tune_embedding=self.tune_embedding,
                                                      name="embedding")
        return self.embedding_layer

    @overrides
    def _get_encoder_layer(self):
        if self.encoder_layer is None:
            self.encoder_layer = OntoAttentionLSTM(input_dim=self.embed_dim, output_dim=self.embed_dim,
                                                   num_senses=self.num_senses, num_hyps=self.num_hyps,
                                                   use_attention=self.use_attention, consume_less="gpu",
                                                   return_sequences=self.return_sequences, name="onto_lstm")
            if self.bidirectional:
                self.encoder_layer = Bidirectional(self.encoder_layer, name="onto_lstm")
        return self.encoder_layer

    @staticmethod
    def get_custom_objects():
        return {"OntoAttentionLSTM": OntoAttentionLSTM,
                "OntoAwareEmbedding": OntoAwareEmbedding}
