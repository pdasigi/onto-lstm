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
        raise NotImplementedError

    @staticmethod
    def get_custom_objects():
        return {}


class LSTMEncoder(Encoder):
    @overrides
    def get_encoded_phrase(self, phrase_input_layer, dropout={}, embedding_file=None):
        if embedding_file is None:
            if not self.tune_embedding:
                print >>sys.stderr, "Pretrained embedding is not given. Setting tune_embedding to True."
                self.tune_embedding = True
            embedding = None
        else:
            # Put the embedding in a list for Keras to treat it as initiali weights of the embeddign layer.
            embedding = [self.data_processor.get_embedding_matrix(embedding_file, onto_aware=False)]
        vocab_size = self.data_processor.get_vocab_size(onto_aware=False)
        embedding_layer = Embedding(input_dim=vocab_size, output_dim=self.embed_dim, weights=embedding,
                                    trainable=self.tune_embedding, mask_zero=True, name="embedding")
        embedded_phrase = embedding_layer(phrase_input_layer)
        embedding_dropout = dropout.pop("embedding", 0.0)
        if embedding_dropout > 0:
            embedded_phrase = Dropout(embedding_dropout)(embedded_phrase)
        encoder = LSTM(input_dim=self.embed_dim, output_dim=self.embed_dim,
                       return_sequences=self.return_sequences, name="encoder")
        if self.bidirectional:
            encoder = Bidirectional(encoder, name="encoder")
        encoded_phrase = encoder(embedded_phrase)
        encoder_dropout = dropout.pop("encoder", 0.0)
        if encoder_dropout > 0:
            encoded_phrase = Dropout(encoder_dropout)(encoded_phrase)
        return encoded_phrase


class OntoLSTMEncoder(Encoder):
    def __init__(self, num_senses, num_hyps, use_attention, set_sense_priors, **kwargs):
        self.num_senses = num_senses
        self.num_hyps = num_hyps
        self.use_attention = use_attention
        self.set_sense_priors = set_sense_priors
        super(OntoLSTMEncoder, self).__init__(**kwargs)

    @overrides
    def get_encoded_phrase(self, phrase_input_layer, dropout={}, embedding_file=None):
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
            # Put the embedding in a list for Keras to treat it as initial weights of the embedding layer.
            embedding_weights = [embedding]
            if self.set_sense_priors and self.tune_embedding:
                initial_sense_prior_parameters = numpy.random.uniform(low=0.01, high=0.99,
                                                                      size=(word_vocab_size, 1))
                embedding_weights.append(initial_sense_prior_parameters)
        embedding_layer = OntoAwareEmbedding(word_vocab_size, synset_vocab_size, self.embed_dim,
                                             weights=embedding_weights, mask_zero=True,
                                             set_sense_priors=self.set_sense_priors,
                                             trainable=self.tune_embedding, name="embedding")
        embedded_phrase = embedding_layer(phrase_input_layer)
        embedding_dropout = dropout.pop("embedding", 0.0)
        if embedding_dropout > 0.0:
            embedded_phrase = Dropout(embedding_dropout)(embedded_phrase)
        encoder = OntoAttentionLSTM(input_dim=self.embed_dim, output_dim=self.embed_dim, num_senses=self.num_senses,
                                    num_hyps=self.num_hyps, use_attention=self.use_attention, consume_less="gpu",
                                    return_sequences=self.return_sequences, name="onto_lstm")
        if self.bidirectional:
            encoder = Bidirectional(encoder, name="onto_lstm")
        encoded_phrase = encoder(embedded_phrase)
        encoder_dropout = dropout.pop("encoder", 0.0)
        if encoder_dropout > 0.0:
            encoded_phrase = Dropout(encoder_dropout)(encoded_phrase)
        return encoded_phrase

    @staticmethod
    def get_custom_objects():
        return {"OntoAttentionLSTM": OntoAttentionLSTM,
                "OntoAwareEmbedding": OntoAwareEmbedding}
