import sys
import argparse
import pickle
import random
import numpy

from keras.models import Model, load_model
from keras.layers import Dropout, Embedding, Input, LSTM, Bidirectional

from embedding import OntoAwareEmbedding
from index_data import DataProcessor
from onto_attention import OntoAttentionLSTM
from attachment_predictor import AttachmentPredictor

class PPAttachmentModel(object):
    def __init__(self, bidirectional=False, tune_embedding=False, **kwargs):
        self.data_processor = DataProcessor()
        self.embed_dim = kwargs.pop("embed_dim", 50)
        self.bidirectional = bidirectional
        self.numpy_rng = numpy.random.RandomState(12345)
        self.model = None
        self.best_epoch = 0  # index of the best epoch
        self.tune_embedding = tune_embedding
        self.model_name_prefix = None
        self.custom_objects = {"AttachmentPredictor": AttachmentPredictor}

    def train(self, train_inputs, train_labels, num_epochs=20, embedding_file=None, num_mlp_layers=0,
              dropout=None, patience=6):
        '''
        train_inputs (numpy_array): Indexed Head + preposition + child
        train_labels (numpy_array): One-hot matrix indicating labels
        num_epochs (int): Maximum number of epochs to run
        embedding (numpy): Optional pretrained embedding
        num_mlp_layers (int): Number of layers between the encoded inputs and the final softmax
        dropout (dict): Dict containing dropout values after "embedding" and "encoder"
        patience (int): Early stopping patience
        '''
        phrase_input_layer = Input(name='phrase', shape=train_inputs.shape[1:], dtype='int32')
        encoded_phrase = self._get_encoded_phrase(phrase_input_layer, dropout, embedding_file)
        attachment_probs = AttachmentPredictor(name='attachment_predictor', proj_dim=20,
                                               num_hidden_layers=num_mlp_layers)(encoded_phrase)
        model = Model(input=phrase_input_layer, output=attachment_probs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        print >>sys.stderr, "PP Attachment model summary:"
        model.summary()
        best_accuracy = 0.0
        num_worse_epochs = 0
        for epoch_id in range(num_epochs):
            print >>sys.stderr, "Epoch: %d" % epoch_id
            history = model.fit(train_inputs, train_labels, validation_split=0.05, nb_epoch=1)
            validation_accuracy = history.history['val_acc'][0]  # history['val_acc'] is a list of size nb_epoch
            if validation_accuracy > best_accuracy:
                self.save_model(epoch_id)
                self.best_epoch = epoch_id
                num_worse_epochs = 0
                best_accuracy = validation_accuracy
            elif validation_accuracy < best_accuracy:
                num_worse_epochs += 1
                if num_worse_epochs >= patience:
                    print >>sys.stderr, "Stopping training."
                    break
        self.save_best_model()

    def _get_encoded_phrase(self, phrase_input_layer, dropout, embedding_file):
        # Subclasses implement their own ways of encoding sentences.
        raise NotImplementedError

    @staticmethod
    def _make_one_hot(indices):
        # Accepts an array of indices, and converts them to a one-hot matrix
        num_classes = max(indices)  # Assuming that the range is [1, max]
        one_hot_indices = numpy.zeros((len(indices), num_classes))
        for i, ind in enumerate(indices):
            one_hot_indices[i][ind-1] = 1.0
        return one_hot_indices

    def process_data(self, input_file, onto_aware, for_test=False):
        dataset_type = "test" if for_test else "training"
        print >>sys.stderr, "Reading %s data" % dataset_type
        label_ind = []
        tagged_sentences = []
        max_sentence_length = 0
        all_sentence_lengths = []
        for line in open(input_file):
            lnstrp = line.strip()
            label, tagged_sentence = lnstrp.split("\t")
            sentence_length = len(tagged_sentence.split())
            all_sentence_lengths.append(sentence_length)
            if sentence_length > max_sentence_length:
                max_sentence_length = sentence_length
            label_ind.append(int(label))
            tagged_sentences.append(tagged_sentence)
        if for_test:
            if not self.model:
                raise RuntimeError, "Model not trained yet!"
            input_shape = self.model.get_input_shape_at(0)  # (num_sentences, num_words, ...)
            sentlenlimit = input_shape[1]
        else:
            sentlenlimit = max_sentence_length
        # We need to readjust the labels because padding would affect the sentence indices.
        for i in range(len(label_ind)):
            length = all_sentence_lengths[i]
            label_ind[i] += sentlenlimit - length
        if not for_test:
            # Shuffling so that when Keras does validation split, it is not always at the end.
            sentences_and_labels = zip(tagged_sentences, label_ind)
            random.shuffle(sentences_and_labels)
            tagged_sentences, label_ind = zip(*sentences_and_labels)
        print >>sys.stderr, "Indexing %s data" % dataset_type
        inputs = self.data_processor.prepare_input(tagged_sentences, onto_aware=onto_aware, sentlenlimit=sentlenlimit,
                                                   for_test=for_test, remove_singletons=False)
        labels = self._make_one_hot(label_ind)
        return inputs, labels

    def test(self, inputs, targets):
        if not self.model:
            raise RuntimeError, "Model not trained!"
        metrics = self.model.evaluate(inputs, targets)
        print >>sys.stderr, "Test accuracy: %.4f" % (metrics[1])  # The first metric is loss.
        predictions = numpy.argmax(self.model.predict(inputs), axis=1)
        test_output_file = open("%s.predictions" % self.model_name_prefix, "w")
        for input_indices, prediction in zip(inputs, predictions):
            # The predictions are indices of words in padded sentences. We need to readjust them.
            padding_length = 0
            for index in input_indices: 
                if numpy.all(index == 0):
                    padding_length += 1
                else:
                    break
            prediction = prediction - padding_length + 1  # +1 because the indices start at 1.
            print >>test_output_file, prediction

    def save_model(self, epoch):
        '''
        Saves the current model using the epoch id to identify the file.
        '''
        self.model.save("%s_%d.model" % (self.model_name_prefix, epoch))
        pickle.dump(self.data_processor, open("%s.dataproc" % self.model_name_prefix, "wb"))

    def save_best_model(self):
        '''
        Copies the model corresponding to the best epoch as the final model file.
        '''
        from shutil import copyfile
        best_model_file = "%s_%d.model" % (self.model_name_prefix, self.best_epoch)
        final_model_file = "%s.model" % self.model_name_prefix
        copyfile(best_model_file, final_model_file)

    def load_model(self, epoch=None):
        '''
        Loads a saved model. If epoch id is provided, will load the corresponding model. Or else,
        will load the best model.
        '''
        if not epoch:
            self.model = load_model("%s.model" % self.model_name_prefix,
                                    custom_objects=self.custom_objects)
        else:
            self.model = load_model("%s_%d.model" % (self.model_name_prefix, epoch),
                                    custom_objects=self.custom_objects)
        self.data_processor = pickle.load(open("%s.dataproc" % self.model_name_prefix, "rb"))


class LSTMAttachmentModel(PPAttachmentModel):
    def __init__(self, **kwargs):
        super(LSTMAttachmentModel, self).__init__(**kwargs)
        self.model_name_prefix = "lstm_ppa_tune-embedding=%s_bi=%s" % (self.tune_embedding, self.bidirectional)

    def _get_encoded_phrase(self, phrase_input_layer, dropout, embedding_file):
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
        encoder = LSTM(input_dim=self.embed_dim, output_dim=self.embed_dim, return_sequences=True, name="encoder")
        if self.bidirectional:
            encoder = Bidirectional(encoder, name="encoder")
        encoded_phrase = encoder(embedded_phrase)
        encoder_dropout = dropout.pop("encoder", 0.0)
        if encoder_dropout > 0:
            encoded_phrase = Dropout(encoder_dropout)(encoded_phrase)
        return encoded_phrase


class OntoLSTMAttachmentModel(PPAttachmentModel):
    def __init__(self, num_senses, num_hyps, use_attention, set_sense_priors, **kwargs):
        super(OntoLSTMAttachmentModel, self).__init__(**kwargs)
        # Set self.data_processor again, now with the right arguments.
        self.data_processor = DataProcessor(word_syn_cutoff=num_senses, syn_path_cutoff=num_hyps)
        self.num_senses = num_senses
        self.num_hyps = num_hyps
        self.attention_model = None  # Keras model with just embedding and encoder to output attention.
        self.set_sense_priors = set_sense_priors
        self.use_attention = use_attention
        self.model_name_prefix = "ontolstm_ppa_att=%s_senses=%d_hyps=%d_sense-priors=%s_tune-embedding=%s_bi=%s" % (
            str(self.use_attention), self.num_senses, self.num_hyps, str(set_sense_priors), str(self.tune_embedding),
            str(self.bidirectional))
        self.custom_objects.update({"OntoAttentionLSTM": OntoAttentionLSTM, "OntoAwareEmbedding": OntoAwareEmbedding})

    def _get_encoded_phrase(self, phrase_input_layer, dropout, embedding_file):
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
                initial_sense_prior_parameters = numpy.random.uniform(low=0.01, high=0.99, size=(word_vocab_size, 1))
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
                                    return_sequences=True, name="onto_lstm")
        if self.bidirectional:
            encoder = Bidirectional(encoder, name="onto_lstm")
        encoded_phrase = encoder(embedded_phrase)
        encoder_dropout = dropout.pop("encoder", 0.0)
        if encoder_dropout > 0.0:
            encoded_phrase = Dropout(encoder_dropout)(encoded_phrase)
        return encoded_phrase

    def get_attention(self, inputs):
        # Takes inputs and returns pairs of synsets and corresponding attention values.
        if not self.attention_model:
            self.define_attention_model()
        attention_outputs = self.attention_model.predict(inputs)
        sent_attention_values = []
        for sentence_input, sentence_attention in zip(inputs, attention_outputs):
            word_attention_values = []
            for word_input, word_attention in zip(sentence_input, sentence_attention):
                if word_input.sum() == 0:
                    # This is just padding
                    continue
                sense_attention_values = []
                for sense_input, sense_attention in zip(word_input, word_attention):
                    if sense_input.sum() == 0:
                        continue
                    hyp_attention_values = []
                    for hyp_input, hyp_attention in zip(sense_input, sense_attention):
                        if hyp_input == 0:
                            continue
                        hyp_attention_values.append((self.data_processor.get_token_from_index(hyp_input,
                                                        onto_aware=True), hyp_attention))
                    sense_attention_values.append(hyp_attention_values)
                word_attention_values.append(sense_attention_values)
            sent_attention_values.append(word_attention_values)
        return sent_attention_values

    """
    def define_attention_model(self):
        # Take necessary parts out of the entailment model to get OntoLSTM attention.
        if not self.model:
            raise RuntimeError, "Model not trained yet!"
        input_shape = self.model.get_input_shape_at(0)
        input_layer = Input(input_shape[1:], dtype='int32')  # removing batch size
        embedding_layer = None
        encoder_layer = None
        for layer in self.model.layers:
            if layer.name == "embedding":
                embedding_layer = layer
            elif layer.name == "encoder":
                # We need to redefine the OntoLSTM layer with the learned weights and set return attention to True.
                # Assuming we'll want attention values for all words (return_sequences = True)
                encoder_layer = OntoAttentionLSTM(input_dim=self.embed_dim,
                                                  output_dim=self.embed_dim, num_senses=self.num_senses,
                                                  num_hyps=self.num_hyps, use_attention=True,
                                                  return_attention=True, return_sequences=True,
                                                  weights=layer.get_weights())
        if not embedding_layer or not encoder_layer:
            raise RuntimeError, "Required layers not found!"
        attention_output = encoder_layer(embedding_layer(input_layer))
        self.attention_model = Model(input=input_layer, output=attention_output)
        self.attention_model.compile(loss="mse", optimizer="sgd")  # Loss and optimizer do not matter!

    def print_attention_values(self, input_file, test_inputs, output_file):
        onto_aware = True
        sent1_attention_outputs = self.get_attention(test_inputs[0])
        sent2_attention_outputs = self.get_attention(test_inputs[1])
        tagged_sentences = [x.strip().split("\t")[1] for x in codecs.open(input_file).readlines()]
        outfile = codecs.open(output_file, "w", "utf-8")
        for sent1_attention, sent2_attention, tagged_sentence in zip(sent1_attention_outputs, sent2_attention_outputs, tagged_sentences):
            print >>outfile, tagged_sentence
            print >>outfile, "Sentence 1:"
            for word_attention in sent1_attention:
                for sense_attention in word_attention:
                    print >>outfile, " ".join(["%s:%f" % (hyp, hyp_att) for hyp, hyp_att in sense_attention])
                print >>outfile
            print >>outfile, "\nSentence 2:"
            for word_attention in sent2_attention:
                for sense_attention in word_attention:
                    print >>outfile, " ".join(["%s:%f" % (hyp, hyp_att) for hyp, hyp_att in sense_attention])
                print >>outfile
        outfile.close()
    """

def main():
    argparser = argparse.ArgumentParser(description="Train preposition phrase attachment model")
    argparser.add_argument('--train_file', type=str, help="TSV file with label and pos tagged phrase")
    argparser.add_argument('--embedding_file', type=str, help="Gzipped embedding file")
    argparser.add_argument('--embed_dim', type=int, help="Word/Synset vector size", default=50)
    argparser.add_argument('--bidirectional', help="Encode bidirectionally followed by pooling", action='store_true')
    argparser.add_argument('--onto_aware', help="Use ontology aware encoder. If this flag is not set, will use traditional encoder", action='store_true')
    argparser.add_argument('--num_senses', type=int, help="Number of senses per word if using OntoLSTM (default 2)", default=2)
    argparser.add_argument('--num_hyps', type=int, help="Number of hypernyms per sense if using OntoLSTM (default 5)", default=5)
    argparser.add_argument('--set_sense_priors', help="Set an exponential prior on sense probabilities", action='store_true')
    argparser.add_argument('--use_attention', help="Use attention in ontoLSTM. If this flag is not set, will use average concept representations", action='store_true')
    argparser.add_argument('--test_file', type=str, help="Optionally provide test file for which accuracy will be computed")
    argparser.add_argument('--load_model_from_epoch', type=int, help="Load model from a specific epoch. Will load best model by default.")
    argparser.add_argument('--attention_output', type=str, help="Print attention values of the validation data in the given file")
    argparser.add_argument('--tune_embedding', help="Fine tune pretrained embedding (if provided)", action='store_true')
    argparser.add_argument('--num_epochs', type=int, help="Number of epochs (default 20)", default=20)
    argparser.add_argument('--num_mlp_layers', type=int, help="Number of mlp layers (default 0)", default=0)
    argparser.add_argument('--embedding_dropout', type=float, help="Dropout after embedding", default=0.0)
    argparser.add_argument('--encoder_dropout', type=float, help="Dropout after encoder", default=0.0)
    args = argparser.parse_args()
    if args.onto_aware:
        attachment_model = OntoLSTMAttachmentModel(num_senses=args.num_senses, num_hyps=args.num_hyps,
                                                   use_attention=args.use_attention,
                                                   set_sense_priors=args.set_sense_priors,
                                                   embed_dim=args.embed_dim,
                                                   bidirectional=args.bidirectional,
                                                   tune_embedding=args.tune_embedding)
    else:
        attachment_model = LSTMAttachmentModel(embed_dim=args.embed_dim, bidirectional=args.bidirectional,
                                               tune_embedding=args.tune_embedding)

    ## Train model or load trained model
    if args.train_file is None:
        attachment_model.load_model(args.load_model_from_epoch)
    else:
        train_inputs, train_labels = attachment_model.process_data(args.train_file, onto_aware=args.onto_aware,
                                                                   for_test=False)
        dropout = {"embedding": args.embedding_dropout,
                   "encoder": args.encoder_dropout}
        attachment_model.train(train_inputs, train_labels, num_epochs=args.num_epochs,
                               dropout=dropout, num_mlp_layers=args.num_mlp_layers,
                               embedding_file=args.embedding_file)
    
    ## Test model
    if args.test_file is not None:
        test_inputs, test_labels = attachment_model.process_data(args.test_file, onto_aware=args.onto_aware,
                                                                 for_test=True)
        attachment_model.test(test_inputs, test_labels)
        if args.attention_output is not None:
            raise NotImplementedError
            #attachment_model.print_attention_values(args.test_file, test_inputs, args.attention_output)

if __name__ == "__main__":
    main()
