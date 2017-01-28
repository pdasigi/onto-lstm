import sys
import argparse
import random
import numpy
import pickle
from overrides import overrides

from keras.layers import Input

from encoders import LSTMEncoder, OntoLSTMEncoder
from index_data import DataProcessor
from preposition_model import PrepositionModel
from preposition_predictors import RelationPredictor


class PPRelationModel(PrepositionModel):
    def __init__(self, tune_embedding, bidirectional, **kwargs):
        super(PPRelationModel, self).__init__(**kwargs)
        self.tune_embedding = tune_embedding
        self.bidirectional = bidirectional
        self.num_relation_types = None
        self.model_name = "PP Relation"
        self.label_map = {}
        self.custom_objects = {"RelationPredictor": RelationPredictor}

    def get_input_layers(self, train_inputs):
        sentence_inputs, preposition_indices = train_inputs
        batch_size = preposition_indices.shape[0]
        sentence_input_layer = Input(name="sentence", shape=sentence_inputs.shape[1:], dtype='int32')
        prep_indices_layer = Input(name="prep_indices", shape=(1,), dtype='int32')
        return sentence_input_layer, prep_indices_layer

    def get_output_layers(self, inputs, dropout, embedding_file, num_mlp_layers):
        sentence_input_layer, prep_indices_layer = inputs
        encoded_input = self.encoder.get_encoded_phrase(sentence_input_layer, dropout, embedding_file)
        predictor = RelationPredictor(self.num_relation_types, name='relation_predictor', proj_dim=20,
                                      composition_type='HPCT', num_hidden_layers=num_mlp_layers)
        outputs = predictor([encoded_input, prep_indices_layer])
        return outputs

    @overrides
    def process_data(self, input_file, onto_aware, for_test=False):
        dataset_type = "test" if for_test else "training"
        print >>sys.stderr, "Reading %s data" % dataset_type
        label_indices = []
        prep_indices = []
        tagged_sentences = []
        max_sentence_length = 0
        all_sentence_lengths = []
        for line in open(input_file):
            lnstrp = line.strip()
            tagged_sentence, prep_index, label = lnstrp.split("\t")
            sentence_length = len(tagged_sentence.split())
            all_sentence_lengths.append(sentence_length)
            if sentence_length > max_sentence_length:
                max_sentence_length = sentence_length
            prep_indices.append(int(prep_index))
            if label not in self.label_map:
                # Making indices start at 1 because _make_one_hot expects that.
                self.label_map[label] = len(self.label_map) + 1
            label_indices.append(self.label_map[label])
            tagged_sentences.append(tagged_sentence)
        self.num_relation_types = len(self.label_map)
        if for_test:
            if not self.model:
                raise RuntimeError, "Model not trained yet!"
            input_shape = self.model.get_input_shape_at(0)[0]  # (num_sentences, num_words, ...)
            sentlenlimit = input_shape[1]
        else:
            sentlenlimit = max_sentence_length
        # We need to readjust the prep_indices because padding would affect the sentence indices.
        for i in range(len(prep_indices)):
            length = all_sentence_lengths[i]
            prep_indices[i] += sentlenlimit - length
        if not for_test:
            # Shuffling so that when Keras does validation split, it is not always at the end.
            sentences_indices_labels = zip(tagged_sentences, prep_indices, label_indices)
            random.shuffle(sentences_indices_labels)
            tagged_sentences, prep_indices, label_indices = zip(*sentences_indices_labels)
        print >>sys.stderr, "Indexing %s data" % dataset_type
        sentence_inputs = self.data_processor.prepare_input(tagged_sentences, onto_aware=onto_aware,
                                                            sentlenlimit=sentlenlimit, for_test=for_test,
                                                            remove_singletons=False)
        prep_indices = numpy.asarray(prep_indices)
        labels = self._make_one_hot(label_indices)
        return [sentence_inputs, prep_indices], labels

    @overrides
    def write_predictions(self, inputs):
        rev_label_map = {j: i for (i, j) in self.label_map.items()}
        predictions = numpy.argmax(self.model.predict(inputs), axis=1)
        test_output_file = open("%s.predictions" % self.model_name_prefix, "w")
        for prediction in predictions:
            print >>test_output_file, rev_label_map[prediction + 1]


    @overrides
    def save_model(self, epoch):
        pickle.dump(self.label_map, open("%s.label_map" % self.model_name_prefix, "wb"))
        super(PPRelationModel, self).save_model(epoch)

    @overrides
    def load_model(self, epoch=None):
        self.label_map = pickle.load(open("%s.label_map" % self.model_name_prefix, "rb"))
        super(PPRelationModel, self).load_model(epoch)


class LSTMRelationModel(PPRelationModel):
    def __init__(self, **kwargs):
        super(LSTMRelationModel, self).__init__(**kwargs)
        self.model_name_prefix = "lstm_prep_rel_tune-embedding=%s_bi=%s" % (self.tune_embedding,
                                                                            self.bidirectional)
        self.encoder = LSTMEncoder(self.data_processor, self.embed_dim, self.bidirectional, self.tune_embedding)
        self.custom_objects.update(self.encoder.get_custom_objects())


class OntoLSTMRelationModel(PPRelationModel):
    def __init__(self, num_senses, num_hyps, use_attention, set_sense_priors, prep_senses_dir, **kwargs):
        super(OntoLSTMRelationModel, self).__init__(**kwargs)
        # Set self.data_processor again, now with the right arguments.
        process_preps = False if prep_senses_dir is None else True
        self.data_processor = DataProcessor(word_syn_cutoff=num_senses, syn_path_cutoff=num_hyps,
                                            process_preps=process_preps, prep_senses_dir=prep_senses_dir)
        self.num_senses = num_senses
        self.num_hyps = num_hyps
        self.attention_model = None  # Keras model with just embedding and encoder to output attention.
        self.set_sense_priors = set_sense_priors
        self.use_attention = use_attention
        use_prep_senses = False if prep_senses_dir is None else True
        self.encoder = OntoLSTMEncoder(self.num_senses, self.num_hyps, self.use_attention, self.set_sense_priors,
                                       data_processor=self.data_processor, embed_dim=self.embed_dim,
                                       bidirectional=self.bidirectional, tune_embedding=self.tune_embedding)
        self.model_name_prefix = "ontolstm_prep_rel_att=%s_senses=%d_hyps=%d_sense-priors=%s_prep-senses=%s_tune-embedding=%s_bi=%s" % (
            str(self.use_attention), self.num_senses, self.num_hyps, str(set_sense_priors), str(use_prep_senses), str(self.tune_embedding),
            str(self.bidirectional))
        self.custom_objects.update(self.encoder.get_custom_objects())


def main():
    argparser = argparse.ArgumentParser(description="Train and test preposition relation prediction model")
    argparser.add_argument('--train_file', type=str, help="TSV file with label and pos tagged phrase")
    argparser.add_argument('--embedding_file', type=str, help="Gzipped embedding file")
    argparser.add_argument('--embed_dim', type=int, help="Word/Synset vector size", default=50)
    argparser.add_argument('--bidirectional', help="Encode bidirectionally followed by pooling", action='store_true')
    argparser.add_argument('--onto_aware', help="Use ontology aware encoder. If this flag is not set, will use traditional encoder", action='store_true')
    argparser.add_argument('--num_senses', type=int, help="Number of senses per word if using OntoLSTM (default 2)", default=2)
    argparser.add_argument('--num_hyps', type=int, help="Number of hypernyms per sense if using OntoLSTM (default 5)", default=5)
    argparser.add_argument('--prep_senses_dir', type=str, help="Directory containing preposition senses (from Semeval07 Task 6)")
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
        attachment_model = OntoLSTMRelationModel(num_senses=args.num_senses, num_hyps=args.num_hyps,
                                                 use_attention=args.use_attention,
                                                 set_sense_priors=args.set_sense_priors,
                                                 prep_senses_dir=args.prep_senses_dir,
                                                 embed_dim=args.embed_dim,
                                                 bidirectional=args.bidirectional,
                                                 tune_embedding=args.tune_embedding)
    else:
        attachment_model = LSTMRelationModel(embed_dim=args.embed_dim, bidirectional=args.bidirectional,
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
