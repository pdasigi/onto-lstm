from keras.models import Model
from keras.layers import Input, Dense

from index_data import DataProcessor
from encoders import OntoLSTMEncoder

## Defining constants
NUM_SENSES = 3
NUM_HYPS = 5
ONTO_ATTENTION = True
SENSE_PRIORS = True
EMBED_DIM = 50
BIDIRECTIONAL = False
TUNE_EMBEDDING = True
EMBEDDING_FILE = None  # Replace with a gzipped embedding file if needed.

## Reading text file
test_file = open('data/test_data.tsv')
labeled_sentences = [x.strip().split('\t') for x in test_file]
labels, tagged_sentences = zip(*labeled_sentences)

## Preparing (indexing) data for classification.
# word_syn_cutoff is the number of senses per word,
# and syn_path_cutoff is the number of hypernyms per sense
data_processor = DataProcessor(word_syn_cutoff=NUM_SENSES, syn_path_cutoff=NUM_HYPS)
indexed_input = data_processor.prepare_input(tagged_sentences, onto_aware=True)
one_hot_labels = data_processor.make_one_hot([int(x) for x in labels])

## Defining Keras model
input_layer = Input(shape=indexed_input.shape[1:], dtype='int32')
onto_lstm = OntoLSTMEncoder(num_senses=NUM_SENSES, num_hyps=NUM_HYPS, use_attention=ONTO_ATTENTION,
                            set_sense_priors=SENSE_PRIORS, data_processor=data_processor,
                            embed_dim=EMBED_DIM, return_sequences=False, bidirectional=BIDIRECTIONAL,
                            tune_embedding=TUNE_EMBEDDING)
encoded_input = onto_lstm.get_encoded_phrase(input_layer, embedding_file=EMBEDDING_FILE)
softmax_layer = Dense(2, activation='softmax')
output_predictions = softmax_layer(encoded_input)
model = Model(input=input_layer, output=output_predictions)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

## Training
model.fit(indexed_input, one_hot_labels, validation_split=0.2, nb_epoch=1)
