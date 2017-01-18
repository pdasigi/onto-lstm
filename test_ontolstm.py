import numpy
from embedding import OntoAwareEmbedding
from keras.layers import Input
from keras.models import Model

from onto_attention import OntoAttentionLSTM

num_samples = 100
num_senses = 3
num_hyps = 5
length = 10
embedding_size = 50
lstm_output_dim = 20

word_vocab_size = 100
synset_vocab_size = 80

# num_hyps + 1 because the word index is added as the last index along all senses.
# See the docstring in embedding.py for more information.
input_layer = Input(shape=(length, num_senses, num_hyps+1), dtype='int32')
embedding = OntoAwareEmbedding(word_vocab_size, synset_vocab_size, embedding_size, mask_zero=True)
lstm = OntoAttentionLSTM(lstm_output_dim, num_senses, num_hyps, return_sequences=True)
input_values = numpy.random.randint(low=0, high=synset_vocab_size, size=(num_samples, length, num_senses, num_hyps+1))

embedded_input = embedding(input_layer)
lstm_output = lstm(embedded_input)

model = Model(input=input_layer, output=lstm_output)
model.summary()
model.compile(loss='mse', optimizer='sgd')

output_values = numpy.random.rand(num_samples, length, lstm_output_dim)

model.fit(input_values, output_values)
