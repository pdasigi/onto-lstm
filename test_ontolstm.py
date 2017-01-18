import numpy
from embedding import OntoAwareEmbedding
from keras.layers import Input, LSTM, Embedding
from keras.models import Model

from onto_attention import OntoAttentionLSTM

num_samples = 10
num_senses = 3
num_hyps = 5
length = 10
embedding_size = 50
lstm_output_dim = 20

word_vocab_size = 100
synset_vocab_size = 80

input_layer = Input(shape=(length,), dtype='int32')
embedding = Embedding(word_vocab_size, embedding_size)
lstm = LSTM(lstm_output_dim, return_sequences=True)
input_values = numpy.random.rand(num_samples, length)

embedded_input = embedding(input_layer)
lstm_output = lstm(embedded_input)

model = Model(input=input_layer, output=lstm_output)
model.summary()
model.compile(loss='mse', optimizer='sgd')

output_values = numpy.random.rand(num_samples, length, lstm_output_dim)

model.fit(input_values, output_values)
