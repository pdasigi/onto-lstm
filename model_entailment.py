import sys
import numpy
import gzip
import argparse
import random
import theano
from index_data import DataProcessor
from onto_attention import OntoAttentionLSTM
from keras.models import Graph, Sequential
from keras.layers.core import Activation, Dense, Dropout
from keras_extensions import HigherOrderEmbedding
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding

class EntailmentModel(object):
  def __init__(self, embed_file, num_senses=2, num_hyps=5):
    self.dp = DataProcessor(word_syn_cutoff=num_senses, syn_path_cutoff=num_hyps)
    self.max_hyps_per_word = num_senses * num_hyps
    self.numpy_rng = numpy.random.RandomState(12345)
    self.word_rep = {}
    self.word_rep_max = -float("inf")
    self.word_rep_min = float("inf")
    for line in gzip.open(embed_file):
      ln_parts = line.strip().split()
      if len(ln_parts) == 2:
        continue
      word = ln_parts[0]
      vec = numpy.asarray([float(f) for f in ln_parts[1:]])
      vec_max, vec_min = vec.max(), vec.min()
      if vec_max > self.word_rep_max:
        self.word_rep_max = vec_max
      if vec_min < self.word_rep_min:
        self.word_rep_min = vec_min
      self.word_rep[word] = vec
    self.word_dim = len(vec)
    self.model = None

  def read_sentences(self, tagged_sentences, sentlenlimit=None):
    num_sentences = len(tagged_sentences)
    all_words = []
    all_pos_tags = []
    maxsentlen = 0
    for tagged_sentence in tagged_sentences:
      sent1_words = []
      sent1_pos_tags = []
      sent2_words = []
      sent2_pos_tags = []
      in_first_sent = True
      # Expects each token to be a "_" separated combination of word and POS tag.
      for word_tag in tagged_sentence.split(" "):
        if word_tag == "|||":
          in_first_sent = False
          continue
        else:
          word, tag = word_tag.split("_")
        word = word.lower()
        if in_first_sent:
          sent1_words.append(word)
          sent1_pos_tags.append(tag)
        else:
          sent2_words.append(word)
          sent2_pos_tags.append(tag)
      if len(sent1_words) > maxsentlen:
        maxsentlen = len(sent1_words)
      if len(sent2_words) > maxsentlen:
        maxsentlen = len(sent2_words)
      all_words.append((sent1_words, sent2_words))
      all_pos_tags.append((sent1_pos_tags, sent2_pos_tags))
    if not sentlenlimit:
      sentlenlimit = maxsentlen
    C1_ind = numpy.zeros((num_sentences, sentlenlimit, self.max_hyps_per_word), dtype='int32')
    S1_ind = numpy.zeros((num_sentences, sentlenlimit), dtype='int32')
    C2_ind = numpy.zeros((num_sentences, sentlenlimit, self.max_hyps_per_word), dtype='int32')
    S2_ind = numpy.zeros((num_sentences, sentlenlimit), dtype='int32')
    S1 = numpy.zeros((num_sentences, sentlenlimit, self.word_dim))
    S2 = numpy.zeros((num_sentences, sentlenlimit, self.word_dim))
    for i, ((sent1_words, sent2_words), (sent1_pos_tags, sent2_pos_tags)) in enumerate(zip(all_words, all_pos_tags)):
      for word in sent1_words + sent2_words:
        if word not in self.word_rep:
          rand_rep = self.numpy_rng.uniform(low=self.word_rep_min, high=self.word_rep_max, size=(self.word_dim))
          self.word_rep[word] = rand_rep
      # Sentence 1 processing
      sent1len = len(sent1_words)
      sent1_word_inds, sent1_syn_inds = self.dp.index_sentence(sent1_words, sent1_pos_tags)
      S1_ind[i][-sent1len:] = sent1_word_inds
      for j in range(sent1len):
        S1[i][-sent1len+j] = self.word_rep[sent1_words[j]]
        syn_ind = sent1_syn_inds[j]
        C1_ind[i][-sent1len+j][-len(syn_ind):] = syn_ind
      # Sentence 2 processing
      sent2len = len(sent2_words)
      sent2_word_inds, sent2_syn_inds = self.dp.index_sentence(sent2_words, sent2_pos_tags)
      S2_ind[i][-sent2len:] = sent2_word_inds
      for j in range(sent2len):
        S2[i][-sent2len+j] = self.word_rep[sent2_words[j]]
        syn_ind = sent2_syn_inds[j]
        C2_ind[i][-sent2len+j][-len(syn_ind):] = syn_ind
    return (S1, S2), (S1_ind, S2_ind), (C1_ind, C2_ind)

  def train(self, S1_ind, S2_ind, C1_ind, C2_ind, label_ind, num_label_types, train_size,ontoLSTM=False, use_attention=False, num_epochs=20):
    word_dim = 50
    assert S1_ind.shape == S2_ind.shape
    assert C1_ind.shape == C2_ind.shape
    num_words = len(self.dp.word_index)
    num_syns = len(self.dp.synset_index)
    length = C1_ind.shape[1]
    label_onehot = numpy.zeros((len(label_ind), num_label_types))
    for i, ind in enumerate(label_ind):
      label_onehot[i][ind] = 1.0
    model = Graph()
    if ontoLSTM:
      print >>sys.stderr, "Using OntoLSTM"
      model.add_input(name='sent1', input_shape=C1_ind.shape[1:])
      model.add_input(name='sent2', input_shape=C2_ind.shape[1:])
      embedding = HigherOrderEmbedding(input_dim=num_syns, output_dim=word_dim)
      model.add_shared_node(embedding, name='sent_embedding', inputs=['sent1', 'sent2'], outputs=['sent1_embedding', 'sent2_embedding'])
      model.add_node(Dropout(0.5), name="sent1_dropout", input='sent1_embedding')
      model.add_node(Dropout(0.5), name="sent2_dropout", input='sent2_embedding')
      lstm = OntoAttentionLSTM(input_dim=word_dim, output_dim=word_dim/2, input_length=length, num_hyps=self.max_hyps_per_word, use_attention=use_attention)
      model.add_shared_node(lstm, name='sent_lstm', inputs=['sent1_dropout', 'sent2_dropout'], outputs=['sent1_lstm', 'sent2_lstm'])
      model.add_node(Dense(output_dim=num_label_types, activation='softmax'), name='label_probs', inputs=['sent1_lstm', 'sent2_lstm'], merge_mode='concat')
      model.add_output(name='output', input='label_probs')
      print >>sys.stderr, model.summary()
      model.compile(optimizer='adam', loss={'output': 'categorical_crossentropy'})
      for _ in range(num_epochs):
        model.fit({'sent1': C1_ind[:train_size], 'sent2': C2_ind[:train_size], 'output': label_onehot[:train_size]}, nb_epoch=1)
        train_probs = model.predict({'sent1': C1_ind[:train_size], 'sent2': C2_ind[:train_size]})['output']
        valid_probs = model.predict({'sent1': C1_ind[train_size:], 'sent2': C2_ind[train_size:]})['output']
        train_preds = numpy.argmax(train_probs, axis=1)
        train_labels = numpy.argmax(label_onehot[:train_size], axis=1)
        valid_preds = numpy.argmax(valid_probs, axis=1)
        valid_labels = numpy.argmax(label_onehot[train_size:], axis=1)
        print >>sys.stderr, "Train accuracy", sum(train_preds == train_labels) / float(len(train_preds))
        print >>sys.stderr, "Valid accuracy", sum(valid_preds == valid_labels) / float(len(valid_preds))
      self.model = model
    else:
      print >>sys.stderr, "Using traditional LSTM"
      model.add_input(name='sent1', input_shape=S1_ind.shape[1:], dtype='int')
      model.add_input(name='sent2', input_shape=S2_ind.shape[1:], dtype='int')
      embedding = Embedding(input_dim=num_words, output_dim=word_dim)
      model.add_shared_node(embedding, name='sent_embedding', inputs=['sent1', 'sent2'], outputs=['sent1_embedding', 'sent2_embedding'])
      model.add_node(Dropout(0.5), name="sent1_dropout", input='sent1_embedding')
      model.add_node(Dropout(0.5), name="sent2_dropout", input='sent2_embedding')
      lstm = LSTM(input_dim=word_dim, output_dim=word_dim/2, input_length=length)
      model.add_shared_node(lstm, name='sent_lstm', inputs=['sent1_dropout', 'sent2_dropout'], outputs=['sent1_lstm', 'sent2_lstm'])
      model.add_node(Dense(output_dim=num_label_types, activation='softmax'), name='label_probs', inputs=['sent1_lstm', 'sent2_lstm'], merge_mode='concat')
      model.add_output(name='output', input='label_probs')
      print >>sys.stderr, model.summary()
      model.compile(optimizer='adam', loss={'output': 'categorical_crossentropy'})
      for _ in range(num_epochs):
        model.fit({'sent1': S1_ind[:train_size], 'sent2': S2_ind[:train_size], 'output': label_onehot[:train_size]}, nb_epoch=1)
        train_probs = model.predict({'sent1': S1_ind[:train_size], 'sent2': S2_ind[:train_size]})['output']
        valid_probs = model.predict({'sent1': S1_ind[train_size:], 'sent2': S2_ind[train_size:]})['output']
        train_preds = numpy.argmax(train_probs, axis=1)
        train_labels = numpy.argmax(label_onehot[:train_size], axis=1)
        valid_preds = numpy.argmax(valid_probs, axis=1)
        valid_labels = numpy.argmax(label_onehot[train_size:], axis=1)
        print >>sys.stderr, "Train accuracy", sum(train_preds == train_labels) / float(len(train_preds))
        print >>sys.stderr, "Valid accuracy", sum(valid_preds == valid_labels) / float(len(valid_preds))
      self.model = model

  def get_attention(self, C_ind):
    if not self.model:
      raise RuntimeError, "Model not trained!"
    model_embedding = None
    model_lstm = None
    for node_name in self.model.nodes:
      if node_name == "sent_embedding":
        model_embedding = self.model.nodes[node_name]
      if node_name == "sent_lstm":
        model_lstm = self.model.nodes[node_name].layer
    if not model_embedding or not model_lstm:
      raise RuntimeError, "Did not find the layers expected"
    embedding_weights = model_embedding.get_weights()
    lstm_weights = model_lstm.get_weights()
    att_model = Sequential()
    embed_in_dim, embed_out_dim = embedding_weights[0].shape
    att_model.add(HigherOrderEmbedding(input_dim=embed_in_dim, output_dim=embed_out_dim, weights=embedding_weights)) 
    att_model.add(OntoAttentionLSTM(input_dim=embed_out_dim, output_dim=embed_out_dim/2, input_length=model_lstm.input_length, num_hyps=self.max_hyps_per_word, use_attention=model_lstm.use_attention, weights=lstm_weights))
    sym_input = att_model.get_input()
    sym_output = att_model.layers[-1].get_attention()
    att_f = theano.function([sym_input], sym_output)
    C_att = att_f(C_ind)
    print >>sys.stderr, "Got attention values. Input, output shapes:", C_ind.shape, C_att.shape
    return C_att

if __name__ == "__main__":
  argparser = argparse.ArgumentParser(description="Train entailment model using ontoLSTM or traditional LSTM")
  argparser.add_argument('repfile', metavar='REP-FILE', type=str, help="Gzipped word embedding file")
  argparser.add_argument('train_file', metavar='TRAIN-FILE', type=str, help="TSV file with label, premise, hypothesis in three columns")
  argparser.add_argument('--use_onto_lstm', help="Use ontoLSTM. If this flag is not set, will use traditional LSTM", action='store_true')
  argparser.add_argument('--num_senses', type=int, help="Number of senses per word if using OntoLSTM (default 2)", default=2)
  argparser.add_argument('--num_hyps', type=int, help="Number of hypernyms per sense if using OntoLSTM (default 5)", default=5)
  argparser.add_argument('--use_attention', help="Use attention in ontoLSTM. If this flag is not set, will use average concept representations", action='store_true')
  argparser.add_argument('--attention_output', type=str, help="Print attention values of the validation data in the given file")
  argparser.add_argument('--num_epochs', type=int, help="Number of epochs (default 20)", default=20)
  args = argparser.parse_args()
  em = EntailmentModel(args.repfile, num_senses=args.num_senses, num_hyps=args.num_hyps)
  tagged_sentences = []
  label_map = {}
  label_ind = []
  for line in open(args.train_file):
    lnstrp = line.strip()
    label, tagged_sentence = lnstrp.split("\t")
    if label not in label_map:
      label_map[label] = len(label_map)
    label_ind.append(label_map[label])
    tagged_sentences.append(tagged_sentence)
  sentence_labels = zip(tagged_sentences, label_ind)
  random.shuffle(sentence_labels)
  tagged_sentences, label_ind = zip(*sentence_labels)
  _, (S1_ind, S2_ind), (C1_ind, C2_ind) = em.read_sentences(tagged_sentences)
  train_size = int(0.9 * C1_ind.shape[0])
  em.train(S1_ind, S2_ind, C1_ind, C2_ind, label_ind, len(label_map), train_size, ontoLSTM=args.use_onto_lstm, use_attention=args.use_attention, num_epochs=args.num_epochs)

  if args.attention_output is not None:
    rev_synset_ind = {ind: syn for (syn, ind) in em.dp.synset_index.items()}
    C_ind = numpy.concatenate([C1_ind[train_size:], C2_ind[train_size:]])
    C_att = em.get_attention(C_ind)
    C1_att, C2_att = numpy.split(C_att, 2)
    # Concatenate sentence 1 and 2 in each data point
    C_sj_ind = numpy.concatenate([C1_ind[train_size:], C2_ind[train_size:]], axis=1)
    C_sj_att = numpy.concatenate([C1_att, C2_att], axis=1)
    outfile = open(args.attention_output, "w")
    for i, (sent, sent_c_inds, sent_c_atts) in enumerate(zip(tagged_sentences[train_size:], C_sj_ind, C_sj_att)):
      print >>outfile, "SENT %d: %s"%(i, sent)
      for word_c_inds, word_c_atts in zip(sent_c_inds, sent_c_atts):
        if sum(word_c_inds) == 0:
          continue
        for c_ind, c_att in zip(word_c_inds, word_c_atts):
          if c_ind == 0:
            continue
          print >>outfile, rev_synset_ind[c_ind], c_att 
        print >>outfile
      print >>outfile
