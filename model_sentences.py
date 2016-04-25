import sys
import numpy
import gzip
import time
import theano
import argparse
import pickle
from index_data import DataProcessor
from onto_attention import OntoAttentionLSTM
from keras.models import Model
from keras.layers import Input, Dropout
from keras_extensions import HigherOrderEmbedding

class SentenceModel(object):
  def __init__(self, embed_file, num_senses=2, num_hyps=5):
    self.dp = DataProcessor(word_syn_cutoff=num_senses, syn_path_cutoff=num_hyps)
    self.num_hyps = num_hyps
    self.num_senses = num_senses
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
      vec_min, vec_max = vec.min(), vec.max()
      if vec_min < self.word_rep_min:
        self.word_rep_min = vec_min
      if vec_max > self.word_rep_max:
        self.word_rep_max = vec_max 
      self.word_rep[word] = vec
    self.word_dim = len(vec)
    self.model = None

  def read_sentences(self, tagged_sentences, sentlenlimit=None):
    num_sentences = len(tagged_sentences)
    all_words = []
    all_pos_tags = []
    maxsentlen = 0
    for tagged_sentence in tagged_sentences:
      words = []
      pos_tags = []
      # Expects each token to be a "_" separated combination of word and POS tag.
      tagged_words = tagged_sentence.split(" ")
      if sentlenlimit is not None:
        tagged_words = tagged_words[:sentlenlimit] 
      for word_tag in tagged_words:
        word, tag = word_tag.split("_")
        word = word.lower()
        words.append(word)
        pos_tags.append(tag)
      if len(words) > maxsentlen:
        maxsentlen = len(words)
      all_words.append(words)
      all_pos_tags.append(pos_tags)
    if not sentlenlimit:
      sentlenlimit = maxsentlen
    C_ind = numpy.zeros((num_sentences, sentlenlimit, self.num_senses, self.num_hyps), dtype='int32')
    S_ind = numpy.zeros((num_sentences, sentlenlimit), dtype='int32')
    S = numpy.zeros((num_sentences, sentlenlimit, self.word_dim))
    for i, (words, pos_tags) in enumerate(zip(all_words, all_pos_tags)):
      for word in words:
        if word not in self.word_rep:
          rand_rep = self.numpy_rng.uniform(low=self.word_rep_min, high=self.word_rep_max, size=(self.word_dim))
          self.word_rep[word] = rand_rep
      sentlen = len(words)
      word_inds, syn_inds = self.dp.index_sentence(words, pos_tags)
      S_ind[i][-sentlen:] = word_inds
      for j in range(sentlen):
        S[i][-sentlen+j] = self.word_rep[words[j]]
        sense_syn_ind = syn_inds[j]
        sense_syn_ind_len = len(sense_syn_ind)
        for k, syn_ind in enumerate(sense_syn_ind):
          C_ind[i][-sentlen+j][-sense_syn_ind_len+k][-len(syn_ind):] = syn_ind
    return S, S_ind, C_ind
    
  def train(self, S, C_ind, use_attention=True, num_epochs=20):
    # Predict next word from current synsets
    X = C_ind[:,:-1,:] # remove the last words' hyps in all sentences
    Y = S[:,1:,:] # remove the first words in all sentences
    # Predict current word from current concepts
    #X = C_ind
    #Y = S
    _, length, word_dim = S.shape
    num_words = len(self.dp.word_index)
    num_syns = len(self.dp.synset_index)
    input = Input(shape=C_ind.shape[1:], dtype='int32')
    sent_rep = HigherOrderEmbedding(name='embedding', input_dim=num_syns, output_dim=word_dim, input_shape=C_ind.shape[1:])(input)
    reg_sent_rep = Dropout(0.5)(sent_rep)
    lstm_out = OntoAttentionLSTM(name='sent_lstm', input_dim=word_dim, output_dim=word_dim, input_length=length, num_senses=self.num_senses, num_hyps=self.num_hyps, return_sequences=True, use_attention=True)(reg_sent_rep)
    model = Model(input=input, output=lstm_out)
    print >>sys.stderr, model.summary()
    precompile_time = time.time()
    model.compile(loss='mse', optimizer='adam')
    postcompile_time = time.time()
    print >>sys.stderr, "Model compilation took %d s"%(postcompile_time - precompile_time)
    model.fit(X, Y, nb_epoch=num_epochs)
    posttrain_time = time.time()
    print >>sys.stderr, "Training took %d s"%(posttrain_time - postcompile_time)
    concept_reps = model.layers[1].get_weights()
    self.model = model
    return concept_reps

  def get_attention(self, C_ind):
    if not self.model:
      raise RuntimeError, "Model not trained!"
    model_embedding = None
    model_weights = None
    for layer in self.model.layers:
      if layer.name.lower() == "embedding":
        model_embedding = layer
      if layer.name.lower() == "sent_lstm":
        model_lstm = layer
    if model_embedding is None or model_lstm is None:
      raise RuntimeError, "Did not find expected layers"
    lstm_weights = model_lstm.get_weights()
    embedding_weights = model_embedding.get_weights()
    embed_in_dim, embed_out_dim = embedding_weights[0].shape
    att_embedding = HigherOrderEmbedding(input_dim=embed_in_dim, output_dim=embed_out_dim, weights=embedding_weights)
    onto_lstm = OntoAttentionLSTM(input_dim=embed_out_dim, output_dim=embed_out_dim, input_length=model_lstm.input_length, num_senses=self.num_senses, num_hyps=self.num_hyps, use_attention=True, return_attention=True, weights=lstm_weights)
    att_input = Input(shape=C_ind.shape[1:], dtype='int32')
    att_sent_rep = att_embedding(att_input)
    att_output = onto_lstm(att_sent_rep)
    att_model = Model(input=att_input, output=att_output)
    att_model.compile(optimizer='adam', loss='mse') # optimizer and loss are not needed since we are not going to train this model.
    C_att = att_model.predict(C_ind)
    print >>sys.stderr, "Got attention values. Input, output shapes:", C_ind.shape, C_att.shape
    return C_att

if __name__ == '__main__':
  argparser = argparse.ArgumentParser(description="Model sentences using ontoLSTM")
  argparser.add_argument('repfile', metavar='REP-FILE', type=str, help="Gzipped word embedding file")
  argparser.add_argument('train_file', metavar='TRAIN-FILE', type=str, help="One sentence per line, POS tagged")
  argparser.add_argument('--test_file', type=str, help="Test file for which attention values will be printed. One sentence per line, POS tagged")
  argparser.add_argument('--num_senses', type=int, help="Number of senses per word if using OntoLSTM (default 2)", default=2)
  argparser.add_argument('--num_hyps', type=int, help="Number of hypernyms per sense if using OntoLSTM (default 5)", default=5)
  argparser.add_argument('--use_attention', help="Use attention in ontoLSTM. If this flag is not set, will use average concept representations", action='store_true')
  argparser.add_argument('--synset_embedding_output', type=str, help="Print learned synset representations in the given file")
  argparser.add_argument('--num_epochs', type=int, help="Number of epochs (default 20)", default=20)
  args = argparser.parse_args()
  sm = SentenceModel(args.repfile)
  ts = [x.strip() for x in open(args.train_file).readlines()]
  if args.test_file is not None:
    assert args.use_attention, "Use OntoLSTM with attention to print attention values of the test file"
  S, S_ind, C_ind = sm.read_sentences(ts)
  _, train_sent_len, _, _ = C_ind.shape 
  concept_reps = sm.train(S, C_ind, use_attention=args.use_attention, num_epochs=args.num_epochs)
  if args.synset_embedding_output is not None:
    concrepfile = open(args.synset_embedding_output, "w")
    for syn in sm.dp.synset_index:
      print >>concrepfile, syn, " ".join(["%s"%x for x in concept_reps[0][sm.dp.synset_index[syn]]])
  rev_synset_ind = {ind: syn for (syn, ind) in sm.dp.synset_index.items()}
  if args.test_file is not None:
    outfile_name = args.test_file.split("/")[-1] + ".att_out"
    outfile = open(outfile_name, "w")
    ts_test = [x.strip() for x in open(args.test_file).readlines()]
    _, _, C_ind_test = sm.read_sentences(ts_test, sentlenlimit=train_sent_len)
    C_att = sm.get_attention(C_ind_test)
    for i, (sent, sent_c_inds, sent_c_atts) in enumerate(zip(ts_test, C_ind_test, C_att)):
      print >>outfile, "SENT %d: %s"%(i, sent)
      words = sent.split()[:train_sent_len]
      word_id = 0
      for word_c_inds, word_c_atts in zip(sent_c_inds, sent_c_atts):
        if word_c_inds.sum() == 0:
          continue
        sense_id = 0
        print >>outfile, "Attention for %s"%(words[word_id])
        word_id += 1
        for s_h_ind, s_h_att in zip(word_c_inds, word_c_atts):
          if sum(s_h_ind) == 0:
            continue
          print >>outfile, "\nSense %d"%(sense_id)
          sense_id += 1
          for h_ind, h_att in zip(s_h_ind, s_h_att):
            print >>outfile, rev_synset_ind[h_ind], h_att 
        print >>outfile
      print >>outfile
