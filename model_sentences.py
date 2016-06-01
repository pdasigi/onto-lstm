import sys
import numpy
import gzip
import math
import itertools
import time
import codecs
import argparse
import pickle
from index_data import DataProcessor
from onto_attention import OntoAttentionLSTM
from keras.models import Model, model_from_yaml
from keras.layers import Input, Dropout, LSTM, Dense
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping
from keras_extensions import HigherOrderEmbedding

class SentenceModel(object):
  def __init__(self, word_dim=50, num_senses=2, num_hyps=5):
    self.dp = DataProcessor(word_syn_cutoff=num_senses, syn_path_cutoff=num_hyps)
    self.num_hyps = num_hyps
    self.num_senses = num_senses
    self.numpy_rng = numpy.random.RandomState(12345)
    self.word_dim = word_dim
    self.model = None

  def read_sentences(self, tagged_sentences, sentlenlimit=None, test=False):
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
        parts = word_tag.split("_")
        tag = parts[-1]
        word = "_".join(parts[:-1]).lower()
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
    for i, (words, pos_tags) in enumerate(zip(all_words, all_pos_tags)):
      sentlen = len(words)
      # test=True locks the word and syn index dicts. No new keys will be added
      word_inds, syn_inds = self.dp.index_sentence(words, pos_tags, test=test)
      S_ind[i][-sentlen:] = word_inds
      for j in range(sentlen):
        sense_syn_ind = syn_inds[j]
        sense_syn_ind_len = len(sense_syn_ind)
        for k, syn_ind in enumerate(sense_syn_ind):
          C_ind[i][-sentlen+j][-sense_syn_ind_len+k][-len(syn_ind):] = syn_ind
    return S_ind, C_ind

  def _make_one_hot(self, word_inds, vec_size):
    onehot = numpy.zeros((word_inds.shape + (vec_size,)))
    for inds in itertools.product(*[numpy.arange(s) for s in word_inds.shape]):
      onehot[inds+(word_inds[inds],)] = 1
    return onehot

  def _factor_target_indices(self, Y_inds, vocab_size=None, base=2):
    if vocab_size is None:
      vocab_size = len(self.dp.word_index)
    print >>sys.stderr, "Factoring targets of vocabulary size: %d"%(vocab_size)
    num_vecs = int(math.ceil(math.log(vocab_size)/math.log(base))) + 1
    base_inds = []
    div_Y_inds = Y_inds
    print >>sys.stderr, "Number of factors: %d"%num_vecs
    for i in range(num_vecs):
      new_inds = div_Y_inds % base
      if i == num_vecs - 1:
        if new_inds.sum() == 0:
          # Most significant "digit" is a zero. Omit it.
          break
      base_inds.append(new_inds)
      div_Y_inds = numpy.copy(div_Y_inds/base)
    base_vecs = [self._make_one_hot(base_inds_i, base) for base_inds_i in base_inds]
    return base_vecs
    
  def train(self, S_ind, C_ind, use_onto_lstm=True, use_attention=True, num_epochs=20,  hierarchical=False, base=2):
    # Predict next word from current synsets
    X = C_ind[:,:-1] if use_onto_lstm else S_ind[:,:-1] # remove the last words' hyps in all sentences
    Y_inds = S_ind[:,1:] # remove the first words in all sentences
    if hierarchical:
      train_targets = self._factor_target_indices(Y_inds, base=base)
    else:
      train_targets = [self._make_one_hot(Y_inds, Y_inds.max() + 1)]
    length = Y_inds.shape[1]
    lstm_outdim = self.word_dim
    
    num_words = len(self.dp.word_index)
    num_syns = len(self.dp.synset_index)
    input = Input(shape=X.shape[1:], dtype='int32')
    embed_input_dim = num_syns if use_onto_lstm else num_words
    embed_layer = HigherOrderEmbedding(name='embedding', input_dim=embed_input_dim, output_dim=self.word_dim, input_shape=X.shape[1:], mask_zero=True)
    sent_rep = embed_layer(input)
    reg_sent_rep = Dropout(0.5)(sent_rep)
    if use_onto_lstm:
      lstm_out = OntoAttentionLSTM(name='sent_lstm', input_dim=self.word_dim, output_dim=lstm_outdim, input_length=length, num_senses=self.num_senses, num_hyps=self.num_hyps, return_sequences=True, use_attention=use_attention)(reg_sent_rep)
    else:
      lstm_out = LSTM(name='sent_lstm', input_dim=self.word_dim, output_dim=lstm_outdim, input_length=length, return_sequences=True)(reg_sent_rep)
    output_nodes = []
    # Make one node for each factored target
    for target in train_targets:
      node = TimeDistributed(Dense(input_dim=lstm_outdim, output_dim=target.shape[-1], activation='softmax'))(lstm_out)
      output_nodes.append(node)

    model = Model(input=input, output=output_nodes)
    print >>sys.stderr, model.summary()
    early_stopping = EarlyStopping()
    precompile_time = time.time()
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    postcompile_time = time.time()
    print >>sys.stderr, "Model compilation took %d s"%(postcompile_time - precompile_time)
    model.fit(X, train_targets, nb_epoch=num_epochs, validation_split=0.1, callbacks=[early_stopping])
    posttrain_time = time.time()
    print >>sys.stderr, "Training took %d s"%(posttrain_time - postcompile_time)
    concept_reps = model.layers[1].get_weights()
    self.model = model
    return concept_reps

  def test(self, vocab_size, use_onto_lstm, S_ind_test=None, C_ind_test=None, hierarchical=False, base=2):
    X_test = C_ind_test[:,:-1] if use_onto_lstm else S_ind_test[:,:-1] # remove the last words' hyps in all sentences
    Y_inds_test = S_ind_test[:,1:]
    if hierarchical:
      test_targets = self._factor_target_indices(Y_inds_test, vocab_size, base=base)
    else:
      test_targets = [self._make_one_hot(Y_inds_test, vocab_size)]
    print >>sys.stderr, "Evaluating model on test data"
    test_loss = self.model.evaluate(X_test, test_targets)
    print >>sys.stderr, "Test loss: %.4f"%test_loss
    factored_test_preds = [-((numpy.log(pred) * target).sum(axis=-1)) for pred, target in zip(self.model.predict(X_test), test_targets)]
    #test_preds = numpy.zeros_like(factored_test_preds[0])
    #for ftp in factored_test_preds:
    #  test_preds = numpy.copy(test_preds + ftp)
    test_preds = sum(factored_test_preds)
    #print [tp.shape for tp in test_preds]
    non_null_probs = []
    for test_pred, inds in zip(test_preds, Y_inds_test):
      wanted_probs = []
      for tp, ind in zip(test_pred, inds):
        if ind != 0:
          wanted_probs.append(tp)
      non_null_probs.append(wanted_probs)
    return non_null_probs

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
  argparser.add_argument('--train_file', type=str, help="One sentence per line, POS tagged")
  argparser.add_argument('--test_file', type=str, help="Test file for which attention values will be printed. One sentence per line, POS tagged")
  argparser.add_argument('--dim', type=int, help="Word/synset dimensionality", default=50)
  argparser.add_argument('--num_senses', type=int, help="Number of senses per word if using OntoLSTM (default 2)", default=2)
  argparser.add_argument('--num_hyps', type=int, help="Number of hypernyms per sense if using OntoLSTM (default 5)", default=5)
  argparser.add_argument('--use_onto_lstm', help="If this flag is not set, will use traditional LSTM", action='store_true')
  argparser.add_argument('--use_attention', help="Use attention in ontoLSTM. If this flag is not set, will use average concept representations", action='store_true')
  argparser.add_argument('--show_attention', help="Print attention values for sentences in the test file (printed in <testfilename>.att_out)", action='store_true')
  argparser.add_argument('--hierarchical', type=int, help="Use hierarchical softmax with the provided number of classes per factor")
  argparser.add_argument('--synset_embedding_output', type=str, help="Print learned synset representations in the given file")
  argparser.add_argument('--num_epochs', type=int, help="Number of epochs (default 20)", default=20)
  args = argparser.parse_args()
  if args.show_attention and (not args.use_attention or not args.use_onto_lstm):
    raise RuntimeError, "Use OntoLSTM with attention to print attention values of the test file"
  sm = SentenceModel(word_dim=args.dim, num_senses=args.num_senses, num_hyps=args.num_hyps)
  hierarchical = False
  base = 2
  if args.hierarchical is not None:
    hierarchical = True
    base = args.hierarchical 
  model_name_prefix = "sent_model_ontolstm=%s_att=%s_senses=%d_hyps=%d"%(str(args.use_onto_lstm), str(args.use_attention), args.num_senses, args.num_hyps)
  do_test = False
  do_train = False

  if args.train_file is not None:
    ts = [x.strip() for x in codecs.open(args.train_file, "r", "utf-8").readlines()]
    print >>sys.stderr, "Reading training data"
    S_ind, C_ind = sm.read_sentences(ts)
    _, train_sent_len, _, _ = C_ind.shape
    do_train = True
  else:
    print >>sys.stderr, "Loading stored model"
    sm.model = model_from_yaml(open("%s.yaml"%model_name_prefix).read(), custom_objects={"HigherOrderEmbedding": HigherOrderEmbedding, "OntoAttentionLSTM": OntoAttentionLSTM})
    print >>sys.stderr, sm.model.summary()
    sm.model.load_weights("%s.h5"%model_name_prefix)
    sm.model.compile(optimizer='adam', loss='categorical_crossentropy')
    dataproc_pkl_file = open("%s_dataproc.pkl"%model_name_prefix)
    sm.dp = pickle.load(dataproc_pkl_file)
    print sm.model.get_input_shape_at(0)
    train_sent_len = sm.model.get_input_shape_at(0)[1] + 1 #because the input shape will be one less than the sentence length

  if args.test_file is not None:
    print >>sys.stderr, "Reading test data"
    ts_test = [x.strip() for x in codecs.open(args.test_file, "r", "utf-8").readlines()]
    lock_index = False if do_train else True
    S_ind_test, C_ind_test = sm.read_sentences(ts_test, sentlenlimit=train_sent_len, test=lock_index)
    do_test = True

  if do_train:
    concept_reps = sm.train(S_ind, C_ind, use_onto_lstm=args.use_onto_lstm, use_attention=args.use_attention, num_epochs=args.num_epochs, hierarchical=hierarchical, base=base)
    model_yaml_string = sm.model.to_yaml()
    open("%s.yaml"%model_name_prefix, "w").write(model_yaml_string)
    sm.model.save_weights("%s.h5"%model_name_prefix, overwrite=True)
    dataproc_pkl_file = open("%s_dataproc.pkl"%model_name_prefix, "w")
    pickle.dump(sm.dp, dataproc_pkl_file)
    
  if do_test:
    vocab_size = len(sm.dp.word_index)
    test_probs = sm.test(vocab_size, args.use_onto_lstm, S_ind_test, C_ind_test, hierarchical, base)
    outfile = open("%s.out"%model_name_prefix, "w")
    for nnp in test_probs:
      print >>outfile, " ".join(["%.4f"%p for p in nnp])
    outfile.close()
  if args.synset_embedding_output is not None:
    concrepfile = open(args.synset_embedding_output, "w")
    for syn in sm.dp.synset_index:
      print >>concrepfile, syn, " ".join(["%s"%x for x in concept_reps[0][sm.dp.synset_index[syn]]])
  rev_synset_ind = {ind: syn for (syn, ind) in sm.dp.synset_index.items()}
  if args.show_attention:
    if args.test_file is None:
      raise RuntimeError, "Test file not provided!"
    outfile_name = args.test_file.split("/")[-1] + ".att_out"
    outfile = open(outfile_name, "w")
    C_att = sm.get_attention(C_ind_test)
    for i, (sent, sent_c_inds, sent_c_atts) in enumerate(zip(ts_test, C_ind_test, C_att)):
      print >>outfile, "SENT %d: %s"%(i, sent)
      words = sent.split()[:train_sent_len]
      word_id = 0
      for word_c_inds, word_c_atts in zip(sent_c_inds, sent_c_atts):
        if word_c_inds.sum() == 0:
          continue
        sense_id = 0
        #print >>outfile, "Attention for %s"%(words[word_id])
        best_sense = ""
        max_sense_prob = 0.0
        for s_h_ind, s_h_att in zip(word_c_inds, word_c_atts):
          if sum(s_h_ind) == 0:
            continue
          #print >>outfile, "\nSense %d"%(sense_id)
          sense_id += 1
          if s_h_att[-1] > max_sense_prob:
            max_sense_prob = s_h_att[-1]
            best_sense = rev_synset_ind[s_h_ind[-1]]
          #for h_ind, h_att in zip(s_h_ind, s_h_att):
          #  print >>outfile, rev_synset_ind[h_ind], h_att 
        print >>outfile, words[word_id], best_sense
        word_id += 1
        #print >>outfile
      print >>outfile
