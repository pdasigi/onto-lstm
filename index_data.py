import re
from nltk.corpus import wordnet as wn

class DataProcessor(object):
  def __init__(self, word_syn_cutoff=2, syn_path_cutoff=5):
    self.word_syn_cutoff = word_syn_cutoff
    self.syn_path_cutoff = syn_path_cutoff

    self.thing_prons = ['it', 'which', 'that', 'this', 'what', 'these', 'itself', 'something', 'anything', 'everything'] # thing
    self.male_prons = ['he', 'him', 'himself'] # man.n.01
    self.female_prons = ['she', 'her', 'herself'] # woman.n.01
    self.people_prons = ['they', 'them', 'themselves', 'we', 'ourselves', 'yourselves'] # people.n.01, people.n.03
    self.person_prons = ['you', 'i', 'who', 'whom', 'whoever', 'anyone', 'everyone', 'myself', 'yourself'] # person.n.01
    self.word_hypernym_map = {}
    self.word_index = {"NONE": 0, "UNK": 1}
    self.synset_index = {"NONE": 0, "UNK": 1}
    self.word_singletons = set([])
    self.word_non_singletons = set([])
    self.conc_singletons = set([])
    self.conc_non_singletons = set([])
    
  def get_hypernyms_syn(self, syn, path_cutoff=None):
    if not path_cutoff:
      path_cutoff = self.syn_path_cutoff
    hypernyms = []
    # Select shortest hypernym path
    all_paths = syn.hypernym_paths()
    shortest_path = all_paths[0]
    for path in all_paths:
      if len(shortest_path) > len(path):
        shortest_path = path
    pruned_path = list(shortest_path) if path_cutoff == -1 or path_cutoff >= len(shortest_path) else [x for x in reversed(shortest_path)][:path_cutoff]
    hypernyms = [s.name() for s in pruned_path]
    return hypernyms

  def get_hypernyms_word(self, word, pos, syn_cutoff=-1):
    wrd_lower = word.lower()
    if not pos:
      syns = []
    else:
      syns = wn.synsets(wrd_lower, pos=pos)
    hypernyms = []
    if wrd_lower in self.thing_prons:
      syns += wn.synsets("thing", "n")
    elif wrd_lower in self.male_prons:
      syns += wn.synsets("man", "n")
    elif wrd_lower in self.female_prons:
      syns += wn.synsets("woman", "n")
    elif wrd_lower in self.people_prons:
      syns += wn.synsets("people", "n")
    elif wrd_lower in self.person_prons:
      syns += wn.synsets("person", "n")
    elif re.match('^[12][0-9]{3}$', word) is not None:
      # The argument looks like an year
      syns += wn.synsets("year", "n") + wn.synsets("number", "n")
    elif re.match('^[0-9]+[.,-]?[0-9]*', word) is not None:
      syns += wn.synsets("number", "n")
    if len(hypernyms) == 0:
      if len(syns) != 0:
        pruned_synsets = list(syns) if self.word_syn_cutoff == -1 else syns[:self.word_syn_cutoff]
        for syn in pruned_synsets:
          hypernyms.append(self.get_hypernyms_syn(syn))
      else:
        hypernyms = [[word]]
    return hypernyms

  def index_sentence(self, words, pos_tags, test=False, remove_singletons=False):
    word_inds = []
    conc_inds = []
    wn_pos_tags = []
    for pos_tag in pos_tags:
      if pos_tag.startswith("J"):
        wn_pos = "a"
      elif pos_tag.startswith("V"):
        wn_pos = "v"
      elif pos_tag.startswith("N"):
        wn_pos = "n"
      elif pos_tag.startswith("R"):
        wn_pos = "r"
      else:
        wn_pos = None
      wn_pos_tags.append(wn_pos)
    for word, wn_pos in zip(words, wn_pos_tags):
      word = word.lower()
      if (word, wn_pos) in self.word_hypernym_map:
        word_hyps = self.word_hypernym_map[(word, wn_pos)]
      else:
        word_hyps = self.get_hypernyms_word(word, wn_pos)
        self.word_hypernym_map[(word, wn_pos)] = word_hyps
      # Add to singletons or non-singletons
      if word not in self.word_non_singletons:
        if word in self.word_singletons:
          self.word_singletons.remove(word)
          self.word_non_singletons.add(word)
        else:
          self.word_singletons.add(word)
      for sense_syns in word_hyps:
        for syn in sense_syns:
          if syn not in self.conc_non_singletons:
            if syn in self.conc_singletons:
              self.conc_singletons.remove(syn)
              self.conc_non_singletons.add(syn)
            else:
              self.conc_singletons.add(syn)
    for word, wn_pos in zip(words, wn_pos_tags):
      word_conc_inds = []
      if word not in self.word_index and not test:
        if remove_singletons and word in self.word_singletons:
          self.word_index[word] = self.word_index['UNK']
        else:
          self.word_index[word] = len(self.word_index)
      word_hyps = self.word_hypernym_map[(word, wn_pos)]
      for sense_syns in word_hyps:
        word_sense_conc_inds = []
        # Most specific concept will be at the end
        for syn in reversed(sense_syns):
          if syn not in self.synset_index and not test:
            if remove_singletons and syn in self.conc_singletons:
              self.synset_index[syn] = self.synset_index['UNK']
            else:
              self.synset_index[syn] = len(self.synset_index)
          conc_ind = self.synset_index[syn] if syn in self.synset_index else self.synset_index['UNK']
          word_sense_conc_inds.append(conc_ind)
        word_conc_inds.append(word_sense_conc_inds)
      word_ind = self.word_index[word] if word in self.word_index else self.word_index['UNK']
      word_inds.append(word_ind)
      conc_inds.append(word_conc_inds)
    return word_inds, conc_inds
