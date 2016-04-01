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
    self.word_index = {"NONE": 0}
    self.synset_index = {"NONE": 0}
    
    #self.man_hypernyms = self.get_hypernyms_syn(wn.synset('man.n.01'))
    #self.woman_hypernyms = self.get_hypernyms_syn(wn.synset('woman.n.01'))
    #self.people_hypernyms = self.get_hypernyms_syn(wn.synset('people.n.01')).union(self.get_hypernyms_syn(wn.synset('people.n.03')))
    #self.loc_hypernyms = self.get_hypernyms_syn(wn.synset('geographical_area.n.01'))
    #self.person_hypernyms = self.get_hypernyms_syn(wn.synset('person.n.01'))
    #self.year_hypernyms = self.get_hypernyms_syn(wn.synset('year.n.01'))
    #self.number_hypernyms = self.get_hypernyms_syn(wn.synset('number.n.01'))

    #self.misc_hypernyms = set(self.loc_hypernyms).union(self.person_hypernyms)

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
    return set(hypernyms)

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
      #hypernyms = list(self.man_hypernyms)
    elif wrd_lower in self.female_prons:
      syns += wn.synsets("woman", "n")
      #hypernyms = list(self.female_prons)
    elif wrd_lower in self.people_prons:
      syns += wn.synsets("people", "n")
      #hypernyms = list(self.people_hypernyms)
    elif wrd_lower in self.person_prons:
      syns += wn.synsets("person", "n")
      #hypernyms = list(self.person_hypernyms)
    elif re.match('^[12][0-9]{3}$', word) is not None:
      # The argument looks like an year
      syns += wn.synsets("year", "n") + wn.synsets("number", "n")
      #hypernyms = list(self.year_hypernyms)
    elif re.match('^[0-9]+[.,-]?[0-9]*', word) is not None:
      syns += wn.synsets("number", "n")
      #hypernyms = list(self.number_hypernyms)
    #elif word[0].isupper():
    #  hypernyms = list(self.misc_hypernyms)
    if len(hypernyms) == 0:
      if len(syns) != 0:
        pruned_synsets = list(syns) if self.word_syn_cutoff == -1 else syns[:self.word_syn_cutoff]
        for syn in pruned_synsets:
          hypernyms.extend(list(self.get_hypernyms_syn(syn)))
      else:
        hypernyms = [word]
    return set(hypernyms)

  def index_sentence(self, words, pos_tags):
    #words = word_tokenize(sentence)
    word_inds = []
    conc_inds = []
    for word, pos_tag in zip(words, pos_tags):
      word = word.lower()
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
      word_conc_inds = []
      if word not in self.word_index:
        self.word_index[word] = len(self.word_index)
      if (word, wn_pos) in self.word_hypernym_map:
        word_hyps = self.word_hypernym_map[(word, wn_pos)]
      else:
        word_hyps = self.get_hypernyms_word(word, wn_pos)
        self.word_hypernym_map[(word, wn_pos)] = word_hyps
      for syn in word_hyps:
        if syn not in self.synset_index:
          self.synset_index[syn] = len(self.synset_index)
        word_conc_inds.append(self.synset_index[syn])
      word_inds.append(self.word_index[word])
      conc_inds.append(word_conc_inds)
    return word_inds, conc_inds
