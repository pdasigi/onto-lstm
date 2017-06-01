# Ontology-Aware Token Embedding and LSTM Encoder
This repository contains a Keras implementation of WordNet grounded context sensitive token embeddings, described in the paper [Ontology-Aware Token Embeddings for Prepositional Phrase Attachment](https://arxiv.org/abs/1705.02925). Primarily, we implement two Keras layers here: `OntologyAwareEmbedding` and `OntoAttentionLSTM`, which (in most cases) will be used together.

## Background
The idea behind WordNet grounding is to represent words as a distribution over their senses and possible hypernyms from WordNet. That is, given a word **pool** as a noun, WordNet 3.1 identifies 9 senses of the word:
* `pool.n.01`: An excavation
* `pool.n.02`: A small lake
* `pool.n.03`: An organization of people or resources
* ...

For each of these senses, WordNet also defines hypernyms (generalizations) in order, like:
* `pool.n.01`: `excavation.n.01`, `artifact.n.01`, ...
* `pool.n.02`: `lake.n.01`, `body_of_water.n.01`, ...
* `pool.n.03`: `organization.n.01`, `social_group.n.01`, ...

We represent a token **pool** as a weighted average of the representations of a fixed number of its hyperyms, coming from a fixed number of its senses.

The weights in the weighted average depend on the context, thus making our word representations context-sensitive (or formally token-level representations instead of type-level). For example, the distribution of weights for the word **pool** in the following sentence:

_Swimming is not allowed in this **pool**._

is different from those in the following sentence:

_I managed to find a car **pool**._

The computation of this context sensitive weight distribution is tied to the computation that happens in the encoder (LSTM in our case).

## Installation

Clone this repository and run

```
pip install -r requirements.txt
```
The code is written in Python 2.7 and depends on Keras 1.2.1.

## Using OntoLSTM in your network

The input needs to be POS tagged. Look at `data/test_data.tsv` as an example. The file contains labels in the first column and POS-tagged sentences in the second. These sentences were tagged using [Stanford POS tagger](https://nlp.stanford.edu/software/tagger.shtml).

`test_ontolstm.py` shows a simple implementation of a binary classifier using OntoLSTM as the encoder. Try running the following to test your installation.

```
python test_ontolstm.py
```

## Reproducing PP Attachment results from ACL 2017 paper

### Set up the data
Download the train and test data created by Yonatan Belinkov, available [here](https://belinkov.mit.edu/data), and set them up in the same format as the sample test data, with labels indicating the index of the correct head words. The dataset comes with POS tags of head words, so you will not need a POS tagger for this. Concatenate the preposition and the dependent of the preposition at the end of the set of head words. Your train and test sets should have tab separated lines that look like this:
```
4  succeed_VB arbitrator_NN decision_NN made_VB in_PP dispute_NN
```
where `4` indicates the label (i.e. the 4th word, `made` is the head word to which the PP `in dispute` attaches to). Note that all verbs are indicated as `VB` and all nouns as `NN`. We do not follow the Penn Tree Bank POS tag set here. In fact, OntoLSTM needs a coarse distinction among nouns, verbs, adjectives and adverbs, and not more since those are the word classes available in WordNet. See the method `get_hypernyms_sentence` in `index_data.py` for the logic.

### Obtain synset embeddings
We got synset vectors by running AutoExtend on GloVe, and used these to initialize our synset embeddings.
You can download them [here](https://drive.google.com/drive/folders/0B6KTy_3y_sxXNXNIekVRWGtvVlE?usp=sharing). Alternatively, you can build them yourself:
Download [AutoExtend](http://www.cis.lmu.de/~sascha/AutoExtend/) and run it on [GloVe](https://nlp.stanford.edu/projects/glove/) vectors, to obtain synset vectors. We use the 100d vectors. Make sure to use WordNet 3.1 as the ontology for AutoExtend.


### Train and test OntoLSTM PP
Run
```
python2.7 model_pp_attachment.py --train_file TRAIN.tsv --test_file TEST.tsv --embedding_file autoextend_glove_100d.txt.gz --embed_dim 100 --tune_embedding --bidirectional --onto_aware --use_attention --num_senses 3 --num_hyps 5 --embedding_dropout 0.5 --encoder_dropout 0.2
```
That's it! This should also save the best trained model on your disk.

## Contact
If you use OntoLSTM in your work, I would love to hear about it. Please feel free to contact [me](mailto:pdasigi@cs.cmu.edu) if you run into any issues using it or even with general feedback.
