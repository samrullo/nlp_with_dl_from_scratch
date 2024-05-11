import sys

sys.path.append('..')

import numpy as np
from ngram_model.utils import build_corpus, get_id_to_word, get_word_to_id_from_vocab, build_vocabulary
from word2vec.simple_cbow import SimpleCBOW
from common.trainers import Trainer
from common.optimizers import Adam
from common.utils import create_contexts_and_targets, convert_one_hot
from ngram_model.utils import tokenize, build_vocabulary, get_word_to_id_from_vocab, get_id_to_word, build_corpus

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 100

text = "Prince loved princess from the bottom of his heart"

words = tokenize(text)
vocab = build_vocabulary(words)
word_to_id = get_word_to_id_from_vocab(vocab)
id_to_word = get_id_to_word(word_to_id)
corpus = build_corpus(words, word_to_id)
vocab_size = len(word_to_id)

contexts, targets = create_contexts_and_targets(corpus, window_size)
contexts = convert_one_hot(np.array(contexts), vocab_size)
targets = convert_one_hot(np.array(targets), vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)
trainer.fit(contexts, targets, max_epoch, batch_size)
trainer.plot()
