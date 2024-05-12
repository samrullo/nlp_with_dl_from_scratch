
import sys

sys.path.append('..')

import torch
import torch.nn as nn
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


class MatMulPT(nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim))
    
    def forward(self,x):
        return torch.mm(x, self.weights)

contexts_tn = torch.tensor(contexts, dtype=torch.float32)
targets_tn = torch.tensor(targets, dtype=torch.float32)

in_layer = MatMulPT(vocab_size,hidden_size)
out_layer = MatMulPT(hidden_size,vocab_size)

h0 = in_layer(contexts_tn[0,0,:].unsqueeze(0))
h1 = in_layer(contexts_tn[0,1,:].unsqueeze(0))
h = (h0 + h1)/2

scores = out_layer(h)
one_target = targets_tn[0].unsqueeze(0)
loss = torch.nn.functional.cross_entropy(scores,one_target)
print(f"loss is {loss}")