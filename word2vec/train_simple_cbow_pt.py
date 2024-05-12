import sys

sys.path.append('..')

import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from ngram_model.utils import build_corpus, get_id_to_word, get_word_to_id_from_vocab, build_vocabulary
from word2vec.simple_cbow import SimpleCBOW
from common.trainers import Trainer
from common.optimizers import Adam
from common.utils import create_contexts_and_targets, convert_one_hot
from ngram_model.utils import tokenize, build_vocabulary, get_word_to_id_from_vocab, get_id_to_word, build_corpus
from word2vec.simple_cbow_pytorch import compute_loss, training_loop,SimpleCBOWPT, CBOWDataset,plot_losses,loss_fn

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

contexts = torch.tensor(contexts, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.long)

cbow_dataset = CBOWDataset(contexts, targets)
cbow_dataloader = DataLoader(cbow_dataset, batch_size, shuffle=True)

cbow_model = SimpleCBOWPT(vocab_size, hidden_size)
optimizer = optim.Adam(cbow_model.parameters(),lr=0.01)
losses = training_loop(max_epoch, optimizer, cbow_model, loss_fn, cbow_dataloader, cbow_dataloader, pathlib.Path.cwd()/"checkpoint")
plot_losses(losses)