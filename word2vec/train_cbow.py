import sys
sys.append('..')

from ngram_model.utils import build_corpus, get_id_to_word, get_word_to_id_from_vocab,build_vocabulary
from word2vec.simple_cbow import SimpleCBOW
from common.trainers import Trainer
from common.optimizers import Adam
from common.utils import create_contexts_and_targets, convert_one_hot

window_size = 1
hidden_size = 5
batch_size=3
max_epoch=1000

text="Prince loved princess from the bottom of his heart"


