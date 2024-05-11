from typing import List
import numpy as np

def convert_one_hot(corpus:np.ndarray, vocab_size:int):
    """
    Convert IDs to one-hot vectors. Input can either be one dimensional or two dimensional numpy array of integers
    """
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N,vocab_size), dtype=np.int32)
        for i, word_id in enumerate(corpus):
            one_hot[i,word_id] = 1
    elif corpus.ndim == 2:
        # data has (N, C) shape where N is batch size, C is the context size
        C = corpus.shape[1]
        one_hot = np.zeros((N,C,vocab_size), dtype=np.int32)

        for idx_0, subcorpus in enumerate(corpus):
            for idx_1, word_id in enumerate(subcorpus):
                one_hot[idx_0, idx_1, word_id] = 1
    return one_hot

def create_contexts_and_targets(corpus:List[int], window_size:int):
    """
    Prepare targets and contexts from a corpus
    """
    targets = corpus[window_size : -window_size]
    contexts = [ [corpus[i+t] for t in range(-window_size, window_size+1) if t != 0] for i in range(window_size, len(corpus)-window_size)]
    return contexts, targets

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate
