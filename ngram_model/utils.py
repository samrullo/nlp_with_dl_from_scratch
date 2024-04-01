from tqdm import tqdm
import re
import numpy as np


def create_cooccurrence_matrix(
    corpus: list[int], window_size: int, vocab_size: int
) -> np.ndarray:
    """
    Create co-occurrence matrix
    """
    cooccur_mat = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    for word_pos, word_id in enumerate(corpus):
        for context_idx in range(1, window_size + 1):
            left_word_pos = word_pos - context_idx
            right_word_pos = word_pos + context_idx

            if left_word_pos >= 0:
                left_word_id = corpus[left_word_pos]
                cooccur_mat[word_id, left_word_id] += 1

            if right_word_pos < len(corpus):
                right_word_id = corpus[right_word_pos]
                cooccur_mat[word_id, right_word_id] += 1
    return cooccur_mat


def tokenize(text: str,special_words:list[str]=None):
    """
    Split text on whitespace and punctuation marks and return tokens
    """
    if special_words:
        special_words_pattern="|".join(special_words)
        pattern=rf"{special_words_pattern}|\w+|[^\w\s]+"
    else:
        pattern=r"\w+|[^\w\s]+"
    return re.findall(pattern, text)


def build_vocabulary(words: list[str]):
    """
    Build vocabulary from list of words. This function simply applies set function on list of words, thus extracting distinct words from the list.
    """
    return set(words)


def get_word_to_id_from_vocab(vocab: list[str]):
    """
    Build word to id dictionary from vocabulary, which is a list of distinct words
    """
    return {word: idx for idx, word in enumerate(vocab)}


def get_id_to_word(word_to_id: dict):
    """
    Build id to word dictionary based on word_to_id dictionary
    """
    return {id: word for word, id in word_to_id.items()}


def build_corpus(words: list[str], word_to_id: dict[str, int]):
    """
    build corpus with word ids from words list and word_to_id dictionary
    """
    return [word_to_id[word] for word in words]


def build_ppmi(C: np.ndarray, verbose: bool = False):
    """
    build ppmi from cooccurrence matrix
    """
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    eps = 1e-8
    counter=0
    total_iterations=C.shape[0]*C.shape[1]
    pbar = tqdm(total=total_iterations,desc="PPMI building")

    for row_idx in range(C.shape[0]):
        for col_idx in range(C.shape[1]):
            pmi = np.log2((N * C[row_idx, col_idx]) / (S[row_idx] * S[col_idx]) + eps)
            ppmi = max(0, pmi)
            M[row_idx, col_idx] = ppmi
            
            pbar.update(1)
            if verbose:
                counter += 1
                if counter % 100 == 0:
                    print(f"finished processing {counter/(C.shape[0]*C.shape[1])*100:.2}% of cooccurrences")
    return M
