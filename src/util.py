
import numpy as np
import phonetics
import Levenshtein

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

def stable_softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator

    return softmax


def create_similarity_matrix(tokenizer, *args):
    tokens = [tokenizer.cls_token]
    for words in args:
        tokens.extend(tokenizer.tokenize(words))
        tokens.append(tokenizer.sep_token)
    maximum = 0
    matrix = []
    for word in tokens:
        row = []
        for comparison in tokens:
            # don't want to scale attention for classifier tokens
            if (word in [tokenizer.cls_token, tokenizer.sep_token] or
                comparison in [
                tokenizer.cls_token, tokenizer.sep_token]
                ):
                row.append(0)
                continue
            dist = Levenshtein.distance(word, comparison)
            row.append(dist)
            if dist > maximum:
                maximum = dist
        matrix.append(row)

    nmatrix = []
    for row in matrix:
        nr = []
        for score in row:
            nr.append(maximum - score + 1)
        nmatrix.append(nr)

    return stable_softmax(np.array(nmatrix))

if __name__ == '__main__':
	tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
	test = ['I am a dog']
	print(create_similarity_matrix(tokenizer, *test))
