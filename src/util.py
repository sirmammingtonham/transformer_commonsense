
import numpy as np
from pyphonetics import Soundex, RefinedSoundex, Metaphone
from tqdm import tqdm

phonetic_algs = {
    'S': Soundex(),
    'RS': RefinedSoundex(),
    'META': Metaphone()
}


def stable_softmax(x):
    z = x - np.max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax


def create_similarity_matrix(tokenizer, *args, pad_length=128, algorithm='META', metric='levenshtein'):
    IGNORE_TOKENS = tokenizer.all_special_tokens + ['']
    ALG = phonetic_algs[algorithm]

    # concatenate sequences if we're doing batches
    sequences = [' '.join(x) for x in zip(*args)] if len(args) > 1 else [x for x in args[0]]
    
    # break sequences into word piece tokens
    batch = [[tokenizer.cls_token] + [x.strip('##') for x in tokenizer.tokenize(
        sequence)] + [tokenizer.sep_token] for sequence in sequences]
    batch = [x[:pad_length] + ['']*(pad_length-len(x)) for x in batch] # pad to max length

    # calculate similarity matrix (i love list comprehension)
    # ignores special tokens
    similarities = [
        [
            [
                ALG.distance(word, comparison, metric=metric)
                if (word not in IGNORE_TOKENS and
                    comparison not in IGNORE_TOKENS)
                else 0
                for comparison in tokens
            ]
            for word in tokens
        ] for tokens in tqdm(batch, leave=False)
    ]

    # scale similarity matrix so highest similarity = highest value (and make min = 1)
    maxes = [np.max(x) for x in similarities]

    similarities = [maxes[i] - similarities[i] + 1 for i in range(len(similarities))] #np.max(similarities) - similarities + 1

    # return softmax of matrix
    return [stable_softmax(x).astype(float) for x in similarities]


if __name__ == '__main__':
    from transformers import AutoTokenizer
    # test that everything works
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    test = [['I am a dog', 'You aren\'t a cat'], ['bruh', 'yee']]
    sim = create_similarity_matrix(tokenizer, *test)
    print(sim)
    print([x.shape for x in sim])
