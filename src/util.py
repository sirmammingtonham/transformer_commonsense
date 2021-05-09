
import numpy as np
from pyphonetics import Soundex, RefinedSoundex, Metaphone
from tqdm import tqdm

from typing import Dict
from transformers.tokenization_utils_base import BatchEncoding

import torch

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

def default_data_collator(features) -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, (int, np.integer)) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


if __name__ == '__main__':
    from transformers import AutoTokenizer
    # test that everything works
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    test = [['I am a dog', 'You aren\'t a cat'], ['bruh', 'yee']]
    sim = create_similarity_matrix(tokenizer, *test)
    print(sim)
    print([x.shape for x in sim])
