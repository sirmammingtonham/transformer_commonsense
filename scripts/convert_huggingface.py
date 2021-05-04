import sys
from datasets import Dataset

def dict_up(features):
    assert(len(features) == 7)
    return {
        features[0]: ids,
        features[1]: firsts,
        features[2]: seconds,
        features[3]: thirds,
        features[4]: fourths,
        features[5]: fifths,
        features[6]: indicies
    }        

CAT_FEATURES = ['story_id', 'first_sentence', 'second_sentence', 'third_sentence', 'fourth_sentence', 'fifth_sentence', 'cat_index']

PRI_FEATURES = ['story_id', 'first_sentence', 'second_sentence', 'third_sentence', 'fourth_sentence', 'fifth_sentence', 'sen_index']

ids = []
firsts = []
seconds = []
thirds = []
fourths = []
fifths = []
indicies = []

if sys.argv[1] == 'cat':
    FILE_PATH = '../baseline_texts/Full_labeled_category_consensus_dataset_for_baseline.txt'
elif sys.argv[1] == 'pri':
    FILE_PATH = '../baseline_texts/Full_labeled_primary_sentence_consensus_dataset_for_baseline.txt'


with open(FILE_PATH, 'r') as f:
    for row in f:
        elements = row.split(' | ')
        assert(len(elements) == 7)

        ids.append(elements[0])
        firsts.append(elements[1])
        seconds.append(elements[2])
        thirds.append(elements[3])
        fourths.append(elements[4])
        fifths.append(elements[5])
        if sys.argv[1] == 'cat':
            indicies.append(elements[6].replace('\n',''))
        elif sys.argv[1] == 'pri':
            indicies.append(elements[6].replace('\n','').split(','))

    if sys.argv[1] == 'cat':
        dataset = Dataset.from_dict(dict_up(CAT_FEATURES))
        dataset.set_format('numpy')
        dataset.save_to_disk('../baseline_data/category')

    elif sys.argv[1] == 'pri':
        dataset = Dataset.from_dict(dict_up(PRI_FEATURES))
        dataset.set_format('numpy')
        dataset.save_to_disk('../baseline_data/primary_sentence')
    
print(dataset)
import random

print(random.choices(dataset['sen_index'], k=10))