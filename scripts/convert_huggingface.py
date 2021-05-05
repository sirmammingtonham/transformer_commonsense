import sys
import datasets
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

CAT_FEATURES = ['story_id', 'first_sentence', 'second_sentence', 'third_sentence', 'fourth_sentence', 'fifth_sentence', 'category']

PRI_FEATURES = ['story_id', 'first_sentence', 'second_sentence', 'third_sentence', 'fourth_sentence', 'fifth_sentence', 'primary']

CATEGORIES = ['behavior_based', 'objective_based', 'emotional_based', 'goal_driven']

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
            indicies.append(int(elements[6].replace('\n','')))
        elif sys.argv[1] == 'pri':
            indicies.append([elements[int(i) + 1] for i in elements[6].replace('\n','').split(',')])

    if sys.argv[1] == 'cat':
        dataset = Dataset.from_dict(dict_up(CAT_FEATURES))
        feats = datasets.Features({
            'story_id': datasets.Value('string'), 
        	'first_sentence': datasets.Value('string'), 
            'second_sentence': datasets.Value('string'),
            'third_sentence': datasets.Value('string'),
            'fourth_sentence': datasets.Value('string'),
            'fifth_sentence': datasets.Value('string'),
            'category': datasets.ClassLabel(4, ['behavior_based', 'objective_based', 'emotional_based', 'goal_driven'])
        })
        dataset = dataset.cast(feats)
        print(dataset.features)
        dataset.set_format('numpy')
        dataset.save_to_disk('../baseline_data/category')

    elif sys.argv[1] == 'pri':
        dataset = Dataset.from_dict(dict_up(PRI_FEATURES))
        feats = datasets.Features({
            'story_id': datasets.Value('string'), 
            'first_sentence': datasets.Value('string'), 
            'second_sentence': datasets.Value('string'),
            'third_sentence': datasets.Value('string'),
            'fourth_sentence': datasets.Value('string'),
            'fifth_sentence': datasets.Value('string'),
            'primary': datasets.features.Sequence(datasets.Value('string'))
        })
        dataset = dataset.cast(feats)
        dataset.set_format('numpy')
        dataset.save_to_disk('../baseline_data/primary_sentence')