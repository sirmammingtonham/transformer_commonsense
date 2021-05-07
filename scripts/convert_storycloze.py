import csv
import datasets
from datasets import Dataset, load_dataset

FILE_PATHS = [r'storycloze\cloze_test_test__spring2016 - cloze_test_ALL_test - cloze_test_test__spring2016 - cloze_test_ALL_test.csv', r'storycloze\cloze_test_val__spring2016 - cloze_test_ALL_val - cloze_test_val__spring2016 - cloze_test_ALL_val.csv']

OUT_PATHS = [r'storycloze_test',r'storycloze_valid']

def convert(FILE_PATH, OUT_PATH):
    ids = []
    firsts = []
    seconds = []
    thirds = []
    fourths = []
    quiz1s = []
    quiz2s = []
    answers = []

    dataset = load_dataset('csv', data_files=FILE_PATH)['train']

    print(dataset)

    feats = datasets.Features({
        'InputStoryid': datasets.Value('string'),
        'InputSentence1': datasets.Value('string'),
        'InputSentence2': datasets.Value('string'),
        'InputSentence3': datasets.Value('string'),
        'InputSentence4': datasets.Value('string'),
        'RandomFifthSentenceQuiz1': datasets.Value('string'),
        'RandomFifthSentenceQuiz2': datasets.Value('string'),
        'AnswerRightEnding': datasets.ClassLabel(2, ['quiz1', 'quiz2'])
    })
    dataset = dataset.cast(feats)
    print(dataset.features)
    dataset.set_format('numpy')
    dataset.save_to_disk(OUT_PATH)

for i in range(len(FILE_PATHS)):
    convert(FILE_PATHS[i], OUT_PATHS[i])