from datasets import load_from_disk
import random


def get_avg_len(dataset):
    count = 0
    for row in dataset:
        s = row['first_sentence'] + ' ' + row['second_sentence'] + ' ' + \
            row['third_sentence'] + ' ' + \
            row['fourth_sentence'] + ' ' + row['fifth_sentence']
        count += len(s.split(' '))
    return count/len(dataset)


reloaded_dataset = load_from_disk('../baseline_data/category')
print(reloaded_dataset.features)
print(get_avg_len(reloaded_dataset))
# print(reloaded_dataset)
# print(random.choices(reloaded_dataset, k=5))
# print('\n\n')

reloaded_dataset = load_from_disk('../baseline_data/importance')
print(reloaded_dataset.features)
print(get_avg_len(reloaded_dataset))
# print(reloaded_dataset)
# print(random.choices(reloaded_dataset, k=5))
