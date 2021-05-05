from datasets import load_from_disk
import random

reloaded_dataset = load_from_disk('../baseline_data/category')
print(reloaded_dataset.features)
print(reloaded_dataset)

print(random.choices(reloaded_dataset, k=5))
print('\n\n')

reloaded_dataset = load_from_disk('../baseline_data/primary_sentence')
print(reloaded_dataset.features)
print(reloaded_dataset)

print(random.choices(reloaded_dataset, k=5))