from datasets import load_from_disk

reloaded_dataset = load_from_disk('../baseline_data/category')
print(reloaded_dataset.features)
print(reloaded_dataset)

reloaded_dataset = load_from_disk('../baseline_data/primary_sentence')
print(reloaded_dataset.features)
print(reloaded_dataset)